# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Rapid cooling of a steel plate
#
# This demo solves the cooling of a hot steel plate whose faces are held at a
# lower temperature, a transient problem in physical units. It shows how to:
#
# - Give a transient model units, check them, and find its dimensionless
#   groups, which involve a reference time, see
#   {doc}`../docs/tutorials/03_units`
# - Step the model in time in dimensionless form, with a
#   {py:class}`Simulation <sumfem.solvers.Simulation>` and a scheme that
#   discretizes the time derivative for you, see
#   {doc}`../docs/tutorials/04_time_stepping`
# - Validate the solution against the series solution of the problem, and
#   observe the order of the scheme
#
# ## The model
#
# A plate of thickness $L$, of volumetric heat capacity $\rho c$ and thermal
# conductivity $k$, starts at the temperature $T_0$. At $t = 0$ both faces are
# brought to the temperature $T_s$ and held there. Across the plate, the
# temperature solves the heat equation
#
# $$
# \begin{aligned}
#   \rho c \frac{\partial T}{\partial t}
#   - \nabla \cdot (k \nabla T) &= 0 \quad \text{in } (0, L), \\
#   T &= T_s \quad \text{at } x = 0 \text{ and } x = L, \\
#   T &= T_0 \quad \text{at } t = 0.
# \end{aligned}
# $$ (cooling)
#
# Its solution is a Fourier sine series {cite}`CarslawJaeger1959` (Chapter
# III), whose modes decay on the conduction time $\tau$ of the plate,
#
# $$
# \begin{aligned}
#   T(x, t) &= T_s + (T_0 - T_s) \sum_{n \text{ odd}} \frac{4}{n \pi}
#   \sin\left(\frac{n \pi x}{L}\right) e^{-n^2 \pi^2 t / \tau}, \\
#   \tau &= \frac{\rho c L^2}{k}.
# \end{aligned}
# $$ (cooling-series)
#
# We import the relevant packages:

# + tags=["hide-input"]
from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import sympy.physics.units as syu
import ufl

from sumfem import discrete, solvers, symbolic
from sumfem.models import Accumulation, Diffusion

# -

# ## The plate
#
# The plate is an interval of quadratic elements, in the precision of PETSc's
# real type, with its two faces marked. The symbolic domain has the
# thickness as its length scale, see {ref}`units-length-scale`:

# +
thickness = 5e-2
dtype = dolfinx.default_real_type
interval = dolfinx.mesh.create_interval(
    MPI.COMM_WORLD, 100, [0.0, thickness], dtype=dtype
)
tolerance = 100 * float(np.finfo(dtype).eps) * thickness
marked = discrete.MeshData.from_locators(
    interval,
    {
        0: {
            "Faces": lambda x: (
                np.isclose(x[0], 0.0, atol=tolerance)
                | np.isclose(x[0], thickness, atol=tolerance)
            )
        }
    },
    boundary=True,
)
plate = symbolic.Domain.from_mesh("plate", interval, length_scale=thickness)
# -

# The properties of steel are parameters with units, and the temperature is a
# field in kelvin, whose scale is the drop of temperature that the cooling
# imposes:

# +
K, m, J, W = syu.kelvin, syu.meter, syu.joule, syu.watt
initial, surface = 1100.0, 300.0
heat_capacity = symbolic.Parameter(
    "rho_c",
    plate,
    default=7850.0 * 490.0,
    unit=J / (m**3 * K),
    doc="Volumetric heat capacity",
)
conductivity = symbolic.Parameter(
    "k", plate, default=45.0, unit=W / (m * K), doc="Thermal conductivity"
)
T = symbolic.Field("T", plate, ("Lagrange", 2), unit=K, scale=initial - surface)
# -

# The model of {eq}`cooling` is the stored heat,
# {py:class}`Accumulation <sumfem.models.Accumulation>` weighted with the heat
# capacity, and conduction, {py:class}`Diffusion <sumfem.models.Diffusion>`
# with the conductivity:

model = Accumulation("storage", T, heat_capacity) + Diffusion(
    "conduction", T, conductivity
)

# ## Units and dimensionless groups
#
# The model is semi-discrete: its time derivative is the placeholder `T.dot`,
# which the analysis measures against a reference time, as the domain against
# its length scale. Its units are checked on the system of the model alone,
# see {ref}`units-checking`. The conduction time $\tau$ of
# {eq}`cooling-series` is the natural reference time:

# +
tau = float(heat_capacity.default_si * thickness**2 / conductivity.default_si)
semi_discrete = symbolic.build_system(model, check_units=True, time_scale=tau)
repeating = [plate, T, heat_capacity, conductivity]
# -

# With the thickness, the temperature scale and the two properties as the
# repeating quantities, {py:func}`pi_groups <sumfem.symbolic.pi_groups>`
# finds one group, of the reference time. It is the Fourier number
# $k t_{\text{ref}} / (\rho c L^2)$, the reference time over the conduction
# time, which is one for the conduction time itself:

(fourier,) = symbolic.pi_groups(semi_discrete, time_scale=tau, repeating=repeating)
print(fourier)
assert np.isclose(fourier.value, 1.0)

# ## Stepping in dimensionless form
#
# The plate is followed over a part of the conduction time, in a number of
# time steps that resolves it. The Fourier number of a time step is the group
# with the time step as the reference time, the part of the conduction time
# one time step takes:

# +
t_end, num_steps = 0.1 * tau, 100
dt = t_end / num_steps
(step_fourier,) = symbolic.pi_groups(semi_discrete, time_scale=dt, repeating=repeating)
print(step_fourier)
assert np.isclose(step_fourier.value, dt / tau)
# -

# A {py:class}`Simulation <sumfem.solvers.Simulation>` replaces the time
# derivative with the second-order backward differentiation formula,
# {py:class}`BDF <sumfem.solvers.BDF>`, which turns the model into the
# equation of one time step. It checks the units of that equation and divides
# it by its typical size, see {ref}`units-scaling`, once, and then solves the
# dimensionless equation with Newton's method at every time step. By default
# the typical size measures the time derivative against the clock's time
# step, so that the change of the stored heat over one time step, the
# dominant term, is of order one; its `time_scale` argument sets another
# reference time. The temperature is stored in kelvin all along. The function
# below runs the cooling with a given number of time steps, with the equation
# scaled or not, and records the temperature at given time steps:


# +
def cool(
    num_steps: int, factorize: bool = True, record: tuple[int, ...] = ()
) -> tuple[dolfinx.fem.Function, dict[int, np.ndarray], solvers.Simulation]:
    clock = symbolic.Clock(plate, dt=t_end / num_steps)
    disc = discrete.discretize([model, clock], discrete.DomainMaps({plate: marked}))
    simulation = solvers.Simulation(
        disc,
        model,
        [symbolic.DirichletBC(T, "Faces", surface)],
        clock=clock,
        scheme=solvers.BDF(2),
        check_units=True,
        factorize_equations=factorize,
    )
    disc.set({T: initial})
    profiles: dict[int, np.ndarray] = {}

    def keep(t: float, step: int) -> None:
        if step in record:
            profiles[step] = disc.function(T).x.array.copy()

    simulation.run(num_steps=num_steps, callback=keep)
    return disc.function(T), profiles, simulation


record = (num_steps // 10, num_steps // 2, num_steps)
temperature, profiles, simulation = cool(num_steps, record=record)
print("Equation scale:", simulation.solver.system.symbolic.equation_scales)
# -

# Scaling the equations changes neither the solution nor its units: the run
# without scaling gives the same temperature, to round-off:

unscaled, _, _ = cool(num_steps, factorize=False)
difference = np.abs(unscaled.x.array - temperature.x.array).max()
print(f"Largest difference without scaling: {difference:.1e} K")
assert difference < 1e-9 * (initial - surface)

# (cooling-validation)=
# ## Validation
#
# The series {eq}`cooling-series` converges fast once the plate has started to
# cool. It is summed to a number of terms far beyond what the mesh resolves:


# +
def series(x: np.ndarray, t: float, terms: int = 2001) -> np.ndarray:
    n = np.arange(1, terms + 1, 2)[:, None]
    modes = np.sin(n * np.pi * x[None, :] / thickness) * np.exp(
        -((n * np.pi) ** 2) * t / tau
    )
    return surface + (initial - surface) * np.sum(4 / (n * np.pi) * modes, axis=0)


# -

# The temperature is compared with it in the relative $L^2$ norm over the
# plate, with the series interpolated in a degree above that of the solution:


# +
def l2_error(temperature: dolfinx.fem.Function, t: float) -> float:
    mesh = temperature.function_space.mesh
    space = dolfinx.fem.functionspace(mesh, ("Lagrange", 4))
    exact = dolfinx.fem.Function(space)
    exact.interpolate(lambda x: series(x[0], t))
    form = dolfinx.fem.form((temperature - exact) ** 2 * ufl.dx)
    squared = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
    return np.sqrt(squared / thickness) / (initial - surface)


error = l2_error(temperature, t_end)
print(f"Relative error: {error:.1e}")
assert error < 1e-4
# -

# The scheme is of second order: halving the time step divides the error by
# four, so the observed rate approaches two:

errors = np.array([l2_error(cool(n)[0], t_end) for n in (25, 50, 100)])
rates = np.log2(errors[:-1] / errors[1:])
print(f"Errors {errors}, rates {rates.round(2)}")
assert np.all(rates > 1.85)

# Across the plate, the temperature at the recorded time steps and the series,
# with the values each process owns gathered on the first:

# +
space = temperature.function_space
owned = space.dofmap.index_map.size_local
comm = space.mesh.comm
x = comm.gather(space.tabulate_dof_coordinates()[:owned, 0], root=0)
gathered = {
    step: comm.gather(values[:owned], root=0) for step, values in profiles.items()
}
if comm.rank == 0:
    x = np.concatenate(x)
    order = np.argsort(x)
    x_plot = np.linspace(0.0, thickness, 400)
    for step, values in gathered.items():
        t = step * dt
        (line,) = plt.plot(1e3 * x_plot, series(x_plot, t), "-")
        plt.plot(
            1e3 * x[order],
            np.concatenate(values)[order],
            ".",
            color=line.get_color(),
            label=f"t = {t:.1f} s",
        )
    plt.xlabel("x (mm)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.show()
# -

# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```
