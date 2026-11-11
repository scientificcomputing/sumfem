"""Meshes and the original RFBniCS forms, for comparison with the models.

The reference residuals are transcribed from the examples of RFBniCS
(https://github.com/Ah-Aghabarari/RFBniCS, MIT licence; Aghabarari et al.,
2026, https://arxiv.org/abs/2610.01840), with one change:
each unknown is its own function, where RFBniCS splits one blocked
function. The tests pass the functions of a discretization, so both sides
assemble on the same spaces.

The tests use pytest's importlib import mode, so these are handed to tests
through the ``reference`` fixture instead of being imported.
"""

import types

from mpi4py import MPI

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import pytest
import sympy.physics.units as syu
import ufl

from sumfem.discrete import MeshData
from sumfem.electrochemistry import Ion, PorousElectrode, RedoxCouple, RedoxHalfCell
from sumfem.symbolic import Domain

W = 2.5e-3
"""Width of the electrode in the examples, in m."""

H = 1e-2
"""Height of the electrode in the 2D example, in m."""

PARAMETERS = {
    "sigma": 3402.1,
    "a": 2.3e4,
    "alpha_a": 0.5,
    "alpha_c": 0.5,
    "k0": 2.5e-7,
    "E_formal": -0.36,
    "df": 0.5e-6,
    "F": 96485.33212,
    "R": 8.314462618,
    "T": 298.15,
    "D_V2": 8.10e-12,
    "D_V3": 1.65e-11,
    "D_H": 9.312e-9,
    "D_HSO4": 1.33e-9,
    "D_SO4": 1.065e-9,
    "c0_V2": 300.0,
    "c0_V3": 300.0,
    "c0_H": 1000.0,
    "c0_HSO4": 1000.0,
    "z_V2": 2.0,
    "z_V3": 3.0,
    "z_H": 1.0,
    "z_HSO4": -1.0,
    "z_SO4": -2.0,
    "mu": 6.963e-3,
    "rho": 1399.8,
    "Kc": 5.55,
    "epsilon": 0.94,
    "u_in": 1.0e-7,
    "p_out": 3.0e5,
    "Lp": 100.0e-6,
}
"""Default parameters of the RFBniCS examples."""

QUADRATURE_DEGREE = 4
"""The quadrature degree of the examples for degree-one elements."""


def interval(num_cells: int = 20) -> MeshData:
    """The 1D example's electrode, with its current collector and membrane."""
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_cells, [0.0, W])
    return MeshData.from_locators(
        mesh,
        {
            0: {
                "CurrentCollector": lambda x: np.isclose(x[0], 0.0),
                "Membrane": lambda x: np.isclose(x[0], W),
            }
        },
        boundary=True,
    )


def rectangle(nx: int = 5, ny: int = 8) -> MeshData:
    """The 2D example's electrode, with flow from the inlet at the bottom."""
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0.0, 0.0], [W, H]]),
        [nx, ny],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    return MeshData.from_locators(
        mesh,
        {
            1: {
                "CurrentCollector": lambda x: np.isclose(x[0], 0.0),
                "Membrane": lambda x: np.isclose(x[0], W),
                "Inlet": lambda x: np.isclose(x[1], 0.0),
                "Outlet": lambda x: np.isclose(x[1], H),
            }
        },
        boundary=True,
    )


def box(n: int = 2) -> MeshData:
    """A unit cube with an electrode and two channels, as in the 3D example.

    The electrode fills ``x > 1/2``. The inlet and outlet channels fill the
    rest, below and above ``y = 1/2``, and open at ``x = 0``.
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2 * n, 2 * n, 2 * n)
    half = 0.5 + 1e-10
    cells = MeshData.from_locators(
        mesh,
        {
            3: {
                "InletChannel": lambda x: (x[0] <= half) & (x[1] <= half),
                "OutletChannel": lambda x: (x[0] <= half) & (x[1] >= 0.5 - 1e-10),
                "Electrode": lambda x: x[0] >= 0.5 - 1e-10,
            }
        },
    )
    sides = {
        "Inlet": lambda x: np.isclose(x[0], 0.0) & (x[1] <= half),
        "Outlet": lambda x: np.isclose(x[0], 0.0) & (x[1] >= 0.5 - 1e-10),
        "CurrentCollector": lambda x: np.isclose(x[2], 0.0) & (x[0] >= 0.5 - 1e-10),
        "Membrane": lambda x: np.isclose(x[0], 1.0),
    }

    def walls(x):
        """The rest of the boundary: markers of facets do not overlap."""
        return ~np.logical_or.reduce([side(x) for side in sides.values()])

    facets = MeshData.from_locators(mesh, {2: {"Walls": walls, **sides}}, boundary=True)
    return MeshData(
        mesh, {**cells.tags, **facets.tags}, {**cells.markers, **facets.markers}
    )


def measures(marked: MeshData, degree: int | None = QUADRATURE_DEGREE):
    """Cell and facet measures of a marked mesh, and a lookup of marker ids."""
    mesh = marked.mesh
    tdim = mesh.topology.dim
    metadata = None if degree is None else {"quadrature_degree": degree}
    dx = ufl.Measure(
        "dx", domain=mesh, subdomain_data=marked.tags.get(tdim), metadata=metadata
    )
    ds = ufl.Measure(
        "ds", domain=mesh, subdomain_data=marked.tags.get(tdim - 1), metadata=metadata
    )

    def ids(dim, *names):
        return tuple(marked.markers[dim][name][0] for name in names)

    return dx, ds, ids


def _constants(mesh, values):
    return {
        name: dolfinx.fem.Constant(mesh, np.float64(value))
        for name, value in values.items()
    }


def electrochemistry(
    marked: MeshData,
    u,
    u_old,
    v,
    dt,
    current,
    *,
    velocity=None,
    regions: bool = False,
    area_ratio: float = 1.0,
    membrane_flux: bool = False,
    outlet: bool = False,
):
    """The residual of the RFBniCS examples, in one form.

    Without a velocity it is the 1D example's, with a velocity the 2D
    example's, and with regions the 3D example's, where the electrode and
    channel cells differ.

    Args:
        marked: The mesh.
        u: Functions of phi_s, phi_l, c_V2, c_V3, c_H and c_HSO4.
        u_old: Functions of the previous concentrations of V2, V3, H, HSO4.
        v: Test functions of the six unknowns.
        dt: Time step, a number or a constant.
        current: Membrane current density, signed, a number or a constant.
        velocity: The flow velocity, if any.
        regions: Whether the mesh has electrode and channel cells.
        area_ratio: Membrane area over current collector area.
        membrane_flux: Whether protons carry the membrane current.
        outlet: Whether ions leave through an outlet.
    """
    mesh = marked.mesh
    c = _constants(mesh, PARAMETERS)
    tdim = mesh.topology.dim
    dx, ds, ids = measures(marked)
    if regions:
        dxE = dx(ids(tdim, "Electrode"))
        dxAll = dxE + dx(ids(tdim, "InletChannel", "OutletChannel"))
        Q0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
        mask_E = dolfinx.fem.Function(Q0)
        mask_E.x.array[marked.entities(tdim, "Electrode")] = 1.0
        mask_F = dolfinx.fem.Function(Q0)
        mask_F.x.array[:] = 1.0 - mask_E.x.array
    else:
        dxE = dxAll = dx
        mask_E = mask_F = None
    ds_cc = ds(ids(tdim - 1, "CurrentCollector"))
    ds_mem = ds(ids(tdim - 1, "Membrane"))

    phi_s, phi_l, c_V2, c_V3, c_H, c_HSO4 = u
    c_V2_old, c_V3_old, c_H_old, c_HSO4_old = u_old
    v_s, v_l, v_c_V2, v_c_V3, v_c_H, v_c_HSO4 = v
    _F, _R, _T, _eps = c["F"], c["R"], c["T"], c["epsilon"]
    dt_c = (
        dt
        if isinstance(dt, dolfinx.fem.Constant)
        else dolfinx.fem.Constant(mesh, np.float64(dt))
    )
    _i_mem = (
        current
        if isinstance(current, dolfinx.fem.Constant)
        else dolfinx.fem.Constant(mesh, np.float64(current))
    )
    _i_cc = _i_mem * area_ratio

    # Electroneutrality
    c_SO4 = (
        -(c["z_V2"] * c_V2 + c["z_V3"] * c_V3 + c["z_H"] * c_H + c["z_HSO4"] * c_HSO4)
        / c["z_SO4"]
    )
    eps_c = dolfinx.fem.Constant(mesh, 1e-14)
    c_V2_eff = 0.5 * (c_V2 + ufl.sqrt(c_V2 * c_V2 + eps_c * eps_c))
    c_V3_eff = 0.5 * (c_V3 + ufl.sqrt(c_V3 * c_V3 + eps_c * eps_c))

    delta_phi = phi_s - phi_l - c["E_formal"]
    A = c["k0"] * ufl.exp(c["alpha_a"] * _F * delta_phi / (_R * _T))
    B = c["k0"] * ufl.exp(-c["alpha_c"] * _F * delta_phi / (_R * _T))
    if velocity is None:
        c_V2_s, c_V3_s = c_V2_eff, c_V3_eff
    else:
        eps_speed = dolfinx.fem.Constant(mesh, 1e-16)
        u_mag = ufl.sqrt(ufl.dot(velocity, velocity) + eps_speed**2)
        Re = c["rho"] * u_mag * c["df"] / c["mu"]
        k_mR = 7.0 * c["D_V2"] / c["df"] * (Re**0.4)
        k_mO = 7.0 * c["D_V3"] / c["df"] * (Re**0.4)
        den = 1.0 + A / (k_mR + eps_c) + B / (k_mO + eps_c)
        c_V3_s = (
            (1.0 + A / (k_mR + eps_c)) * c_V3_eff + (A / (k_mO + eps_c)) * c_V2_eff
        ) / den
        c_V2_s = (
            (1.0 + B / (k_mO + eps_c)) * c_V2_eff + (B / (k_mR + eps_c)) * c_V3_eff
        ) / den
    i_F = _F * (A * c_V2_s - B * c_V3_s)
    if mask_E is not None:
        i_F = mask_E * i_F
    f = c["a"] * i_F

    def effective(D):
        if mask_E is None:
            return D * _eps**1.5
        return mask_E * (D * _eps**1.5) + mask_F * D

    D_V2, D_V3, D_H, D_HSO4, D_SO4 = (
        effective(c[f"D_{ion}"]) for ion in ("V2", "V3", "H", "HSO4", "SO4")
    )
    storage = _eps if mask_E is None else mask_E * _eps + mask_F * 1.0

    kappa = (_F**2 / (_R * _T)) * (
        c["z_V2"] ** 2 * D_V2 * c_V2_eff
        + c["z_V3"] ** 2 * D_V3 * c_V3_eff
        + c["z_H"] ** 2 * D_H * c_H
        + c["z_HSO4"] ** 2 * D_HSO4 * c_HSO4
        + c["z_SO4"] ** 2 * D_SO4 * c_SO4
    )
    i_l = -kappa * ufl.grad(phi_l) - _F * (
        c["z_V2"] * D_V2 * ufl.grad(c_V2)
        + c["z_V3"] * D_V3 * ufl.grad(c_V3)
        + c["z_H"] * D_H * ufl.grad(c_H)
        + c["z_HSO4"] * D_HSO4 * ufl.grad(c_HSO4)
        + c["z_SO4"] * D_SO4 * ufl.grad(c_SO4)
    )
    sigma = (1.0 - _eps) ** 1.5 * c["sigma"]
    residual = ufl.inner(sigma * ufl.grad(phi_s), ufl.grad(v_s)) * dxE + f * v_s * dxE
    residual += ufl.inner(-i_l, ufl.grad(v_l)) * dxAll - ufl.inner(f, v_l) * dxE

    def flux(D, z, conc):
        N = -D * ufl.grad(conc) - D * z * _F * conc / (_R * _T) * ufl.grad(phi_l)
        return N if velocity is None else N + velocity * conc

    N_V2 = flux(D_V2, c["z_V2"], c_V2)
    N_V3 = flux(D_V3, c["z_V3"], c_V3)
    N_H = flux(D_H, c["z_H"], c_H)
    N_HSO4 = flux(D_HSO4, c["z_HSO4"], c_HSO4)
    residual += (
        storage * (c_V2 - c_V2_old) / dt_c * v_c_V2 * dxAll
        - ufl.inner(N_V2, ufl.grad(v_c_V2)) * dxAll
        + (1.0 / _F) * f * v_c_V2 * dxE
    )
    residual += (
        storage * (c_V3 - c_V3_old) / dt_c * v_c_V3 * dxAll
        - ufl.inner(N_V3, ufl.grad(v_c_V3)) * dxAll
        - (1.0 / _F) * f * v_c_V3 * dxE
    )
    residual += (
        storage * (c_H - c_H_old) / dt_c * v_c_H * dxAll
        - ufl.inner(N_H, ufl.grad(v_c_H)) * dxAll
    )
    residual += (
        storage * (c_HSO4 - c_HSO4_old) / dt_c * v_c_HSO4 * dxAll
        - ufl.inner(N_HSO4, ufl.grad(v_c_HSO4)) * dxAll
    )
    residual -= _i_cc * v_s * ds_cc
    residual += _i_mem * v_l * ds_mem
    if membrane_flux:
        residual += _i_mem / (c["z_H"] * _F) * v_c_H * ds_mem
    if outlet:
        n = ufl.FacetNormal(mesh)
        ds_out = ds(ids(tdim - 1, "Outlet"))
        for conc, test in (
            (c_V2, v_c_V2),
            (c_V3, v_c_V3),
            (c_H, v_c_H),
            (c_HSO4, v_c_HSO4),
        ):
            residual += ufl.dot(velocity, n) * conc * test * ds_out
    return residual


def _darcy_coefficient(mesh):
    """The Darcy coefficient, permeability over viscosity, of the 2D example."""
    c = _constants(mesh, PARAMETERS)
    return (
        (c["df"] ** 2 / (c["Kc"] * c["mu"]))
        * (c["epsilon"] ** 3)
        / ((1.0 - c["epsilon"]) ** 2)
    )


def darcy(marked: MeshData, p, q):
    """The pressure equation of the 2D example, with the inflow."""
    mesh = marked.mesh
    dx, ds, ids = measures(marked)
    u_in = dolfinx.fem.Constant(mesh, np.float64(PARAMETERS["u_in"]))
    a = ufl.inner(_darcy_coefficient(mesh) * ufl.grad(p), ufl.grad(q)) * dx
    return a - u_in * q * ds(ids(mesh.topology.dim - 1, "Inlet"))


def darcy_velocity(marked: MeshData, p):
    """The Darcy velocity of the 2D example."""
    return -_darcy_coefficient(marked.mesh) * ufl.grad(p)


DIFFUSIVITY = syu.m**2 / syu.second
CONCENTRATION = syu.mol / syu.m**3

IONS = (
    Ion("V2", 2.0, 8.10e-12 * DIFFUSIVITY, positive=True),
    Ion("V3", 3.0, 1.65e-11 * DIFFUSIVITY, positive=True),
    Ion("H", 1.0, 9.312e-9 * DIFFUSIVITY),
    Ion("HSO4", -1.0, 1.33e-9 * DIFFUSIVITY),
    Ion("SO4", -2.0, 1.065e-9 * DIFFUSIVITY),
)
"""The ions of the examples, sulfate by electroneutrality."""

RESERVOIR = {
    "V2": 300.0 * CONCENTRATION,
    "V3": 300.0 * CONCENTRATION,
    "H": 1000.0 * CONCENTRATION,
    "HSO4": 1000.0 * CONCENTRATION,
}
"""The concentrations of the reservoir of the examples."""


def electrode(domain: Domain, markers: tuple[str, ...] = ()) -> PorousElectrode:
    """The porous electrode of the examples."""
    return PorousElectrode(
        "electrode",
        domain,
        markers,
        porosity=0.94,
        conductivity=3402.1 * syu.siemens / syu.m,
        specific_area=2.3e4 / syu.m,
        fiber_diameter=0.5 * syu.micrometer,
        kozeny_carman_constant=5.55,
        pore_length=100.0 * syu.micrometer,
    )


COUPLE = RedoxCouple(
    reduced="V2",
    oxidized="V3",
    rate_constant=2.5e-7 * syu.m / syu.second,
    formal_potential=-0.36 * syu.volt,
    oxidation_on_discharge=True,
)
"""V2+ <-> V3+ + e- of the examples, oxidized on discharge."""


def half_cell(domain: Domain, porous=None, couple=None, **kwargs) -> RedoxHalfCell:
    """The negative half-cell of the examples, by default with their couple."""
    return RedoxHalfCell(
        domain,
        IONS,
        RESERVOIR,
        COUPLE if couple is None else couple,
        electrode(domain) if porous is None else porous,
        **kwargs,
    )


@pytest.fixture
def reference():
    return types.SimpleNamespace(
        PARAMETERS=PARAMETERS,
        QUADRATURE_DEGREE=QUADRATURE_DEGREE,
        W=W,
        H=H,
        interval=interval,
        rectangle=rectangle,
        box=box,
        measures=measures,
        IONS=IONS,
        RESERVOIR=RESERVOIR,
        COUPLE=COUPLE,
        electrode=electrode,
        half_cell=half_cell,
        electrochemistry=electrochemistry,
        darcy=darcy,
        darcy_velocity=darcy_velocity,
    )
