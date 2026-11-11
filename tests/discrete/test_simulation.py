"""Output and archived metadata of simulations."""

import json
import pathlib

from mpi4py import MPI

import numpy as np
import pytest
import sympy.physics.units as syu

from sumfem.discrete import DomainMaps, discretize
from sumfem.solvers import BDF, Output, Simulation
from sumfem.symbolic import (
    Clock,
    DirichletBC,
    Domain,
    Field,
    Parameter,
    Residual,
    UnitError,
)


def shared_directory(tmp_path: pathlib.Path) -> pathlib.Path:
    """One directory for all processes, which write output collectively."""
    return pathlib.Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))


def test_output_and_metadata(toy, square, tmp_path):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1), unit=syu.K)
    mean = Field("mean", domain, "Real")
    target = Parameter("target", domain, default=0.25, unit=syu.K, doc="Mean of c")

    class MeanConstraint(Residual):
        """Fix the mean of c with a Lagrange multiplier."""

        def residual(self):
            dx = domain.dx()
            return mean * c.v * dx + (c - target) * mean.v * dx

    model = (
        toy.Accumulation("dc", c)
        + toy.Diffusion("d", c)
        + toy.Source("s", c, f=1.0)
        + MeanConstraint("m")
    )
    directory = shared_directory(tmp_path) / "run"
    disc = discretize([], DomainMaps({"cell": square()}))
    clock = Clock(domain, dt=0.5)
    output = Output(directory)
    Simulation(disc, model, clock=clock, scheme=BDF(1), output=output).run(num_steps=2)
    MPI.COMM_WORLD.barrier()

    assert (directory / "c.bp").is_dir()
    scalars = np.loadtxt(directory / "real_fields.csv", delimiter=",", skiprows=1)
    # The initial values and both steps
    np.testing.assert_allclose(scalars[:, 0], [0.0, 0.5, 1.0])
    assert scalars[-1, 1] == float(disc.fields["mean"].x.array[0])

    metadata = json.loads((directory / "metadata.json").read_text())
    assert {"dolfinx", "ufl", "sumfem", "python"} <= set(metadata["versions"])
    assert metadata["parameters"]["target"] == {
        "value": 0.25,
        "unit": "kelvin",
        "doc": "Mean of c",
    }
    assert metadata["parameters"]["d.k"]["value"] == 1.0
    assert metadata["model"]["unknowns"] == ["c", "mean"]
    assert metadata["model"]["summands"] == ["dc", "d", "s", "m"]
    assert metadata["time"] == {
        "t0": 0.0,
        "dt": 0.5,
        "scheme": {"type": "BDF", "order": 1},
        "time_scale": 0.5,
    }
    assert metadata["meshes"]["cell"]["cell_type"] == "triangle"


def test_output_writes_only_the_chosen_fields(toy, square, sides, tmp_path):
    domain = Domain("cell", "triangle")
    a = Field("a", domain, ("Lagrange", 1))
    b = Field("b", domain, ("Lagrange", 1))
    model = (
        toy.Diffusion("da", a)
        + toy.Source("sa", a, f=1.0)
        + toy.Diffusion("db", b)
        + toy.Source("sb", b, f=a)
    )
    directory = shared_directory(tmp_path) / "chosen"
    disc = discretize([], DomainMaps({"cell": square()}))
    bcs = [DirichletBC(a, sides, 0.0), DirichletBC(b, sides, 0.0)]
    output = Output(directory, fields=("b",))
    clock = Clock(domain, dt=1.0)
    Simulation(disc, model, bcs, clock=clock, output=output).run(num_steps=1)
    MPI.COMM_WORLD.barrier()
    assert (directory / "b.bp").is_dir()
    assert not (directory / "a.bp").exists()


def test_simulation_checks_units_on_request(toy, square, sides):
    domain = Domain("cell", "triangle")
    c = Field("c", domain, ("Lagrange", 1))
    # A dimensionless source against a Laplacian, per square metre
    model = toy.Diffusion("d", c) + toy.Source("s", c, f=1.0)
    bcs = [DirichletBC(c, sides, 0.0)]
    clock = Clock(domain, dt=1.0)
    Simulation(discretize([], DomainMaps({"cell": square()})), model, bcs, clock=clock)
    with pytest.raises(UnitError, match="equation for 'c'"):
        Simulation(
            discretize([], DomainMaps({"cell": square()})),
            model,
            bcs,
            clock=clock,
            check_units=True,
        )
