"""Finite element models written as summable components, on FEniCSx.

The generic framework has two layers, imported as subpackages:

- :mod:`sumfem.symbolic`, the symbolic description, without mesh data:
  domains, fields, parameters, residuals and functionals, systems of
  equations, sequences of stages, time derivatives, boundary conditions,
  transfers and dimensional analysis.
- :mod:`sumfem.discrete`, which binds them to meshes and solves them:
  discretizations, discrete systems and sequences, solvers, time stepping
  and simulations.

Battery physics and the other model families live in
:mod:`sumfem.models`, and post-processing in
:mod:`sumfem.post_processing`::

    from sumfem import discrete, symbolic

    domain = symbolic.Domain("square", "triangle")
    u = symbolic.Field("u", domain, ("Lagrange", 1))
"""

# The subpackages are imported here once they are migrated
__all__: list[str] = []
