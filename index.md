# sumfem

sumfem is a library of reusable, summable variational models for
[FEniCSx](https://fenicsproject.org/).

A model is written once as a *component* that returns its contribution to a
weak form. Components are described symbolically, without mesh data,
summed with `+` into the equations of a problem, and extended by
inheritance. The description is then bound to meshes, compiled once, and
solved with DOLFINx and PETSc, in serial or in parallel.

The [tutorials](docs/tutorials/01_first_model.py) introduce the concepts one at a time.

```{tableofcontents}
```
