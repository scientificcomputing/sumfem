# Installation

sumfem needs the main branch of [DOLFINx](https://github.com/FEniCS/dolfinx/)
(0.12.0.dev0 or newer), for its fixes of mixed-dimensional and manifold
assembly. The nightly Docker image and Spack's `main` versions provide it.
The Ubuntu, Debian and conda packages carry releases, so they work once
DOLFINx 0.12 is released.

## Docker

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm ghcr.io/fenics/dolfinx/dolfinx:nightly
```

## Spack

```bash
spack env create dolfinx_env
spack env activate dolfinx_env
spack add py-fenics-dolfinx@main+petsc py-matplotlib py-pip
spack concretize
spack install -j 4
```

## Installing sumfem

From the root of the repository:

```bash
python3 -m pip install -e .
```

The tutorials and demos need the `examples` extra, and building this
documentation the `docs` extra:

```bash
python3 -m pip install -e ".[docs]"
jupyter-book build .
```
