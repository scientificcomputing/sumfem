# sumfem

Finite element models written as summable components, with
[DOLFINx](https://github.com/FEniCS/dolfinx/). It was developed for redox flow
batteries, whose models are in `sumfem.models`.

## Installation

This software package requires an installation of FEniCS on your system.
It needs the main branch of DOLFINx (0.12.0.dev0 or newer), for its fixes of
mixed-dimensional and manifold assembly. The nightly Docker image and Spack's
`main` versions provide it. The Ubuntu, Debian and conda packages carry
releases, so they work once DOLFINx 0.12 is released.
FEniCS can be installed in many ways, including:

### Ubuntu/Debian packages

```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx
```

and then call

```bash
python3 -m pip install -e .
```

from the root of this repo to install the package.

### Docker

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm ghcr.io/fenics/dolfinx/dolfinx:nightly
```

and then call

```bash
python3 -m pip install -e .
```

from the root of this repo to install the package.

### Conda

```bash
conda create -n fenicsx_env
conda activate fenicsx_env
conda install -c conda-forge fenics-dolfinx matplotlib
```

and then call

```bash
python3 -m pip install -e .
```

from the root of this repo to install the package.

### Spack

```bash
spack env create dolfinx_env
spack env activate dolfinx_env
spack add py-fenics-dolfinx@main+petsc py-matplotlib py-pip
spack concretize
spack install -j 4
python3 -m pip install -e .
```

## Funding

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101137725.

See [NMBU: BatCat](https://www.nmbu.no/en/research/projects/batcat) for more information regarding the project.
