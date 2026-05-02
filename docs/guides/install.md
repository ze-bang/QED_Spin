# Installation

This page covers the supported install paths. Pick the one that matches your
intended workflow:

| Workflow                                        | Recommended path                |
|-------------------------------------------------|---------------------------------|
| C++ developer, wants to hack on the source      | [From source (CMake)](#from-source-cmake) |
| C++ user, wants to consume the library          | [Install + `find_package(ED)`](#consume-via-find_packageed-config) |
| Python user, wants the `qed` package     | [Install the wheel (pip)](#install-the-python-wheel) |
| Casual user, wants a sandboxed environment      | [Docker / dev container](#docker-and-dev-containers) (planned) |

(from-source-cmake)=
## From source (CMake)

Prerequisites (Ubuntu 22.04 names; the package names on other distros are
analogous):

```bash
sudo apt-get install -y \
    build-essential cmake ninja-build pkg-config \
    libopenblas-dev liblapacke-dev \
    libhdf5-dev libhdf5-cpp-103 \
    libeigen3-dev libarpack2-dev
```

Optional:

- **MPI** (`libopenmpi-dev openmpi-bin`) — enables ScaLAPACK distributed
  diagonalization and MPI-aware workflows.
- **CUDA Toolkit ≥ 12.x** — enables the GPU solvers (`ed_solvers_gpu`).

Then:

```bash
git clone https://github.com/.../exact_diagonalization_clean.git
cd exact_diagonalization_clean/exact_diagonalization_cpp
cmake --preset ci-linux            # or pick another preset (see CMakePresets.json)
cmake --build --preset ci-linux -- -j$(nproc)
ctest --preset ci-linux
```

(consume-via-find_packageed-config)=
## Consume via `find_package(ED CONFIG)`

After building, install to a prefix of your choice:

```bash
cmake --install build --prefix ~/.local
```

In a downstream project:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(ED CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE ED::ed_solvers_cpu)
```

The exported targets are:

- `ED::ed_io`
- `ED::ed_core`
- `ED::ed_dssf`
- `ED::ed_solvers_cpu`
- `ED::ed_solvers_gpu` (only if the install was built `WITH_CUDA=ON`)

(install-the-python-wheel)=
## Install the Python wheel

The `qed` Python package is built via `scikit-build-core`, which
drives the same CMake build under the hood:

```bash
python -m pip install "scikit-build-core>=0.10" "pybind11>=2.13"
python -m pip install .
```

For a **lean** wheel (no CUDA, no MPI, OpenBLAS only — useful for CI and
laptops):

```bash
CMAKE_ARGS="-DBUILD_ED_TESTS=OFF -DWITH_CUDA=OFF -DWITH_MPI=OFF -DBLAS_PROFILE=OPENBLAS" \
  python -m pip install .
```

To verify the install:

```python
import qed
print(qed.__version__)
op = qed.Operator(num_sites=2, spin_length=0.5)
print(op.dimension)   # -> 4
```

(docker-and-dev-containers)=
## Docker and dev containers

A reproducible `Dockerfile.dev` plus a Nix flake are tracked under audit
items P2.12 / P2.7. They will land alongside the next minor release.
