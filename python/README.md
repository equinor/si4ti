# si4ti - Python bindings

## Building

### Dependencies

-   SEGYIO (TODO: Should be removed as it should not be needed, except for
    usage of xtgeo or tests (afaik))

-   xtgeo


### Installation via pip (developer mode)

We need to specify the installation directory of the dependencies if they are
not installed in a path where CMake picks up the corresponding CMake
configuration files. On my machines I need to specify the paths fo:

-   segyio
-   OpenMP
-   pybind11

I need a comman like this:

```text
pip install -Ccmake.define.CMAKE_PREFIX_PATH="$(pybind11-config --cmakedir);/opt/homebrew/opt/libomp;/Users/AEJ/projects/github/si4ti/py312-segyio-master-experimental-debug/share/segyio/cmake" -e . --verbose
```
