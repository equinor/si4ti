# si4ti - Python bindings

## Building

### Dependencies

-   OpenMP
-   Eigen3
-   pybind

These are all compile-time dependencies. Once the Python package was built, the
dependencies are not needed anymore.

Note: OpenMP might be needed after all.

## Testing

### Dependencies

-   NumPy
-   pytest
-   xtgeo

### Installation via pip (developer mode)

We need to specify the installation directory of the dependencies if they are
not installed in a path where CMake picks up the corresponding CMake
configuration files. On my machines I need to specify the paths fo:

-   OpenMP
-   pybind11

I need a command like this on MacOS:

```text
pip install -Ccmake.define.CMAKE_PREFIX_PATH="$(pybind11-config --cmakedir);/opt/homebrew/opt/libomp" -e . --verbose
```
