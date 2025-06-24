# si4ti changelog

## 1.1.0 - 2025-06-26

- Added Python bindings for the impedance calculation. Check the README for
  more information.
- Bumped minimum required CMake version to 3.15.
- Bumped minimum required C++ version to C++14.
- Introduced new CMAKE build flags to allow for more fine-grained builds:
  - `BUILD_TESTING`: Build of tests.
  - `BUILD_IMPEDANCE`: Build impedance command line tool.
  - `BUILD_TIMESHIFT`: Build timeshift command line tool.
- Improved and fixed the regression tests.
- Migrated from Travis CI to GitHub Actions. As part of this, tests were
  migrated to Red Hat Enterprise Linux (RHEL) 8 because RHEL 7 reached its end
  of life.

## 1.0.0 - 2021-06-02

Initial release
