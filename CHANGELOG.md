# si4ti changelog

## Unreleased

- Drop support for Python 3.10 (end of life).

## 1.2.0 - 2026-08-11

- Add support for Python 3.14.
  - Python 3.14t (free-threaded Python) is not supported.
  - Python 3.15 is not yet supported.
- Support for macOS on Intel-based machines is dropped. Pre-built wheels
  now require macOS 15.0 or newer (arm64 only).

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
