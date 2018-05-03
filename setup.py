#!/usr/bin/env python

from setuptools import setup, Extension

setup(name='simpli',
      description='simpli',
      long_description='simpli',
      author='me',
      packages=['simpli', 'simpli.timeshift'],
      ext_modules=[Extension('simpli.timeshift.ts',
                   sources=['simpli/timeshift/ts.cpp'],
                   define_macros=[('EIGEN_FFTW_DEFAULT', '1')],
                   extra_compile_args=['-isystem/usr/include/eigen3',
                                       '-std=c++11',
                                      ],
                   libraries=['fftw3f'],
      )],
      platforms='any',
      install_requires=['numpy', 'scipy'],
      setup_requires=['setuptools >=28', 'setuptools_scm', 'pytest-runner'],
      tests_require=['pytest', 'hypothesis'],
      entry_points = {
          'console_scripts' : ['ts=simpli.timeshift.main:main'],
      },
)
