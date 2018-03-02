#!/usr/bin/env python

from setuptools import setup, Extension

setup(name='simpli',
      description='simpli',
      long_description='simpli',
      author='me',
      packages=['simpli', 'simpli.timeshift'],
      ext_modules=[Extension('simpli.timeshift.bspline',
                   sources=['simpli/timeshift/bspline.cpp'],
                   libraries=['sml'],
      )],
      platforms='any',
      install_requires=['numpy', 'scipy'],
      setup_requires=['setuptools >=28', 'setuptools_scm', 'pytest-runner'],
      tests_require=['pytest', 'hypothesis'],
)
