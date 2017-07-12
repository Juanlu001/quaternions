#!/usr/bin/env python
# -*- coding: utf8 -*-

# from distutils.core import setup
from setuptools import setup, find_packages
import glob

setup(name='quaternions',
      version='0.1',
      description='quaternions library',
      author='Matias Graña',
      author_email='matias@satellogic.com',
      packages=find_packages(),
      scripts=glob.glob('*.py'),
     )
