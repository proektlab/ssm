#!/usr/bin/env python

import os
import warnings
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

extra_compile_args = []
extra_link_args = []

# Only compile with OpenMP if user asks for it
USE_OPENMP = os.environ.get('USE_OPENMP', False)
if USE_OPENMP:
	print("Using OpenMP for parallel message passing.")
	extra_compile_args.append('-fopenmp')
	extra_link_args.append('-fopenmp')
	extra_link_args.append('-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/8/')
else:
    warnings.warn("Not using OpenMP for parallel message passing. "
                  "This will incur a significant performance hit. "
                  "To compile with OpenMP support, make sure you are "
                  "using the GNU gcc and g++ compilers and then run "
                  "'export USE_OPENMP=True' before installing.")

ext_modules = cythonize('**/*.pyx')
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)


setup(name='ssm',
      version='0.0.1',
      description='State space models in python',
      author='Scott Linderman',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['ssm'],
      ext_modules=ext_modules,
      include_dirs=[np.get_include(),],
      )
