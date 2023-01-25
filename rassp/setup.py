from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


COMPILE_ARGS = ['-O3', '-std=c++17', '-march=native', '-ffast-math']


extensions = [
    Extension("rassp.binutils_fast", ["rassp/msutil/binutils_fast.pyx",],
              language='c++', 
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              include_dirs = [np.get_include()],
              extra_compile_args= COMPILE_ARGS),

    Extension("rassp.masseval",
              sources=["rassp/msutil/masseval.pyx",
                       "rassp/msutil/floatmasseval.cc",
                       "rassp/msutil/intmasseval.cc"
              ],

              depends=['rassp/msutil/intmasseval.h',
                       'rassp/msutil/floatmasseval.h',
                       'rassp/msutil/shared.h'],
              language='c++', 
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              include_dirs = [np.get_include()],
              extra_compile_args= COMPILE_ARGS),

    Extension("rassp.vertsubsetgen_fast",
              sources=["rassp/msutil/vertsubsetgen_fast.pyx"],
              depends=[], 
              language='c++', 
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              include_dirs = [np.get_include()],
              extra_compile_args= COMPILE_ARGS)
]

setup(
    name='rassp',
    version='1.0.0',
    description='RASSP: EI-MS prediction',
    author='The Jonas Lab',
    author_email='jonaslab@uchicago.edu',
    packages=['rassp', 'rassp.model', 'rassp.msutil',
              'rassp.featurize',
              'rassp.dataset'],
    install_requires=['wheel'],
    ext_modules=cythonize(extensions)
)
