from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

from Cython.Distutils import build_ext

from Cython.Compiler.Options import get_directive_defaults, annotate
directive_defaults = get_directive_defaults()


directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

#to use the sequential version of multi-class EFC, change
#the source file to "classification_functions_seq.pyx"
ext_modules1 = [Extension(
    name="classification_functions",
    sources=["classification_functions_parallel.pyx"],
    include_dirs = [numpy.get_include()],
    define_macros=[('CYTHON_TRACE', '1')]
    )]

ext_modules2 = [Extension(
    name="dca_functions",
    sources=["dca_functions.pyx"],
    include_dirs = [numpy.get_include()],
    define_macros=[('CYTHON_TRACE', '1')]
    )]


setup(
    name='Classification functions',
    ext_modules=cythonize(ext_modules1),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)

setup(
    name='Dca functions',
    ext_modules=cythonize(ext_modules2),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)
