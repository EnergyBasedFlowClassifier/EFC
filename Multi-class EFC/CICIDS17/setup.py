from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
import numpy
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

setup(
    name='Classification functions app',
    ext_modules=cythonize("classification_functions_parallel.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)

setup(
    name='Dca functions app',
    ext_modules=cythonize("dca_functions.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)
