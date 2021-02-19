from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Classification functions seq app',
    ext_modules=cythonize("classification_functions.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)

setup(
    name='Dca functions app',
    ext_modules=cythonize("dca_functions.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)
