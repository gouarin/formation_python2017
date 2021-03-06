from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name = "kernels",
    ext_modules = cythonize([Extension("*", ["*.pyx"])]),
)