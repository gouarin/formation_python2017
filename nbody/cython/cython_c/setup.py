from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules= [
    Extension("nbody.barnes_hut_array.quadTree", 
              ["nbody/barnes_hut_array/quadTree.pyx"]),
              ]

setup(
    name           = "nbody",
    ext_modules    = cythonize(ext_modules),
    packages       = find_packages(exclude=['tests*'])
)
