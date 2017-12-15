from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules= [
    Extension("nbody.barnes_hut_array.quadTree", 
              ["nbody/barnes_hut_array/quadTree.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'],
              ),
    Extension("nbody.barnes_hut_array.energy", 
              ["nbody/barnes_hut_array/energy.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'],
              )
    ]

setup(
    name           = "nbody",
    ext_modules    = cythonize(ext_modules),
    packages       = find_packages(exclude=['tests*'])
)
