from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="pyGalaxy",
    #ext_modules=cythonize('pygalaxy/quadtree.pyx'),  # accepts a glob pattern
    packages=find_packages(exclude=['tests*']),
)
