from setuptools import setup, find_packages

setup(
    name="pyGalaxy",
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'matplotlib',
        'docopt',
        'numba',

    ],
    extras_require = {
        'opengl': ["pyopengl"]
    }
)
