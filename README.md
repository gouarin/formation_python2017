# pyGalaxy

In this project, we study the nbody problem and the Barnes-Hut algorithm.

The Barnes-Hut algorithm uses a tree data structure to store the bodies and perform the computation of the acceleration in nlog(n). Unfortunately, in order to use Numba on this problem, it is not suitable to have this type of data structure. So, we write the tree using an array where the first components are the bodies and the folowing entries are the cells which are represented by 4 integers (ie the quad tree).

For more information about the Barnes-Hut algorithm

https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation?oldid=469278664

## Install
To install this package, we strongly encourage to use a [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or a [`conda env`](https://conda.io/docs/user-guide/tasks/manage-environments.html). Go to the root of this project (that means the folder that contains the `setup.py` file) and then run:

`pip install .`

**With this standard installation you cannot run the examples with the OpenGL rendering engine. To do that you need to install the `opengl` variant**:

`pip install .[opengl]`

## Examples

There are two examples in the `examples` directory of each version:

- solar system
- two galaxies with 3000 bodies

To try the examples just run the examples doing: 

`python examples/galaxy.py`

You can print an help test doing:

`python examples/galaxy.py -h`. 

If you compiled the `opengl` version, you can specify the renderer with

`python examples/galaxy.py -R opengl`

# Contributors
Check the [CONTRIBUTORS.md](CONTRIBUTORS.md) file.
