# nbody

In this example, we study the nbody problem using a naive implementation and the Barnes-Hut algorithm.

The Barnes-Hut algorithm uses a tree data structure to store the bodies and perform the computation of the acceleration in nlog(n). Unfortunately, to use Pythran and Numba on this problem, it is not suitable to have this type of data structure. So, we write the tree using an array where the first components are the bodies and the folowing entries are the cells which are represented by 4 integers (ie the quad tree).

For more information about the Barnes-Hut algorithm

https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation?oldid=469278664

There are two examples in the tests directory of each version:

- solar system
- two galaxies with 3000 bodies

To try one of the versions, go for example in python directory. Then, install the nbody package using the command line

```
python setup.py install [--user]
```

Then, go to the tests directory and execute the command line

```
python test_solar_system.py 
```