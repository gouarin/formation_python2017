from six import print_

import numpy as np
from .quadTree import quadArray
import time

from . import numba_functions
import numba

@numba.njit(parallel=True)
def compute_force( nbodies, child, center_of_mass, mass, cell_radius, particles, energy):
    for i in numba.prange(particles.shape[0]):
        acc = numba_functions.computeForce( nbodies, child, center_of_mass, mass, cell_radius, particles[i] )
        energy[i, 2] = acc[0]
        energy[i, 3] = acc[1]

def compute_energy(mass, particles, energy):
    print('compute energy:')
    t_tot = time.time()

    bmin = np.min(particles[: ,:2], axis=0)
    bmax = np.max(particles[: ,:2], axis=0)
    root = quadArray(bmin, bmax, particles.shape[0])

    print_('\tbuild tree:    ', end='', flush=True)
    t1 = time.time()
    root.buildTree(particles)
    t2 = time.time()
    print_('{:9.4f}ms'.format(1000*(t2-t1)))

    print_('\tcompute mass:  ', end='', flush=True)
    t1 = time.time()
    root.computeMassDistribution(particles, mass)
    t2 = time.time()
    print_('{:9.4f}ms'.format(1000*(t2-t1)))

    print_('\tcompute force: ', end='', flush=True)
    t1 = time.time()    
    compute_force( root.nbodies, root.child, root.center_of_mass, root.mass, root.cell_radius, particles, energy )
    energy[:, :2] = particles[:, 2:]
    t2 = time.time()
    print_('{:9.4f}ms'.format(1000*(t2-t1)))

    print_('\ttotal:       {:11.4f}ms'.format(1000*(time.time()-t_tot)))
