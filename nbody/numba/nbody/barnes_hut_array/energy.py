from __future__ import print_function

import numpy as np
from .quadTree import quadArray
import time


def compute_energy(mass, particles, energy):
    print('compute energy:')
    t_tot = time.time()

    bmin = np.min(particles[: ,:2], axis=0)
    bmax = np.max(particles[: ,:2], axis=0)
    root = quadArray(bmin, bmax, particles.shape[0])

    print('\tbuild tree:    ', end='', flush=True)
    t1 = time.time()
    root.buildTree(particles)
    t2 = time.time()
    print('{:9.4f}ms'.format(1000*(t2-t1)))

    print('\tcompute mass:  ', end='', flush=True)
    t1 = time.time()
    root.computeMassDistribution(particles, mass)
    t2 = time.time()
    print('{:9.4f}ms'.format(1000*(t2-t1)))

    print('\tcompute force: ', end='', flush=True)
    t1 = time.time()    
    for i in range(particles.shape[0]):
        acc = root.computeForce(particles[i])
        energy[i, 2:] = acc
    energy[:, :2] = particles[:, 2:]
    t2 = time.time()
    print('{:9.4f}ms'.format(1000*(t2-t1)))

    print('\ttotal:       {:11.4f}ms'.format(1000*(time.time()-t_tot)))
