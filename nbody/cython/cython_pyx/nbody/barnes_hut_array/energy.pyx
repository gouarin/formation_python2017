# cython: profile=True
import cython

import numpy as np
cimport numpy as cnp
from .quadTree import quadArray

@cython.boundscheck(False)
def compute_energy(double[:] mass, double[:,:] particles, double[:,:] energy):
    cdef:
        int i, nparticules
        double acc[2]

    bmin = np.min(particles[: ,:2], axis=0)
    bmax = np.max(particles[: ,:2], axis=0)
    energy_view = energy
    root = quadArray(bmin, bmax, particles.shape[0])
    root.buildTree(particles)
    root.computeMassDistribution(particles, mass)
    nparticles = particles.shape[0]

    for i in range(nparticles):
        acc = root.computeForce(particles[i])
        energy_view[i, 2] = acc[0]
        energy_view[i, 3] = acc[1]
    energy[:, :2] = particles[:, 2:]