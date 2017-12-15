from ..forces import force
import numpy as np

def compute_energy(mass, particles, energy):
    energy[:] = 0.
    N = energy.shape[0]
    for i in range(N):
        for j in range(N):
            if i != j:
                F = force(particles[i, :2], particles[j,:2], mass[j])
                energy[i, 2:] += F
    energy[:, :2] = particles[:, 2:]