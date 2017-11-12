import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

import nbody
#from nbody.naive import compute_energy
from nbody.barnes_hut_array import compute_energy

sys.path.append('opengl')
#sys.path.append('matplotlib')
from animation import Animation

np.random.seed(42)

class Galaxy:
    def __init__(self, blackHole, dt = 1., display_step = 1):
        self.mass, self.particles = nbody.init_collisions(blackHole)
        self.time_method = nbody.ADB6(dt, self.particles.shape[0], compute_energy)
        self.display_step = display_step
        self.it = 0

    def next(self):
        for i in range(self.display_step):
            self.time_method.update(self.mass, self.particles)
            self.it += 1

    def coords(self):
        return self.particles[:, :2]


blackHole = [
            {'coord': [0, 0], 'mass': 1000000, 'svel': 1, 'stars': 2000, 'radstars': 3},
            {'coord': [3, 3], 'mass': 1000000, 'svel': 0.9, 'stars': 1000, 'radstars': 1}
            ]

sim = Galaxy(blackHole)
anim = Animation( sim, axis=[-10, 10, -10, 10] )
anim.main_loop()

