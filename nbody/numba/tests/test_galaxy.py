import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

import nbody
#from nbody.naive import compute_energy
from nbody.barnes_hut_array import compute_energy

sys.path.append('opengl')
sys.path.append('..')
from animation import Animation

np.random.seed(42)

class Galaxy:
    def __init__(self, blackHole, dt = 1.):
        self.mass, self.particles = nbody.init_collisions(blackHole)
        self.time_method = nbody.ADB6(dt, self.particles.shape[0], compute_energy)
        self.it = 0

    def next(self):
        self.time_method.update(self.mass, self.particles)

    def coords(self):
        return self.particles[:, :2]


blackHole = [
            {'coord': [0, 0], 'mass': 1000000, 'svel': 1, 'stars': 2000, 'radstars': 3},
            {'coord': [3, 3], 'mass': 1000000, 'svel': 0.9, 'stars': 1000, 'radstars': 1}
            ]

sim = Galaxy(blackHole)
anim = Animation( sim, axis=[-10, 10, -10, 10] )
anim.main_loop()

"""
def animate(i):
    print(i)
    for t in range(5):
        time_method.update(mass, particles)
    scatter.set_offsets(particles[:, :2])
    return scatter,

dt = 1.
time_method = nbody.ADB6(dt, particles.shape[0], compute_energy)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_facecolor('black')
ax.axis([-10, 10, -10, 10])

scatter = plt.scatter(particles[:, 0], particles[:, 1], c='white', s=.5)

anim = animation.FuncAnimation(fig, animate, blit=True)
plt.show()
"""
