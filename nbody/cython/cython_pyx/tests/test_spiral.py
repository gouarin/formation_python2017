import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import nbody
#from nbody.naive import compute_energy
from nbody.barnes_hut_array import compute_energy

np.random.seed(42)

mass, particles = nbody.make_spiral_galaxy(10000)
energy = np.zeros_like(particles)

def animate(i):
    print(i)
    for t in range(10):
        time_method.update(mass, particles)
    scatter.set_offsets(particles[:, :2])
    return scatter,

dt = 1.e5
time_method = nbody.ADB6(dt, particles.shape[0], compute_energy)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_facecolor('black')
#ax.axis([-10, 10, -10, 10])

scatter = plt.scatter(particles[:, 0], particles[:, 1], c='white', s=.5)

anim = animation.FuncAnimation(fig, animate, blit=True)
plt.show()
