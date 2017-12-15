import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import nbody
#from nbody.naive import compute_energy
from nbody.barnes_hut_array import compute_energy

mass, particles = nbody.init_solar_system()
energy = np.zeros_like(particles)

def animate(i):
    for t in range(5):
        time_method.update(mass, particles)
    scatter.set_offsets(particles[:, :2])
    return scatter,

dt = nbody.physics.day_in_sec
time_method = nbody.RK4(dt, particles.shape[0], compute_energy)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

bmin = np.min(particles[: ,:2], axis=0)
bmax = np.max(particles[: ,:2], axis=0)
xmin = -1.25*np.max(np.abs([*bmin, *bmax]))
ax.axis([xmin, -xmin, xmin, -xmin])
scatter = plt.scatter(particles[:, 0], particles[:, 1])

anim = animation.FuncAnimation(fig, animate, blit=True)
plt.show()
