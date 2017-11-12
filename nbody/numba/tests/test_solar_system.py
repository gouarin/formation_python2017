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

class SolarSystem:
    def __init__(self, dt = nbody.physics.day_in_sec, display_step = 1):
        self.mass, self.particles = nbody.init_solar_system()
        self.time_method = nbody.RK4(dt, self.particles.shape[0], compute_energy)
        self.display_step = display_step

    def next(self):
        for i in range(self.display_step):
            self.time_method.update(self.mass, self.particles)

    def coords(self):
        return self.particles[:, :2]

sim = SolarSystem( nbody.physics.day_in_sec )

bmin = np.min(sim.coords(), axis=0)
bmax = np.max(sim.coords(), axis=0)
xmin = -1.25*np.max(np.abs([*bmin, *bmax]))

anim = Animation( sim, [xmin, -xmin, xmin, -xmin] )
anim.main_loop()
