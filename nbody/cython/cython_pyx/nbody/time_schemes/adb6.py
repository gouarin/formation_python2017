import numpy as np
from .rk4 import RK4

class ADB6:
    def __init__(self, dt, nbodies, method):
        self.dt = dt
        self.method = method
        self.c = [4277.0 / 1440.0,
                 -7923.0 / 1440.0,
                  9982.0 / 1440.0,
                 -7298.0 / 1440.0,
                  2877.0 / 1440.0,
                  -475.0 / 1440.0]
        self.f = np.zeros((6, nbodies, 4)) 

    def init(self, mass, particles):
        nbodies = mass.nbodies
        rk4 = RK4(self.dt, nbodies, self.method)

        for i in range(5):
            rk4.update(self, mass, particles)
            self.f[i, :] = rk4.k1 
        
        self.method(mass, particles, self.f[5])

    def update(self, mass, particles):
        particles[:, :] += self.dt * (self.c[0] * self.f[5] +
                                      self.c[1] * self.f[4] +
                                      self.c[2] * self.f[3] +
                                      self.c[3] * self.f[2] +
                                      self.c[4] * self.f[1] +
                                      self.c[5] * self.f[0])
        self.f = np.roll(self.f, -1, axis=0)
        self.method(mass, particles, self.f[5])
