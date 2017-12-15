import numpy as np
from .physics import gamma_1

def init_solar_system():
    bodies = np.array([[        0, 0, 0,      0], #sun
                       [    -46e9, 0, 0, -58980], #mercury
                       [ -10748e7, 0, 0, -35260], #venus
                       [-147095e6, 0, 0, -30300], #earth
                       [ -20662e7, 0, 0, -26500], #mars
                    #    [ -74052e7, 0, 0, -13720], #jupiter
                    #    [-135255e7, 0, 0, -10180], #saturn
                    #    [ -27413e8, 0, 0,  -7110], #uranus
                    #    [-444445e7, 0, 0,  -5500], #neptune
                     ])
    mass = np.array([  1.989e30, # sun
                      .33011e24, #mercury
                      4.8675e24, #venus
                       5.972e24, #earth
                      6.4171e23, #mars
                    #  1898.19e24, #jupiter
                    #   568.34e24, #saturn
                    #   86.813e24, #uranus
                    #  102.413e24, #neptune 
                    ])

    return mass, bodies

def getOrbitalVelocity(xb, yb, mb, xs, ys):
    # Calculate distance from the planet with index idx_main
    r = [xb - xs, yb - ys]

    # distance in parsec
    dist = np.sqrt(r[0] * r[0] + r[1] * r[1])

    # Based on the distance from the sun calculate the velocity needed to maintain a circular orbit
    v = np.sqrt(gamma_1 * mb / dist)

    # Calculate a suitable vector perpendicular to r for the velocity of the tracer
    vxs = ( r[1] / dist) * v
    vys = (-r[0] / dist) * v
    return vxs, vys

def init_collisions(blackHole):
    npart = len(blackHole)
    for b in blackHole:
        npart += b['stars']

    particles = np.empty((npart, 4))
    mass = np.empty(npart)

    ind = 0
    for ib, b in enumerate(blackHole):
        particles[ind, :2] = b['coord']
        mass[ind] = b['mass']

        if (ib == 0):
            particles[ind, 2] = 0.
            particles[ind, 3] = 0.
        else:
            vx, vy = getOrbitalVelocity(blackHole[0]['coord'][0], blackHole[0]['coord'][1], blackHole[0]['mass'], b['coord'][0], b['coord'][1])
            particles[ind, 2] = b['svel']*vx 
            particles[ind, 3] = b['svel']*vy 
        ind += 1

        nstars = b['stars']
        rad = b['radstars']
        r = 0.1 + .8 * (rad * np.random.rand(nstars))
        a = 2*np.pi*np.random.rand(nstars)
        tmp_mass = 0.03 + 20*np.random.rand(nstars)
        x = b['coord'][0] + r*np.sin(a)
        y = b['coord'][1] + r*np.cos(a)

        vx, vy = getOrbitalVelocity(b['coord'][0], b['coord'][1], b['mass'], x, y)

        particles[ind:ind+nstars, 0] = x
        particles[ind:ind+nstars, 1] = y
        particles[ind:ind+nstars, 2] = 1e2*vx
        particles[ind:ind+nstars, 3] = 1e2*vy
        mass[ind:ind+nstars] = tmp_mass
        ind += nstars

    return mass, particles
