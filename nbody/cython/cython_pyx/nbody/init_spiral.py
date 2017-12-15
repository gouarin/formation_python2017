#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a spiral galaxy. 

TODO:
Add central mass, 
Add mass array to output
"""

from math import *
import random
import numpy as np

# Constants
grav = 6.67408e-11  # m3 kg-1 s-2
msun = 1.98855e30  # kg
mearth = 5.9722e24  # kg
au = 1.496e11 # m

def kepler (e, oldl):
    """
    Parameters
    ----------
    oldl
    e
    """
    #  ------------------------------------------------------------------------------
    
    twopi = 2.*np.pi
    piby2 = .5*np.pi
    
    #   Reduce mean anomaly to lie in the range 0 < l < pi
    l = oldl%twopi
    mask = oldl<0
    l[mask] += twopi
    
    sign = np.ones_like(l)
    mask = l>np.pi
    l[mask] = twopi - l[mask]   
    sign[mask] = -1.     
    
    ome = 1.0 - e
    
    if e <.55:
        u1 = np.empty_like(l)

        #   Regions A,B or C in Nijenhuis
        #   -----------------------------
        
        #   Rough starting value for eccentric anomaly
        mask = l < ome
        u1[mask] = ome

        mask = np.logical_not(mask)
        mask1 = np.logical_and(mask, l>np.pi-1.+e)
        u1[mask1] = (l[mask1] + e*np.pi)/(1. + e)
        mask1 = np.logical_and(mask, l<=np.pi-1.+e)
        u1[mask1] = l[mask1] + e
        
        #   Improved value using Halley's method
        x = np.copy(u1)
        flag = u1 > piby2
        x[flag] = np.pi - u1[flag]

        x2 = x*x
        sn = x*(1. + x2*(-.16605 + x2*.00761) )
        dsn = 1.0 + x2*(-.49815 + x2*.03805)
        dsn[flag] = -dsn[flag]
        f2 = e*sn
        f0 = u1 - f2 - l
        f1 = 1.0 - e*dsn
        u2 = u1 - f0/(f1 - .5*f0*f2/f1)
    #else:
    # mask = l >= .45
    # else:
         
    #      #   Region D in Nijenhuis
    #      #   ---------------------
         
    #      #   Rough starting value for eccentric anomaly
    #      z1 = 4.0     *e + .5
    #      p = ome / z1
    #      q = .5 * l / z1
    #      p2 = p*p
    #      z2 = exp( log( sqrt( p2*p + q*q ) + q )/1.5 )
    #      u1 = 2.0     *q / ( z2 + p + p2/z2 )
         
    #      #   Improved value using Newton's method
    #      z2 = u1*u1
    #      z3 = z2*z2
    #      u2 = u1 - .075*u1*z3 / (ome + z1*z2 + .375*z3)
    #      u2 = l + e*u2*( 3.0      - 4.0     *u2*u2 )
    
    #   Accurate value using 3rd-order version of Newton's method
    #   N.B. Keep cos(u2) rather than sqrt( 1-sin^2(u2) ) to maintain accuracy
    
    #   First get accurate values for u2 - sin(u2) and 1 - cos(u2)
    bigg = u2 > piby2
    z3 = np.copy(u2)
    z3[bigg] = np.pi - u2[bigg]
    
    x[:] = z3
    big = z3 > (.5*piby2)
    x[big] = piby2 - z3[big]
    
    x2 = x*x
    ss = 1.0     
    cc = 1.0     
    
    ss = x*x2/6.*(1. - x2/20.*(1. - x2/42.*(1. - x2/72.*(1. - x2/110.*(1. - x2/156.*(1. - x2/210.*(1. - x2/272.)))))))
    cc =   x2/2.*(1. - x2/12.*(1. - x2/30.*(1. - x2/56.*(1. - x2/ 90.*(1. - x2/132.*(1. - x2/182.*(1. - x2/240.*(1. - x2/306.))))))))
    
    z1 = np.copy(ss)
    z2 = np.copy(cc)

    z1[big] = cc[big] + z3[big] - 1.     
    z2[big] = ss[big] + z3[big] + 1. - piby2

    z1[bigg] = 2.*u2[bigg] + z1[bigg] - pi
    z2[bigg] = 2. - z2[bigg]
    
    f0 = l - u2*ome - e*z1
    f1 = ome + e*z2
    f2 = .5*e*(u2-z1)
    f3 = e/6.*(1. - z2)
    z1 = f0/f1
    z2 = f0/(f2*z1+f1)
    
    return sign*(u2 + f0/((f3*z1+f2)*z2+f1) )

def element2cartesian(gm, a, e, i, p, n, l):
    """
    @brief Calculates Cartesian coordinates and velocities given Keplerian orbital
    elements (for elliptical, parabolic or hyperbolic orbits).
    \n\n
    Based on a routine from Levison and Duncan's SWIFT integrator.
    
    Parameters
    ----------
    
    gm: float
        gm = grav const * (central + secondary mass)
    a: float
        semimajor axis (m)
    e: float
        e = eccentricity (has to be 0 <= e < 1)
    p: float
        p = longitude of perihelion (radians)
    l: float
        l = mean anomaly  (radians)
    i: float
        inclination (radians)
    n: float
        yet another angle, I don't remember which one (radians)
    
    Output
    ------
    x: float
        Cartesian positions  ( units the same as a )
    y: float
        Cartesian positions  ( units the same as a )
    z: float
        Cartesian positions  ( units the same as a )
    u: float
        Cartesian velocities ( units the same as sqrt(gm/a) )
    v: float
        Cartesian velocities ( units the same as sqrt(gm/a) )
    w: float
        Cartesian velocities ( units the same as sqrt(gm/a) )
    
    # Input/Output
    real(double_precision), intent(inout) :: i
    real(double_precision), intent(inout) :: n
    
    """

    #------------------------------------------------------------------------------
    
    # Change from longitude of perihelion to argument of perihelion
    g = p - n
    
    # Rotation factors
    si = np.sin(i)
    sg = np.sin(g)
    sn = np.sin(n)
    ci = np.cos(i)
    cg = np.cos(g)
    cn = np.cos(n)
    
    z1 = cg * cn
    z2 = cg * sn
    z3 = sg * cn
    z4 = sg * sn
    d11 =  z1 - z4*ci
    d12 =  z2 + z3*ci
    d13 = sg * si
    d21 = -z3 - z2*ci
    d22 = -z4 + z1*ci
    d23 = cg * si
    
    # Semi-major axis
    q = a * (1.0 - e)
    
    # Ellipse
    if (e < 1.0):
        romes = sqrt(1.0 - e*e)
        temp = kepler(e, l)
        se = np.sin(temp)
        ce = np.cos(temp)
        
        z1 = a * (ce - e)
        z2 = a * romes * se
        temp = np.sqrt(gm/a) / (1.0 - e*ce)
        z3 = -se * temp
        z4 = romes * ce * temp
   
    pos = np.asarray([d11 * z1  +  d21 * z2,
                      d12 * z1  +  d22 * z2,
                      d13 * z1  +  d23 * z2             
    ])

    vel = np.asarray([d11 * z3  +  d21 * z4,
                      d12 * z3  +  d22 * z4,
                      d13 * z3  +  d23 * z4
    ])
    
    return pos, vel


def make_spiral_galaxy(nb_particles=10, central_mass=msun, shift=(0.,0.,0.)):
    """
    Make a fake galaxy. Return the positions and velocities
    
    
    Parameter
    --------
    nb_particles: int
        total number of particles
    central_mass: float
        central mass in kg (by default, sun mass)
    shift: tuple(float, float, float)
        shift to apply to the galaxies positions, in metres
    
    gm # gm = grav const * (central + secondary mass)
    a  # semimajor axis
    e  # e = eccentricity
    p  # p = longitude of perihelion ### 
    l  # l = mean anomaly 
    
    Return
    ------
    (mass, pos, vel)
    mass is a 1D (n) array, pos and vel are (3, n) arrays
    """
    
    m_min = 1*mearth  # kg
    m_max = 20*mearth # kg
    a_min = au        # m
    a_max = 10 * au  # m
    e_val = 0.1 # 0<e<1 ; always the same value
    i_val = 0. # radians ; always the same value
    #~ l_values = [0, 0.25*np.pi, -0.5*np.pi, np.pi] # list of possible mean anomaly (perihelion or aphelion to ensure the spiral visual right from the start instead of waiting a long time
        
    mass = np.random.uniform(m_min, m_max, nb_particles)
    a = np.random.uniform(a_min, a_max, nb_particles) # Semimajor axis
    p  = a * (2*np.pi/a_max)

    e  = e_val # 0<e<1
    i  = i_val  # radians
    
    gm = grav * (central_mass+mass)
    n  = np.random.uniform(0, 2*np.pi, nb_particles)  # radians
    l  = np.random.uniform(0, 2*pi, nb_particles)  # radians
    #~ l  = np.random.choice(l_values, nb_particles)  # radians

    pos, vel = element2cartesian(gm, a, e, i, p, n, l)
    
    # We insert the central object in the arrays
    # Central object is assumed to have no velocity at all. 
    mass = np.insert(mass, 0, central_mass)
    pos = np.insert(pos, 0, [0.,0.,0.], axis=1)
    vel = np.insert(vel, 0, [0.,0.,0.], axis=1)
    
    # Shift position if needed
    if (shift != (0.,0.,0.)):
        pos += shift
    

    return mass, np.concatenate((pos[:2], vel[:2])).T

def test_plot():
    """
    display the positions of the particles. 
    """
    (mass, x, v) = make_spiral_galaxy(10000)
    
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plot1 = fig.add_subplot(1, 1, 1)
    plot1.plot(x[0],x[1], '+')
    
    plt.show()

