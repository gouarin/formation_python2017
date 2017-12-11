from pygalaxy.quadtree import Body
import numpy as np


def test_body_init():
    pos = np.array([1, 1])
    vel = np.array([2, 2])
    mass = 5.0

    body = Body(pos=pos, vel=vel, mass=mass)

    assert np.array_equal(body.pos, pos)
    assert np.array_equal(body.vel, vel)
    assert np.array_equal(body.mass, mass)
