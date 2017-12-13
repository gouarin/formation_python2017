import numpy as np
from pygalaxy.quadtree import Body
import pytest


@pytest.fixture(scope="module")
def pos_vel_mass():
    pos = np.array([1, 1])
    vel = np.array([2, 2])
    mass = 5.0

    yield pos, vel, mass


@pytest.fixture(scope="module")
def body(pos_vel_mass):
    pos, vel, mass = pos_vel_mass

    yield Body(pos=pos, vel=vel, mass=mass)


def test_body_init(pos_vel_mass, body):
    pos, vel, mass = pos_vel_mass
    assert np.array_equal(body.pos(), pos)
    assert np.array_equal(body.vel(), vel)
    assert np.array_equal(body.mass, mass)
