import numpy as np
import pytest
from particle_model import lj_force, potential_energy, total_energy, simulate

def test_newton_third_law():
    f=lj_force(np.array([[-0.6,0.0],[0.6,0.0]]))
    assert np.allclose(f[0],-f[1])

def test_lj_potential_minimum():
    r=2**(1/6)
    assert potential_energy(np.array([[-r/2,0],[r/2,0]])) == pytest.approx(-1.0)

def test_energy_is_finite():
    r=simulate(steps=100,dt=0.001)
    assert np.isfinite(r.energy).all()

def test_verlet_has_better_energy_behavior_than_euler():
    e=simulate('euler',steps=1000,dt=0.001)
    v=simulate('verlet',steps=1000,dt=0.001)
    de=np.max(np.abs(e.energy-e.energy[0])); dv=np.max(np.abs(v.energy-v.energy[0]))
    assert dv < de

def test_shapes():
    r=simulate(steps=20)
    assert r.positions.shape==(20,2,2)
    assert r.velocities.shape==(20,2,2)

def test_invalid_method():
    with pytest.raises(ValueError): simulate('rk4')

def test_coincident_particles_rejected():
    with pytest.raises(ValueError): lj_force(np.zeros((2,2)))
