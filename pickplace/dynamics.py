"""Rigid-body dynamics of the 2-link planar arm.

Equation of motion:  M(q) q'' + C(q, q') q' + B q' = tau
The arm is horizontal, so gravity does not act on the joints.
Links are modelled as uniform rods.
"""
import numpy as np

from .kinematics import L1, L2

M1, M2 = 1.0, 0.6          # link masses [kg]
LC1, LC2 = L1 / 2, L2 / 2  # centre-of-mass distances [m]
I1 = M1 * L1**2 / 12       # link inertias about their centre [kg m^2]
I2 = M2 * L2**2 / 12
FRICTION = np.array([0.05, 0.03])  # viscous joint friction [N m s/rad]
TAU_MAX = np.array([10.0, 4.0])    # motor torque limits [N m]

_A = I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2)
_B = M2 * L1 * LC2
_D = I2 + M2 * LC2**2


def mass_matrix(q):
    c2 = np.cos(q[1])
    return np.array([
        [_A + 2 * _B * c2, _D + _B * c2],
        [_D + _B * c2, _D],
    ])


def coriolis(q, qd):
    """Coriolis and centrifugal torques C(q, q') q'."""
    s2 = np.sin(q[1])
    return np.array([
        -_B * s2 * (2 * qd[0] * qd[1] + qd[1] ** 2),
        _B * s2 * qd[0] ** 2,
    ])


def inverse_dynamics(q, qd, qdd):
    """Torque needed to produce acceleration qdd (used as feedforward)."""
    return mass_matrix(q) @ qdd + coriolis(q, qd) + FRICTION * qd


class Arm:
    """Simulated arm state, integrated with semi-implicit Euler."""

    def __init__(self, q0):
        self.q = np.array(q0, float)
        self.qd = np.zeros(2)

    def step(self, tau, dt):
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        rhs = tau - coriolis(self.q, self.qd) - FRICTION * self.qd
        qdd = np.linalg.solve(mass_matrix(self.q), rhs)
        self.qd = self.qd + qdd * dt
        self.q = self.q + self.qd * dt
        return tau
