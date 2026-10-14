"""Smooth joint-space trajectories (quintic polynomials).

A quintic starts and ends with zero velocity and zero acceleration,
which avoids jerky motion and is common in industrial robot planners.
"""
import numpy as np

V_MAX = 2.5   # max joint velocity [rad/s]
A_MAX = 8.0   # max joint acceleration [rad/s^2]
T_MIN = 0.3   # shortest allowed move [s]


def duration(q0, qf, v_max=V_MAX, a_max=A_MAX, t_min=T_MIN):
    """Shortest move time that respects joint velocity and acceleration limits.

    For a quintic: peak velocity = 1.875 * dq / T and
    peak acceleration = 5.774 * dq / T^2.
    """
    dq = np.abs(np.asarray(qf) - np.asarray(q0)).max()
    t_vel = 1.875 * dq / v_max
    t_acc = np.sqrt(5.774 * dq / a_max)
    return max(t_vel, t_acc, t_min)


def quintic(q0, qf, T, t):
    """Reference position, velocity and acceleration at time t in [0, T]."""
    q0, qf = np.asarray(q0, float), np.asarray(qf, float)
    s = np.clip(t / T, 0.0, 1.0)
    dq = qf - q0
    pos = q0 + dq * (10 * s**3 - 15 * s**4 + 6 * s**5)
    vel = dq * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    acc = dq * (60 * s - 180 * s**2 + 120 * s**3) / T**2
    if t >= T:
        vel, acc = np.zeros_like(dq), np.zeros_like(dq)
    return pos, vel, acc
