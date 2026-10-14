"""Kinematics of a 2-link planar (SCARA-style) robot arm.

The arm moves in the horizontal x-y plane. Joint 1 sits at the origin,
joint 2 (the elbow) at the end of link 1. Angles are in radians.
"""
import numpy as np

L1 = 0.30  # link 1 length [m]
L2 = 0.25  # link 2 length [m]


def forward(q, l1=L1, l2=L2):
    """Forward kinematics: joint angles -> (elbow position, end-effector position)."""
    q1, q2 = q
    elbow = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    ee = elbow + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    return elbow, ee


def jacobian(q, l1=L1, l2=L2):
    """2x2 Jacobian mapping joint velocities to end-effector velocity."""
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([
        [-l1 * s1 - l2 * s12, -l2 * s12],
        [l1 * c1 + l2 * c12, l2 * c12],
    ])


def reachable(x, y, l1=L1, l2=L2, margin=0.01):
    r = np.hypot(x, y)
    return abs(l1 - l2) + margin <= r <= l1 + l2 - margin


def inverse(x, y, q_current=None, l1=L1, l2=L2):
    """Analytic inverse kinematics (law of cosines).

    A 2-link arm has two solutions for most targets (elbow on either side).
    We return the one closest to the current joint angles, so the arm never
    "flips" its elbow unnecessarily.
    """
    if not reachable(x, y, l1, l2, margin=0.0):
        raise ValueError(f"Target ({x:.3f}, {y:.3f}) is outside the workspace")

    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)

    solutions = []
    for sign in (+1, -1):
        s2 = sign * np.sqrt(1 - c2 * c2)
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
        solutions.append(np.array([q1, q2]))

    if q_current is None:
        return solutions[0]

    # Unwrap q1 so it is the closest equivalent angle to the current one
    best = None
    for sol in solutions:
        sol = sol.copy()
        sol[0] = q_current[0] + (sol[0] - q_current[0] + np.pi) % (2 * np.pi) - np.pi
        if best is None or np.abs(sol - q_current).max() < np.abs(best - q_current).max():
            best = sol
    return best
