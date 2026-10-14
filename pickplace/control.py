"""Joint-space PID controller with optional model-based feedforward."""
import numpy as np

from . import dynamics


class JointPID:
    """Independent PID loop on each joint.

    Gains are chosen from a target closed-loop natural frequency (wn) and
    damping ratio (zeta), using each joint's nominal inertia J:
        Kp = J * wn^2,   Kd = 2 * zeta * wn * J,   Ki = Kp * wn / 10
    """

    def __init__(self, wn=30.0, zeta=1.0, feedforward=True):
        J = np.diag(dynamics.mass_matrix([0.0, np.pi / 2]))
        self.kp = J * wn**2
        self.kd = 2 * zeta * wn * J
        self.ki = self.kp * wn / 10
        self.feedforward = feedforward
        self.integral = np.zeros(2)
        self.i_limit = 0.05  # anti-windup clamp [rad s]

    def reset(self):
        self.integral[:] = 0.0

    def __call__(self, q, qd, q_ref, qd_ref, qdd_ref, dt):
        err = q_ref - q
        derr = qd_ref - qd
        self.integral = np.clip(self.integral + err * dt, -self.i_limit, self.i_limit)
        tau = self.kp * err + self.kd * derr + self.ki * self.integral
        if self.feedforward:
            tau = tau + dynamics.inverse_dynamics(q, qd, qdd_ref)
        return tau
