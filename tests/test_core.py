"""Basic checks. Run with:  python -m pytest tests   (or: python tests/test_core.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pickplace import kinematics as kin
from pickplace import sim, trajectory, vision


def test_ik_inverts_fk():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform([-np.pi, -2.8], [np.pi, 2.8])
        x, y = kin.forward(q)[1]
        q_ik = kin.inverse(x, y, q_current=q)
        assert np.allclose(kin.forward(q_ik)[1], [x, y], atol=1e-9)


def test_unreachable_target_raises():
    try:
        kin.inverse(1.0, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_jacobian_matches_finite_difference():
    q = np.array([0.4, 1.0])
    eps = 1e-6
    numeric = np.column_stack([
        (kin.forward(q + eps * np.eye(2)[i])[1] - kin.forward(q)[1]) / eps for i in range(2)
    ])
    assert np.allclose(kin.jacobian(q), numeric, atol=1e-5)


def test_quintic_boundary_conditions():
    q0, qf, T = np.array([0.0, 1.0]), np.array([1.0, -0.5]), 2.0
    p, v, a = trajectory.quintic(q0, qf, T, 0.0)
    assert np.allclose(p, q0) and np.allclose(v, 0) and np.allclose(a, 0)
    p, v, a = trajectory.quintic(q0, qf, T, T)
    assert np.allclose(p, qf) and np.allclose(v, 0)


def test_vision_localises_objects():
    rng = np.random.default_rng(3)
    objects = sim.random_scene(rng)
    img = vision.render(objects, rng)
    detections = vision.detect(img, vision.calibrate(img))
    assert len(detections) == len(objects)
    for d in detections:
        err = min(np.hypot(d["x"] - o["x"], d["y"] - o["y"]) for o in objects)
        assert err < 0.003  # within 3 mm


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
