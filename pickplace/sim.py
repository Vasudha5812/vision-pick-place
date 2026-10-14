"""Pick-and-place task: scene generation, task ordering and closed-loop execution."""
import itertools

import numpy as np

from . import kinematics as kin
from . import trajectory as traj
from . import vision
from .control import JointPID
from .dynamics import Arm

DT = 0.001             # simulation step [s]
GRASP_TIME = 0.25      # gripper close (lower + suction + lift abstracted) [s]
RELEASE_TIME = 0.20    # gripper open [s]
GRASP_TOL = 0.012      # suction cup must be within 12 mm of the object centre
BINS = {"red": (0.00, 0.45), "green": (0.25, 0.40), "blue": (-0.25, 0.40)}
BIN_RADIUS = 0.04
HOME = np.array([0.10, 0.45])  # home position of the end-effector [m]


def random_scene(rng, n_min=3, n_max=6, min_gap=0.06):
    """Random non-overlapping cubes in the pick area."""
    n = rng.integers(n_min, n_max + 1)
    objects = []
    while len(objects) < n:
        x, y = rng.uniform(0.22, 0.44), rng.uniform(-0.18, 0.08)
        if all(np.hypot(x - o["x"], y - o["y"]) > min_gap for o in objects):
            objects.append({"x": x, "y": y,
                            "color": rng.choice(list(BINS)),
                            "angle": rng.uniform(0, 90)})
    return objects


def _move_time(p_to, q_from):
    q_to = kin.inverse(*p_to, q_current=q_from)
    return traj.duration(q_from, q_to), q_to


def plan_time(order, targets, start_q):
    """Planned cycle time for visiting targets in the given order."""
    q = start_q
    total = 0.0
    for i in order:
        obj = targets[i]
        t1, q = _move_time((obj["x"], obj["y"]), q)
        t2, q = _move_time(BINS[obj["color"]], q)
        total += t1 + GRASP_TIME + t2 + RELEASE_TIME
    return total


def order_targets(targets, start_q, strategy):
    """Choose the order in which to pick objects.

    naive   - the order the vision system reported them
    greedy  - always go to the object that is quickest to reach next
    optimal - try every permutation (fine for up to ~7 objects)
    """
    idx = list(range(len(targets)))
    if strategy == "naive" or len(idx) < 2:
        return idx
    if strategy == "optimal":
        return list(min(itertools.permutations(idx),
                        key=lambda o: plan_time(o, targets, start_q)))
    if strategy == "greedy":
        order, q, remaining = [], start_q, set(idx)
        while remaining:
            best = min(remaining, key=lambda i: _move_time(
                (targets[i]["x"], targets[i]["y"]), q)[0])
            order.append(best)
            remaining.remove(best)
            _, q = _move_time((targets[best]["x"], targets[best]["y"]), q)
            _, q = _move_time(BINS[targets[best]["color"]], q)
        return order
    raise ValueError(strategy)


class Episode:
    """Runs one full pick-and-place episode and records everything."""

    def __init__(self, objects, rng, strategy="greedy", feedforward=True, record=False):
        self.objects = objects
        self.rng = rng
        self.strategy = strategy
        self.arm = Arm(kin.inverse(*HOME))
        self.pid = JointPID(feedforward=feedforward)
        self.record = record
        self.log = {"t": [], "q": [], "q_ref": [], "held": [], "obj_pos": []}
        self.t = 0.0
        self.held = None                 # index into self.objects
        self.obj_pos = [np.array([o["x"], o["y"]]) for o in objects]
        self.picked_ok = 0
        self.placed_ok = 0
        self.max_track_err = 0.0
        self.track_err_sq, self.track_n = 0.0, 0

    # ---------- low-level motion ----------
    def _run(self, q_ref_fn, T):
        steps = int(round(T / DT))
        for _ in range(steps):
            q_ref, qd_ref, qdd_ref = q_ref_fn(self.t_local)
            tau = self.pid(self.arm.q, self.arm.qd, q_ref, qd_ref, qdd_ref, DT)
            self.arm.step(tau, DT)
            self.t += DT
            self.t_local += DT
            if self.held is not None:
                self.obj_pos[self.held] = kin.forward(self.arm.q)[1].copy()
            ee_err = np.linalg.norm(kin.forward(self.arm.q)[1] - kin.forward(q_ref)[1])
            self.max_track_err = max(self.max_track_err, ee_err)
            self.track_err_sq += ee_err**2
            self.track_n += 1
            if self.record and int(round(self.t / DT)) % 20 == 0:
                self.log["t"].append(self.t)
                self.log["q"].append(self.arm.q.copy())
                self.log["q_ref"].append(np.array(q_ref))
                self.log["held"].append(self.held)
                self.log["obj_pos"].append([p.copy() for p in self.obj_pos])

    def move_to(self, x, y):
        q0 = self.arm.q.copy()
        qf = kin.inverse(x, y, q_current=q0)
        T = traj.duration(q0, qf)
        self.t_local = 0.0
        self._run(lambda t: traj.quintic(q0, qf, T, t), T)

    def hold(self, T):
        """Hold position at the current target (gripper opening/closing)."""
        q_hold = kin.inverse(*self._last_target, q_current=self.arm.q)
        self.t_local = 0.0
        self._run(lambda t: (q_hold, np.zeros(2), np.zeros(2)), T)

    # ---------- task level ----------
    def run(self):
        img = vision.render(self.objects, self.rng)
        H = vision.calibrate(img)
        detections = vision.detect(img, H)
        self.image, self.detections = img, detections

        start_q = self.arm.q.copy()
        order = order_targets(detections, start_q, self.strategy)

        for i in order:
            det = detections[i]
            self._last_target = (det["x"], det["y"])
            self.move_to(*self._last_target)
            self.hold(GRASP_TIME)                     # gripper closes
            ee = kin.forward(self.arm.q)[1]
            dists = [np.linalg.norm(ee - p) for p in self.obj_pos]
            j = int(np.argmin(dists))
            grasped = dists[j] < GRASP_TOL
            if grasped:
                self.held = j
                self.picked_ok += 1
            self._last_target = BINS[det["color"]]
            self.move_to(*self._last_target)
            self.hold(RELEASE_TIME)                   # gripper opens
            if self.held is not None:
                in_bin = np.linalg.norm(self.obj_pos[j] - np.array(BINS[det["color"]])) < BIN_RADIUS
                right_bin = self.objects[j]["color"] == det["color"]
                self.placed_ok += int(in_bin and right_bin)
                self.held = None
        self._last_target = tuple(HOME)
        self.move_to(*HOME)

        rms = np.sqrt(self.track_err_sq / max(self.track_n, 1))
        loc_err = _localisation_error(self.objects, detections)
        return {
            "n_objects": len(self.objects),
            "n_detected": len(detections),
            "placed_ok": self.placed_ok,
            "cycle_time_s": self.t,
            "rms_tracking_error_mm": rms * 1000,
            "max_tracking_error_mm": self.max_track_err * 1000,
            "mean_localisation_error_mm": loc_err * 1000,
        }


def _localisation_error(objects, detections):
    errs = []
    for d in detections:
        errs.append(min(np.hypot(d["x"] - o["x"], d["y"] - o["y"]) for o in objects))
    return float(np.mean(errs)) if errs else float("nan")
