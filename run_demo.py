"""Run one pick-and-place episode and save the demo outputs to results/.

  results/camera_detections.png  - camera view with OpenCV detections
  results/demo.gif               - animation of the arm sorting the cubes
  results/joint_tracking.png     - reference vs actual joint angles

Usage:  python run_demo.py [--seed 7]
"""
import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from pickplace import kinematics as kin
from pickplace import sim, vision

COLOR_HEX = {"red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4"}


def save_animation(ep, path):
    log = ep.log
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    frames = range(0, len(log["t"]), 2)  # 25 fps

    def draw(k):
        ax.clear()
        ax.set_xlim(-0.35, 0.60)
        ax.set_ylim(-0.30, 0.62)
        ax.set_aspect("equal")
        ax.set_title(f"Vision-guided pick and place   t = {log['t'][k]:.1f} s", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.add_patch(plt.Rectangle((vision.X_RANGE[0], vision.Y_RANGE[0]),
                                   vision.X_RANGE[1] - vision.X_RANGE[0],
                                   vision.Y_RANGE[1] - vision.Y_RANGE[0],
                                   fc="#eeeeee", ec="#999999", ls="--", lw=1))
        ax.text(vision.X_RANGE[0] + 0.005, vision.Y_RANGE[0] + 0.01, "camera view", fontsize=7, color="#777")
        for color, (bx, by) in sim.BINS.items():
            ax.add_patch(plt.Circle((bx, by), sim.BIN_RADIUS, fc="none", ec=COLOR_HEX[color], lw=2))
        for i, p in enumerate(log["obj_pos"][k]):
            c = COLOR_HEX[ep.objects[i]["color"]]
            s = vision.OBJECT_SIZE
            ax.add_patch(plt.Rectangle((p[0] - s / 2, p[1] - s / 2), s, s, fc=c, ec="k", lw=0.5, zorder=3))
        elbow, ee = kin.forward(log["q"][k])
        ax.plot([0, elbow[0], ee[0]], [0, elbow[1], ee[1]], "-o", color="#333", lw=5, ms=8, zorder=4)
        _, ee_ref = kin.forward(log["q_ref"][k])
        ax.plot(*ee_ref, "x", color="orange", ms=8, zorder=5)
        held = log["held"][k] is not None
        ax.plot(*ee, "o", ms=12, mfc="none", mec="gold" if held else "#333", mew=2, zorder=5)

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(path, writer=PillowWriter(fps=25))
    plt.close(fig)


def save_tracking_plot(ep, path):
    t = np.array(ep.log["t"])
    q = np.degrees(np.array(ep.log["q"]))
    q_ref = np.degrees(np.array(ep.log["q_ref"]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, q_ref[:, j], "--", color="orange", label="reference")
        ax.plot(t, q[:, j], color="#333", lw=1, label="actual")
        ax.set_ylabel(f"joint {j + 1} [deg]")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("time [s]")
    fig.suptitle("Joint tracking: quintic trajectory + PID with dynamics feedforward")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    objects = sim.random_scene(rng, n_min=5, n_max=5)
    ep = sim.Episode(objects, rng, strategy="greedy", record=True)
    result = ep.run()

    cv2.imwrite("results/camera_detections.png", vision.draw_detections(ep.image, ep.detections))
    save_tracking_plot(ep, "results/joint_tracking.png")
    save_animation(ep, "results/demo.gif")

    for key, value in result.items():
        print(f"{key:28s} {value:.2f}" if isinstance(value, float) else f"{key:28s} {value}")
    print("Saved results/camera_detections.png, results/joint_tracking.png, results/demo.gif")


if __name__ == "__main__":
    main()
