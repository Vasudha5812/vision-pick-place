"""Overhead camera: renders the table and detects coloured objects with OpenCV.

Pipeline:
  1. Detect the 4 black calibration markers and compute a homography
     (pixel -> world) with cv2.findHomography.
  2. Threshold each colour in HSV space, clean the mask with morphology,
     find contours and take each blob's centroid.
  3. Convert centroids from pixels to world coordinates (metres).
"""
import cv2
import numpy as np

# Table region seen by the camera (world metres)
X_RANGE = (0.15, 0.50)
Y_RANGE = (-0.25, 0.15)
PX_PER_M = 1000  # resolution of the internal top-down render

ORTHO_W = int((Y_RANGE[1] - Y_RANGE[0]) * PX_PER_M)  # 400
ORTHO_H = int((X_RANGE[1] - X_RANGE[0]) * PX_PER_M)  # 350
CAM_W, CAM_H = 640, 480

# Calibration markers in world coordinates, ordered as they appear in the
# image: top-left, top-right, bottom-left, bottom-right.
MARKERS_WORLD = np.array([
    [0.48, -0.23], [0.48, 0.13], [0.17, -0.23], [0.17, 0.13],
], dtype=np.float32)

# The camera is tilted, so the table appears as a trapezoid (perspective).
_SRC = np.float32([[0, 0], [ORTHO_W, 0], [ORTHO_W, ORTHO_H], [0, ORTHO_H]])
_DST = np.float32([[120, 60], [520, 60], [600, 440], [40, 440]])
H_CAMERA = cv2.getPerspectiveTransform(_SRC, _DST)

COLORS_BGR = {"red": (40, 40, 210), "green": (60, 170, 50), "blue": (200, 90, 30)}
HSV_RANGES = {
    "red": [((0, 120, 80), (10, 255, 255)), ((170, 120, 80), (180, 255, 255))],
    "green": [((40, 100, 60), (85, 255, 255))],
    "blue": [((100, 120, 60), (130, 255, 255))],
}
OBJECT_SIZE = 0.03  # cube side [m]


def _world_to_ortho(x, y):
    return (y - Y_RANGE[0]) * PX_PER_M, (X_RANGE[1] - x) * PX_PER_M


def render(objects, rng):
    """Render a camera image (BGR) of the table with the given objects.

    objects: list of dicts with keys x, y, color, angle.
    """
    # Table surface with uneven lighting
    gx = np.linspace(0, 1, ORTHO_W)[None, :]
    gy = np.linspace(0, 1, ORTHO_H)[:, None]
    shade = 175 + 45 * gx - 25 * gy
    table = np.dstack([shade * 0.95, shade, shade * 1.02]).clip(0, 255).astype(np.uint8)

    for mx, my in MARKERS_WORLD:
        u, v = _world_to_ortho(mx, my)
        cv2.circle(table, (int(u), int(v)), 10, (20, 20, 20), -1, cv2.LINE_AA)

    half = OBJECT_SIZE / 2 * PX_PER_M
    for obj in objects:
        u, v = _world_to_ortho(obj["x"], obj["y"])
        box = cv2.boxPoints(((u, v), (2 * half, 2 * half), obj["angle"]))
        cv2.fillPoly(table, [box.astype(np.int32)], COLORS_BGR[obj["color"]], cv2.LINE_AA)

    img = cv2.warpPerspective(table, H_CAMERA, (CAM_W, CAM_H),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 70, 70))
    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _order_markers(points):
    """Sort 4 points into top-left, top-right, bottom-left, bottom-right."""
    pts = sorted(points, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return np.array(top + bottom, dtype=np.float32)


def calibrate(img):
    """Find the black markers and return the pixel -> world homography."""
    hsv = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 90, 60))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 60:
            continue
        # Circularity of the convex hull ignores ragged, noisy edges
        hull = cv2.convexHull(c)
        perimeter = cv2.arcLength(hull, True)
        circularity = 4 * np.pi * cv2.contourArea(hull) / (perimeter**2 + 1e-9)
        if circularity > 0.75:
            m = cv2.moments(c)
            candidates.append((area, (m["m10"] / m["m00"], m["m01"] / m["m00"])))

    if len(candidates) < 4:
        raise RuntimeError("Calibration failed: fewer than 4 markers found")
    candidates.sort(reverse=True)
    pixels = _order_markers([p for _, p in candidates[:4]])
    H, _ = cv2.findHomography(pixels, MARKERS_WORLD)
    return H


def pixel_to_world(H, u, v):
    p = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def detect(img, H):
    """Return detected objects as dicts: color, x, y (world), u, v (pixels)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    detections = []
    for color, ranges in HSV_RANGES.items():
        mask = np.zeros(img.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 150:
                continue
            m = cv2.moments(c)
            u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
            x, y = pixel_to_world(H, u, v)
            detections.append({"color": color, "x": x, "y": y, "u": u, "v": v})
    return detections


def draw_detections(img, detections):
    out = img.copy()
    for d in detections:
        u, v = int(d["u"]), int(d["v"])
        cv2.drawMarker(out, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        label = f'{d["color"]} ({d["x"]*100:.1f}, {d["y"]*100:.1f}) cm'
        cv2.putText(out, label, (u + 12, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (u + 12, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
