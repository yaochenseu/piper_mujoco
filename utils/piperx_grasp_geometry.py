"""Schematic TCP/gripper geometry for visualization (no MuJoCo dependency)."""

import numpy as np


def grasp_segments(T, width):
    """Return RGB axes, cyan gripper lines, and the yellow approach segment."""
    A = np.asarray(T, dtype=float)
    if A.shape != (4, 4) or not np.isfinite(A).all():
        raise ValueError("T must be a finite 4x4 matrix")
    R, p = A[:3, :3], A[:3, 3]
    if not np.allclose(R.T @ R, np.eye(3), rtol=0.0, atol=1e-6) or not np.isclose(
        np.linalg.det(R), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError("T must contain a proper rotation")

    w = np.asarray(width)
    if w.shape != () or w.dtype.kind not in "iuf":
        raise ValueError("width must be a real scalar")
    w = float(w)
    if not np.isfinite(w) or not 0.0 < w <= 0.1:
        raise ValueError("width must satisfy 0 < width <= 0.1")

    def seg(a, b, rgba, kind, label):
        return {
            "start": np.asarray(a, dtype=float),
            "end": np.asarray(b, dtype=float),
            "rgba": rgba,
            "kind": kind,
            "label": label,
        }

    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    out = [
        seg(p, p + 0.05 * x, (1.0, 0.0, 0.0, 1.0), "axis", "X"),
        seg(p, p + 0.05 * y, (0.0, 1.0, 0.0, 1.0), "axis", "Y"),
        seg(p, p + 0.05 * z, (0.0, 0.0, 1.0, 1.0), "axis", "Z"),
    ]
    cyan = (0.0, 1.0, 1.0, 1.0)
    back, front = p - 0.06 * z, p + 0.005 * z
    for sign, label in ((-1.0, "finger_left"), (1.0, "finger_right")):
        offset = sign * (w / 2.0) * y
        out.append(seg(back + offset, front + offset, cyan, "gripper", label))
    out.append(seg(back - (w / 2.0) * y, back + (w / 2.0) * y, cyan, "gripper", "backbar"))
    out.append(
        seg(
            p - 0.08 * z,
            p - 0.012 * z,
            (1.0, 1.0, 0.0, 1.0),
            "approach",
            "approach",
        )
    )
    return out
