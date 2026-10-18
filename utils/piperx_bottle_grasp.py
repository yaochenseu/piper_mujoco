"""Fit the visible lower bottle wall; retain the model's scores and jaw yaw."""
import numpy as np


def fit_bottle_body(points):
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 50:
        raise RuntimeError("Too few bottle points for a body fit")
    base = float(np.quantile(points[:, 2], 0.02))
    wall = points[(points[:, 2] >= base + 0.004) & (points[:, 2] <= base + 0.025)]
    if len(wall) < 30:
        raise RuntimeError("Bottle lower wall is not visible")
    xy = wall[:, :2]
    origin = xy.mean(axis=0)
    delta = xy - origin
    design = np.column_stack((2 * delta, np.ones(len(delta))))
    solution, _, rank, _ = np.linalg.lstsq(design, (delta ** 2).sum(axis=1), rcond=None)
    center = origin + solution[:2]
    radius_squared = solution[2] + np.dot(solution[:2], solution[:2])
    radius = float(np.sqrt(max(0., radius_squared)))
    residual = float(np.median(np.abs(np.linalg.norm(xy - center, axis=1) - radius)))
    if rank < 3 or not 0.006 <= radius <= 0.04 or residual > 0.003:
        raise RuntimeError("Visible points do not support a cylindrical bottle body")
    # Most pixels can belong to the neck/top. Use the lowest visible wall
    # samples for contact height rather than a quantile biased towards it.
    contact_z = float(np.quantile(points[:, 2], 0.002)) + 0.020
    return {"center": [float(center[0]), float(center[1]), contact_z],
            "radius_m": radius, "median_residual_m": residual, "wall_points": len(wall),
            "source": "SAM RGB-D lower wall, circle fit; no simulator object pose"}


def refine_bottle_grasps(grasps, camera_pose, fit):
    """Return new camera-frame grasps before NMS and collision filtering.

    A vertical approach avoids grazing the neck. The contact plane is the
    AGX TCP plane, so insertion depth is zero for the collision proxy.
    """
    rotation = np.asarray(camera_pose)[:3, :3]
    translation = np.asarray(camera_pose)[:3, 3]
    world_rotations = rotation @ grasps.rotation_matrices
    jaw = world_rotations[:, :, 1].copy()
    jaw[:, 2] = 0
    lengths = np.linalg.norm(jaw, axis=1)
    valid = lengths > 0.2
    refined = grasps[valid]
    jaw = jaw[valid] / lengths[valid, None]
    approach = np.tile([0., 0., -1.], (len(refined), 1))
    world = np.stack((approach, jaw, np.cross(approach, jaw)), axis=2)
    refined.rotation_matrices = rotation.T @ world
    optical_center = rotation.T @ (np.asarray(fit["center"]) - translation)
    refined.translations = np.tile(optical_center, (len(refined), 1))
    refined.widths = np.full(len(refined), min(0.095, 2 * fit["radius_m"] + 0.012))
    refined.depths = np.zeros(len(refined))
    return refined
