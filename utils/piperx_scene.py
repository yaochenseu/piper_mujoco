"""Scene-level free-body discovery and point-cloud matching for PiperX."""
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_ASSET_DIR = ROOT / "manipulator_grasp/assets/agilex_piper_x"
TABLE_TOP_Z = 0.740
OBJECT_CLEARANCE_Z = 0.002


def scene_bodies(env):
    """Return {body_name: free_joint_name} for every free joint in the scene."""
    bodies = {}
    for joint_id in range(int(env.model.njnt)):
        if int(env.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        body_id = int(env.model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if body_name:
            bodies[body_name] = joint_name
    return bodies


def resolve_scene_body(env, object_points, *, max_distance_m=0.06):
    """Match a segmented point cloud to the nearest scene body in the XY plane."""
    points = np.asarray(object_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("object_points must be a non-empty (N, 3) array")
    centroid = points.mean(axis=0)
    candidates = []
    for body_name, joint_name in scene_bodies(env).items():
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        position = env.data.xipos[body_id].copy()
        distance = float(np.linalg.norm(position[:2] - centroid[:2]))
        candidates.append({
            "body": body_name,
            "joint": joint_name,
            "position": position.tolist(),
            "distance_m": distance,
        })
    candidates.sort(key=lambda entry: entry["distance_m"])
    report = {
        "centroid": centroid.tolist(),
        "max_distance_m": float(max_distance_m),
        "candidates": candidates,
    }
    if not candidates:
        return None, float("inf"), report
    nearest = candidates[0]
    if nearest["distance_m"] > max_distance_m:
        return None, nearest["distance_m"], report
    return nearest["body"], nearest["distance_m"], report
