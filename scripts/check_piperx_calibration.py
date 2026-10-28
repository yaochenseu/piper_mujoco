#!/usr/bin/env python3
"""Read-only check that the configured hand-eye calibration is consistent.

The robot base sits on the table, so after transforming the wrist camera cloud
into base coordinates the dominant table plane must be nearly z=0 and nearly
horizontal.  A stale hand-eye transform shows up as a tilted/offset plane.
This script never opens the control gate and never commands motion.
"""
from __future__ import annotations

import base64
from pathlib import Path
import sys
import zlib

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.piperx_real_env import RosBridgeClient  # noqa: E402


def fit_plane(points, seed=0, iterations=500, threshold=0.005):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) < threshold
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, normal.copy(), offset, inliers.copy())
    if best is None:
        raise RuntimeError("could not fit a table plane")
    count, normal, offset, inliers = best
    if normal[2] < 0:
        normal = -normal
        offset = -offset
    return count, normal, offset, inliers


def main():
    config_path = ROOT / "configs/real_piperx.yaml"
    config = yaml.safe_load(config_path.read_text())
    handeye = Path(config["camera"]["handeye_json"])
    if not handeye.is_absolute():
        handeye = (ROOT / handeye).resolve()
    calib_dir = ROOT.parent / "piper_control/tools/calib_results"
    latest = max(calib_dir.glob("*_handeye.json"), key=lambda p: p.stat().st_mtime)
    print(f"configured hand-eye: {handeye}")
    print(f"newest hand-eye:     {latest}")
    if handeye.name != latest.name:
        print("WARNING: config is not using the newest *_handeye.json file")

    bridge = config["bridge"]
    client = RosBridgeClient(
        bridge["socket_path"], timeout=bridge.get("request_timeout_s", 45.0))
    try:
        state = client.request("state")
        joints = [
            np.degrees(state["arm_rad"][f"joint{index}"])
            for index in range(1, 7)]
        print("joints_deg:", np.round(joints, 2).tolist())
        print("arm_status:", state.get("arm_status"))
        data = client.request("camera")
    finally:
        client.close()

    depth = np.frombuffer(
        zlib.decompress(base64.b64decode(data["depth_zlib_b64"])),
        dtype=np.float32,
    ).reshape(int(data["height"]), int(data["width"])).copy()
    transform = np.asarray(data["T_base_camera"], dtype=float)
    fx, fy = float(data["fx"]), float(data["fy"])
    ppx, ppy = float(data["ppx"]), float(data["ppy"])

    height, width = depth.shape
    v, u = np.mgrid[0:height:4, 0:width:4]
    z = depth[::4, ::4]
    valid = np.isfinite(z) & (z > 0.20) & (z < 1.5)
    u, v, z = u[valid], v[valid], z[valid]
    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy
    points = np.stack([x, y, z], axis=1)
    points = points @ transform[:3, :3].T + transform[:3, 3]
    if len(points) < 1000:
        raise RuntimeError(
            f"too few camera points ({len(points)}); is the camera running?")

    count, normal, offset, inliers = fit_plane(points)
    z_origin = float(-offset / normal[2])
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    inlier_z = float(points[inliers, 2].mean())
    print(f"table plane: inliers={count}/{len(points)}")
    print(f"  normal={np.round(normal, 4).tolist()}")
    print(f"  z_at_origin={z_origin:.4f} m")
    print(f"  tilt_from_base_z={tilt:.3f} deg")
    print(f"  inlier_mean_z={inlier_z:.4f} m")
    ok = tilt < 1.5 and abs(z_origin) < 0.010 and abs(inlier_z) < 0.008
    print("CALIBRATION OK" if ok else
          "CALIBRATION LOOKS OFF — do not grasp; re-check/re-run hand-eye")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
