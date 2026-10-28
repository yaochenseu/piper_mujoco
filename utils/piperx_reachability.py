"""Kinematically check real grasp poses against the Piper X model.

The real bridge executes Cartesian moves, but its ``move_l`` path does not
report an unreachable wrist pose as an error: it can move part-way and then
time out with ``arm_status=0``.  Candidate selection therefore needs a model
IK preflight before any motion is sent.
"""
from __future__ import annotations

import numpy as np

from manipulator_grasp.env.piperx_grasp_env import (
    HOME_JOINTS_DEG,
    PiperXGraspEnv,
)


class Reachability:
    """Small, viewer-free MuJoCo model used only for IK feasibility checks."""

    def __init__(self, *, random_seeds=6):
        self.env = PiperXGraspEnv(
            viewer=False, realtime=False, render_size=(320, 240))
        self.base = np.asarray(
            self.env.data.xpos[self.env.model.body("base_link").id],
            dtype=float).copy()
        self.home = np.radians(HOME_JOINTS_DEG)
        self.limits = self.env.limits
        rng = np.random.default_rng(20261028)
        self.random_seeds = [
            rng.uniform(self.limits[:, 0], self.limits[:, 1])
            for _ in range(max(0, int(random_seeds)))
        ]

    def feasible(self, tcp, pregrasp, *, seed_joints=None):
        """Return whether both the pregrasp and final TCP poses have an IK solution."""
        tcp = np.asarray(tcp, dtype=float)
        pregrasp = np.asarray(pregrasp, dtype=float)
        if tcp.shape != (4, 4) or pregrasp.shape != (4, 4):
            raise ValueError("Expected two 4x4 TCP transforms")
        seeds = []
        if seed_joints is not None:
            seeds.append(np.asarray(seed_joints, dtype=float))
        seeds.extend([self.home.copy(), np.zeros(6)])
        seeds.extend(seed.copy() for seed in self.random_seeds)
        last_error = ""
        for label, pose in (("pregrasp", pregrasp), ("final", tcp)):
            if not np.isfinite(pose).all():
                return {"feasible": False, "failed_pose": label,
                        "error": "non-finite pose", "seeds_tried": 0}
            target = pose[:3, 3] + self.base
            rotation = pose[:3, :3]
            solved = False
            for seed in seeds:
                try:
                    self.env.solve_ik(target, rotation, seed=seed)
                    solved = True
                    break
                except Exception as error:  # IKError, invalid pose, etc.
                    last_error = str(error)
            if not solved:
                return {"feasible": False, "failed_pose": label,
                        "error": last_error, "seeds_tried": len(seeds)}
        return {"feasible": True, "failed_pose": None, "error": "",
                "seeds_tried": len(seeds)}

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False
