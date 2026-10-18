"""Replay one saved GraspNet candidate in the CPU MuJoCo scene.

The script recreates the object pose that produced the saved RGB-D frame,
executes approach/descend/close/lift, and records the jaw width and current
contacts at each stage.  It does not read the object pose for control.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv
from utils.piperx_vision_control import (
    parallel_jaw_rotation_variants,
    plan_pick_place,
)


DEFAULT_GRASPS = ROOT / "docs/vision_runs/vlm_semantic_banana/full/cycle_00/grasps.json"
DEFAULT_RESULT = ROOT / "docs/vision_runs/vlm_semantic_banana/full/result.json"
DEFAULT_OUTPUT = ROOT / "docs/piperx_rank9_cpu_check.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasps-json", type=Path, default=DEFAULT_GRASPS)
    parser.add_argument("--reference-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--rank", type=int, default=9)
    parser.add_argument("--branch", choices=("near", "far"), default="near")
    parser.add_argument("--depth-offset", type=float, default=0.0,
                        help="Experiment only: shift TCP along the approach axis")
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def contact_rows(env):
    rows = []
    force = np.zeros(6)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        names = [env.model.geom(contact.geom1).name or "",
                 env.model.geom(contact.geom2).name or ""]
        if not any(name.startswith("gripper_link") for name in names):
            continue
        mujoco.mj_contactForce(env.model, env.data, index, force)
        rows.append({
            "geoms": names,
            "distance_m": float(contact.dist),
            "position_world": np.asarray(contact.pos).tolist(),
            "normal_force_n": float(np.linalg.norm(force[:3])),
        })
    return rows


def snapshot(env):
    pads = {}
    for name in ("gripper_link1_pad", "gripper_link2_pad"):
        geom_id = env.model.geom(name).id
        pads[name] = np.asarray(env.data.geom_xpos[geom_id]).tolist()
    return {
        "tcp_pose": np.asarray(env.tcp_pose).tolist(),
        "object_xipos": np.asarray(env.object_position).tolist(),
        "gripper_width_m": float(env.gripper_width),
        "finger_contact": env.finger_contact_state(),
        "pad_centres_world": pads,
        "finger_contacts": contact_rows(env),
    }


def main():
    args = parse_args()
    grasps = json.loads(args.grasps_json.read_text())["candidates"]
    entry = next((item for item in grasps if item.get("rank") == args.rank), None)
    if entry is None:
        raise SystemExit(f"rank {args.rank} is not present in {args.grasps_json}")
    reference = None
    if args.reference_result.is_file():
        reference = json.loads(args.reference_result.read_text()).get("initial_position")

    env = PiperXGraspEnv(realtime=False, viewer=False, target="banana")
    try:
        env.wait(args.settle_seconds)
        address = env.model.joint(env.target_free_joint_name).qposadr[0]
        dof_address = env.model.joint(env.target_free_joint_name).dofadr[0]
        settled_qpos = env.data.qpos[address:address + 7].copy()
        env.data.qpos[address:address + 7] = settled_qpos
        env.data.qvel[dof_address:dof_address + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)

        home_pose = env.tcp_pose.copy()
        base_tcp = np.asarray(entry["world_tcp"], dtype=float)
        variants = parallel_jaw_rotation_variants(
            base_tcp[:3, :3], reference=home_pose[:3, :3])
        variant = next(item for item in variants if item["branch"] == args.branch)
        tcp = base_tcp.copy()
        tcp[:3, :3] = variant["rotation"]
        tcp[:3, 3] += args.depth_offset * variant["rotation"][:, 2]

        result = {
            "grasps_json": str(args.grasps_json),
            "rank": args.rank,
            "score": entry.get("score"),
            "grasp_width_m": entry.get("width"),
            "jaw_variant": variant["name"],
            "jaw_branch": variant["branch"],
            "depth_offset_m": args.depth_offset,
            "settled_free_joint_qpos": settled_qpos.tolist(),
            "object_xipos_before": env.object_position.tolist(),
            "reference_initial_position": reference,
        }
        if reference is not None:
            result["reference_xipos_error_m"] = float(
                np.linalg.norm(env.object_position - np.asarray(reference)))
        result["initial"] = snapshot(env)

        try:
            plans = plan_pick_place(env, tcp, (0.22, -0.34), home_pose)
        except RuntimeError as error:
            result["plan_error"] = str(error)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            return 1

        result["planned_segments"] = {name: len(plan["goals"])
                                      for name, plan in plans.items()}
        result["commanded_tcp"] = tcp.tolist()
        env.execute_plan(plans["approach"])
        result["approach"] = snapshot(env)
        env.execute_plan(plans["descend"])
        result["descend"] = snapshot(env)
        env.gripper_control(0.0)
        result["closed"] = snapshot(env)
        lift_start_z = float(env.tcp_pose[2, 3])
        env.execute_plan(plans["lift"])
        result["lifted"] = snapshot(env)
        result["tcp_lift_m"] = float(env.tcp_pose[2, 3] - lift_start_z)
        result["object_lift_m"] = float(
            env.object_position[2] - result["object_xipos_before"][2])
        result["grasped"] = bool(
            result["object_lift_m"] >= 0.04 and
            result["lifted"]["finger_contact"]["bilateral"])

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({
            "rank": result["rank"],
            "jaw": f"{result['jaw_variant']}/{result['jaw_branch']}",
            "commanded_tcp": np.round(tcp[:3, 3], 5).tolist(),
            "descend_tcp": np.round(
                np.asarray(result["descend"]["tcp_pose"])[:3, 3], 5).tolist(),
            "closed_width_m": result["closed"]["gripper_width_m"],
            "closed_bilateral": result["closed"]["finger_contact"]["bilateral"],
            "object_lift_m": round(result["object_lift_m"], 5),
            "tcp_lift_m": round(result["tcp_lift_m"], 5),
            "lifted_bilateral": result["lifted"]["finger_contact"]["bilateral"],
            "grasped": result["grasped"],
        }, indent=2))
        print(f"Wrote: {args.output}")
        return 0 if result["grasped"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
