"""Piper X pick/place with contact physics, no vision weights required."""
import argparse
import json

import numpy as np

from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv, TOP_DOWN
from utils.piperx_semantic import DEFAULT_OBJECT, target_choices


def run_demo(env, *, drop_xy=(0.42, 0.16), log=print):
    env.wait(0.4)
    initial = env.object_position
    log(f"Target: {initial.round(4).tolist()}")
    # This baseline uses the simulated object's known centre. The grasp pose
    # can later be supplied by RGB-D / GraspNet without changing execution.
    grasp = initial.copy()
    grasp[2] += 0.003
    above = grasp + [0, 0, 0.06]
    yaw = np.deg2rad(env.target_spec.grasp_yaw_deg)
    grasp_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                               [np.sin(yaw), np.cos(yaw), 0.0],
                               [0.0, 0.0, 1.0]]) @ TOP_DOWN
    log("1/7 Approach")
    env.gripper_control(0.1)
    env.move_to(above, grasp_rotation)
    log("2/7 Descend")
    env.move_to(grasp, grasp_rotation, duration=1.5)
    log("3/7 Close gripper")
    env.gripper_control(0.0)
    log("4/7 Lift")
    env.move_to(above, grasp_rotation)
    lifted = env.object_position
    lift_height = float(lifted[2] - initial[2])
    if lift_height < 0.04 or not env.bilateral_contacts:
        raise RuntimeError(f"Grasp failed: lift={lift_height:.4f} m, "
                           f"bilateral contact={env.bilateral_contacts}")
    log(f"Lifted {lift_height:.3f} m by finger contact")
    log("5/7 Transfer")
    destination = np.array([*drop_xy, above[2]])
    env.move_to(destination, grasp_rotation)
    log("6/7 Place and release")
    destination[2] = grasp[2] + 0.004
    # A loaded wrist can settle a few millimetres above the planned release
    # pose.  Position and attitude tolerances are relaxed only for placement.
    env.move_to(destination, grasp_rotation, duration=1.5, position_tolerance=0.025,
                angle_tolerance=0.06)
    env.gripper_control(0.1)
    log("7/7 Retreat")
    env.move_to(destination + [0, 0, 0.06], grasp_rotation)
    env.wait(0.5)
    final = env.object_position
    xy_error = float(np.linalg.norm(final[:2] - drop_xy))
    success = xy_error < 0.04 and abs(final[2] - initial[2]) < 0.015
    report = dict(success=bool(success), target=env.target_key, lift_height_m=lift_height,
                  bilateral_contact=bool(env.bilateral_contacts),
                  initial_position=initial.tolist(), final_position=final.tolist(),
                  placement_error_m=xy_error, simulation_time_s=float(env.data.time))
    log(json.dumps(report, indent=2))
    if not success:
        raise RuntimeError("Object did not settle inside the placement region")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run physics without opening a viewer")
    parser.add_argument("--fast", action="store_true", help="Do not synchronize viewer to real time")
    parser.add_argument("--hold", action="store_true", help="Keep viewer open after the grasp")
    parser.add_argument("--target", choices=target_choices(), default=DEFAULT_OBJECT,
                        help="Semantic tabletop object to pick and place")
    args = parser.parse_args()
    try:
        with PiperXGraspEnv(viewer=not args.headless, realtime=not args.fast,
                           target=args.target) as env:
            run_demo(env)
            if args.hold and not args.headless:
                while env.viewer.is_running():
                    env.step()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
