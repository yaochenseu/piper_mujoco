"""Real Piper X adapter for the vision pick/place policy.

The vision process runs in the conda environment.  A separate ROS 2 bridge
owns rclpy, the Piper driver topics and the D435i topics.  This module talks
to that bridge over a local Unix socket and exposes the subset of
PiperXGraspEnv used by the existing policy.
"""
from __future__ import annotations

import base64
import json
import socket
import time
from pathlib import Path
import zlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from utils.piperx_vision_control import inside_reach_sphere


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path):
    import yaml
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def _resolve_config(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return ROOT / "configs" / path
    return (ROOT / path).resolve()


def _as_transform(value, name):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError(f"{name} rotation must be orthonormal")
    return transform


def _rotation_angle(first, second):
    cosine = 0.5 * (np.trace(np.asarray(first).T @ np.asarray(second)) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


class RosBridgeClient:
    def __init__(self, socket_path, *, timeout=30.0):
        self.socket_path = str(socket_path)
        self.timeout = float(timeout)
        self._socket = None
        self._stream = None
        self._connect()

    def _connect(self):
        self.close()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
        except OSError as error:
            sock.close()
            raise RuntimeError(
                f"cannot connect to ROS bridge at {self.socket_path}; "
                "start scripts/piperx_real_bridge.sh first") from error
        self._socket = sock
        self._stream = sock.makefile("rwb")

    def request(self, command, **payload):
        if self._stream is None:
            self._connect()
        message = {"command": command, **payload}
        try:
            self._stream.write(
                (json.dumps(message, separators=(",", ":")) + "\n").encode())
            self._stream.flush()
            line = self._stream.readline()
        except OSError as error:
            self.close()
            raise RuntimeError(f"ROS bridge request failed: {error}") from error
        if not line:
            self.close()
            raise RuntimeError("ROS bridge closed the connection")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "ROS bridge request failed"))
        return response.get("data", {})

    def close(self):
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._stream = None
        self._socket = None


class RealPiperXEnv:
    is_real = True

    def __init__(self, *, config_path="real_piperx.yaml", execute=False,
                 open_gate=False, log=print, client=None):
        self.config_path = _resolve_config(config_path)
        self.config = _load_yaml(self.config_path)
        self.log = log
        self.execute = bool(execute)
        self.log = log
        bridge_cfg = self.config["bridge"]
        self.client = client or RosBridgeClient(
            bridge_cfg["socket_path"],
            timeout=bridge_cfg.get("request_timeout_s", 30.0),
        )
        self._opened_gate = False
        self._camera_snapshot = None
        self._state_snapshot = None
        self._state_time = 0.0

        handeye_path = Path(self.config["camera"]["handeye_json"])
        if not handeye_path.is_absolute():
            handeye_path = (ROOT / handeye_path).resolve()
        with handeye_path.open() as stream:
            handeye = json.load(stream)
        self.T_flange_cam = _as_transform(
            handeye["result"]["matrix4x4"], "T_flange_cam")

        tcp_xyz = np.asarray(
            self.config["arm"]["flange_tcp_xyz_m"], dtype=float)
        tcp_rpy = np.asarray(
            self.config["arm"]["flange_tcp_rpy_deg"], dtype=float)
        self.T_flange_tcp = np.eye(4)
        self.T_flange_tcp[:3, :3] = Rotation.from_euler(
            "xyz", tcp_rpy, degrees=True).as_matrix()
        self.T_flange_tcp[:3, 3] = tcp_xyz
        self.T_tcp_cam = np.linalg.inv(self.T_flange_tcp) @ self.T_flange_cam

        workspace = self.config["workspace"]
        self.table_top_z = float(workspace["table_top_z"])
        self.table_x = tuple(map(float, workspace["x"]))
        self.table_y = tuple(map(float, workspace["y"]))
        self.table_z = tuple(map(float, workspace["table_z"]))
        self.object_z = tuple(map(float, workspace["object_z"]))
        self.grasp_z = tuple(map(float, workspace["grasp_z"]))
        # URDF reach envelope: a ball centred on joint2 (the shoulder), not a
        # rectangle around the table.  Optional so older configs and tests can
        # still run without it.
        reach_radius = workspace.get("reach_radius_m")
        self.reach_center = None
        self.reach_radius_m = None
        self.reach_min_radius_m = 0.0
        if reach_radius is not None:
            center_xy = workspace.get("reach_center_xy", (0.0, 0.0))
            if len(center_xy) != 2:
                raise ValueError("workspace.reach_center_xy must contain two values")
            self.reach_center = np.array([
                float(center_xy[0]), float(center_xy[1]),
                float(workspace.get("reach_center_z", 0.123)),
            ], dtype=float)
            self.reach_radius_m = float(reach_radius)
            self.reach_min_radius_m = float(
                workspace.get("reach_min_radius_m", 0.0))
            if self.reach_radius_m <= 0.0:
                raise ValueError("workspace.reach_radius_m must be positive")
        self.max_command_distance_m = float(
            self.config["arm"]["max_command_distance_m"])
        self.max_command_rotation_rad = np.deg2rad(
            float(self.config["arm"]["max_command_rotation_deg"]))
        self.pregrasp_camera_heights = tuple(
            float(value) for value in
            self.config["arm"].get(
                "pregrasp_camera_heights_m", (0.955, 0.935, 0.915)))

        self.target_key = ""
        self.target_label = ""
        self.target_body = ""
        self.target_body_id = -1
        self.target_free_joint_name = ""
        self.viewer = None

        self.client.request("ping")
        if self.execute and open_gate:
            gate = self.client.request("open_gate", enable=True)
            if not gate.get("success", False):
                raise RuntimeError(
                    "failed to open /control_enable gate: " +
                    str(gate.get("message", gate)))
            self._opened_gate = True
        self.log(
            f"Real bridge ready: {self.client.socket_path}; "
            f"execute={self.execute}")

    def _state(self, *, max_age=0.15):
        if (self._state_snapshot is None or
                time.monotonic() - self._state_time > max_age):
            self._state_snapshot = self.client.request("state")
            self._state_time = time.monotonic()
        return self._state_snapshot

    @property
    def tcp_pose(self):
        state = self._state()
        return _as_transform(state["T_base_tcp"], "T_base_tcp").copy()

    @property
    def flange_pose(self):
        state = self._state()
        return _as_transform(state["T_base_flange"], "T_base_flange").copy()

    def get_joint(self):
        state = self._state()
        joints = state.get("arm_rad", {})
        values = []
        for index in range(1, 7):
            name = f"joint{index}"
            if name not in joints:
                raise RuntimeError(f"ROS bridge did not report {name}")
            values.append(float(joints[name]))
        return np.asarray(values, dtype=float)

    def joint_limit_violations(self):
        """Return (joint_index, value_deg, low_deg, high_deg) violations."""
        limits = self.config["arm"].get("joint_limits_deg")
        if not limits:
            return []
        values = np.degrees(self.get_joint())
        if len(limits) != len(values):
            raise ValueError("arm.joint_limits_deg must contain six pairs")
        violations = []
        for index, (value, bounds) in enumerate(zip(values, limits), start=1):
            low, high = map(float, bounds)
            if value < low or value > high:
                violations.append((index, float(value), low, high))
        return violations

    @property
    def gripper_width(self):
        value = self._state().get("gripper_width_m")
        return None if value is None else float(value)

    def finger_contact_state(self):
        state = self._state(max_age=0.0)
        width = state.get("gripper_width_m")
        force = state.get("gripper_force_n")
        width = 0.0 if width is None else float(width)
        # The AgX gripper reports a signed force; gripping a payload is
        # negative (e.g. -1.04 N).  The contact proxy must compare magnitude.
        force = 0.0 if force is None else abs(float(force))
        cfg = self.config["gripper"]
        bilateral = (
            force >= float(cfg["contact_force_threshold_n"]) and
            width >= float(cfg["contact_min_width_m"]))
        return {
            "left": bilateral,
            "right": bilateral,
            "bilateral": bilateral,
            "width": width,
            "force": force,
            "source": "gripper_force_proxy",
        }

    def render(self, camera="wrist", width=1280, height=720):
        if camera != "wrist":
            raise ValueError("real adapter only exposes the wrist D435i camera")
        data = self.client.request("camera")
        rgb_jpeg = base64.b64decode(data["rgb_jpeg_b64"])
        rgb = cv2.imdecode(
            np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError("failed to decode RGB frame from bridge")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = np.frombuffer(
            zlib.decompress(base64.b64decode(data["depth_zlib_b64"])),
            dtype=np.float32,
        ).reshape(int(data["height"]), int(data["width"])).copy()
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(
                depth, (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        self._camera_snapshot = {
            "rgb": rgb,
            "depth": depth,
            "camera": {
                "name": "wrist",
                "parent_body": "gripper_base",
                "width": int(data["width"]),
                "height": int(data["height"]),
                "fx": float(data["fx"]),
                "fy": float(data["fy"]),
                "ppx": float(data["ppx"]),
                "ppy": float(data["ppy"]),
                "scale": float(data.get("depth_scale_m", 1.0)),
                "T_world_camera": _as_transform(
                    data["T_base_camera"], "T_base_camera"),
            },
            "T_base_flange": _as_transform(
                data["T_base_flange"], "T_base_flange"),
            "T_base_tcp": _as_transform(data["T_base_tcp"], "T_base_tcp"),
            "frame_age_s": float(data.get("frame_age_s", 0.0)),
        }
        return {"img": rgb, "depth": depth}

    def camera_parameters(self, camera="wrist", width=1280, height=720):
        if self._camera_snapshot is None:
            raise RuntimeError("render() must be called before camera_parameters()")
        if camera != "wrist":
            raise ValueError("real adapter only exposes the wrist D435i camera")
        return dict(self._camera_snapshot["camera"])

    def camera_offset_from_tcp(self, camera="wrist"):
        if camera != "wrist":
            raise ValueError("real adapter only exposes the wrist D435i camera")
        return self.T_tcp_cam[:3, 3].copy()

    def camera_rotation_from_tcp(self, camera="wrist"):
        if camera != "wrist":
            raise ValueError("real adapter only exposes the wrist D435i camera")
        return self.T_tcp_cam[:3, :3].copy()

    def camera_grasp_to_world(self, translation, rotation, camera="wrist", *,
                              T_world_camera=None):
        camera_grasp = np.eye(4)
        camera_grasp[:3, :3] = np.asarray(rotation, dtype=float)
        camera_grasp[:3, 3] = np.asarray(translation, dtype=float)
        grasp_tcp = np.eye(4)
        grasp_tcp[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        transform = (
            self.camera_parameters(camera)["T_world_camera"]
            if T_world_camera is None else
            np.asarray(T_world_camera, dtype=float))
        return transform @ camera_grasp @ grasp_tcp

    def _check_pose(self, position, rotation):
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("Expected a 3D position and 3x3 rotation")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Pose must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Rotation must be orthonormal")
        if not (self.table_x[0] <= position[0] <= self.table_x[1] and
                self.table_y[0] <= position[1] <= self.table_y[1] and
                self.table_z[0] <= position[2] <= self.table_z[1]):
            raise ValueError(f"TCP target outside workspace: {position.tolist()}")
        if (self.reach_center is not None and
                not inside_reach_sphere(
                    position, self.reach_center, self.reach_radius_m,
                    self.reach_min_radius_m)):
            distance = float(np.linalg.norm(position - self.reach_center))
            raise ValueError(
                f"TCP target outside URDF reach sphere: {position.tolist()} "
                f"distance={distance:.4f} m, allowed "
                f"{self.reach_min_radius_m:.3f}..{self.reach_radius_m:.3f} m")

    def plan_to(self, position, rotation, duration=2.0, *, start_joint=None):
        self._check_pose(position, rotation)
        current = self.tcp_pose
        current_joint = self.get_joint()
        distance = float(np.linalg.norm(
            np.asarray(position, dtype=float) - current[:3, 3]))
        angle = _rotation_angle(current[:3, :3], rotation)
        if distance > self.max_command_distance_m:
            raise RuntimeError(
                f"target distance {distance:.3f} m exceeds "
                f"{self.max_command_distance_m:.3f} m")
        if angle > self.max_command_rotation_rad:
            raise RuntimeError(
                f"target rotation {angle:.3f} rad exceeds "
                f"{self.max_command_rotation_rad:.3f} rad")
        target = np.eye(4)
        target[:3, :3] = np.asarray(rotation, dtype=float)
        target[:3, 3] = np.asarray(position, dtype=float)
        return {
            "start_joint": current_joint.copy(),
            "goals": np.asarray([current_joint]),
            "position": np.asarray(position, dtype=float).copy(),
            "rotation": np.asarray(rotation, dtype=float).copy(),
            "target_tcp": target,
            "duration": float(duration),
            "linear": bool(self.config["arm"].get("linear_moves", True)),
        }

    def execute_plan(self, plan, *, position_tolerance=0.01,
                     angle_tolerance=0.03):
        target = plan.get("target_tcp")
        if target is None:
            raise ValueError("real plan must contain target_tcp")
        target = _as_transform(target, "target_tcp")
        result = self.client.request(
            "move_tcp",
            tcp_pose=target.tolist(),
            linear=bool(plan.get("linear", True)),
            timeout_s=max(30.0, 6.0 * float(plan.get("duration", 2.0))),
            position_tolerance=float(position_tolerance),
            angle_tolerance=float(angle_tolerance),
            dry_run=not self.execute,
        )
        if self.execute and "final_tcp_pose" in result:
            self._state_snapshot["T_base_tcp"] = result["final_tcp_pose"]
            self._state_time = time.monotonic()
        return result

    def move_to(self, position, rotation, duration=2.0, *,
                position_tolerance=0.01, angle_tolerance=0.03):
        plan = self.plan_to(position, rotation, duration)
        return self.execute_plan(
            plan, position_tolerance=position_tolerance,
            angle_tolerance=angle_tolerance)

    def move_to_joint(self, joints, duration=3.0):
        """Joint-space move; used to return to the captured baseline safely."""
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Expected six finite joint angles")
        result = self.client.request(
            "move_joints",
            joints=joints.tolist(),
            timeout_s=max(30.0, 6.0 * float(duration)),
            joint_tolerance_rad=0.01,
            dry_run=not self.execute,
        )
        if self.execute and "state" in result:
            self._state_snapshot = result["state"]
            self._state_time = time.monotonic()
        return result

    def gripper_control(self, width, duration=1.5):
        width = float(width)
        cfg = self.config["gripper"]
        if not 0.0 <= width <= float(cfg["max_width_m"]):
            raise ValueError("Gripper width must be between 0 and max_width_m")
        result = self.client.request(
            "gripper",
            width_m=width,
            force_n=float(cfg["default_force_n"]),
            timeout_s=max(5.0, 3.0 * float(duration)),
            dry_run=not self.execute,
        )
        if self.execute and "state" in result:
            self._state_snapshot = result["state"]
            self._state_time = time.monotonic()
        return result

    def wait(self, duration):
        time.sleep(max(0.0, float(duration)))

    def set_target(self, target):
        self.target_key = "" if target is None else str(target)
        self.target_label = self.target_key
        self.target_body = ""
        self.target_body_id = -1

    @property
    def object_position(self):
        raise RuntimeError("Real robot has no simulator object truth")

    def emergency_stop(self):
        return self.client.request("estop")

    def close(self):
        if self._opened_gate and self.config["safety"].get(
                "close_gate_on_exit", True):
            try:
                self.client.request("open_gate", enable=False)
            except Exception as error:
                self.log(f"failed to close control gate: {error}")
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
