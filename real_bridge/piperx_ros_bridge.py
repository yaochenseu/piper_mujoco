#!/usr/bin/env python3
"""Local JSON bridge between the vision process and the ROS 2 Piper driver."""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import socket
import threading
import time
import zlib

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Empty, SetBool

try:
    from agx_arm_msgs.msg import AgxArmStatus, GripperStatus
except ImportError:
    AgxArmStatus = None
    GripperStatus = None


ROOT = Path(__file__).resolve().parents[1]

# Piper firmware /feedback/arm_status codes (AgxArmStatus.arm_status).
ARM_STATUS_NAMES = {
    0: "normal",
    1: "emergency_stop",
    2: "no_solution",
    3: "singularity_point",
    4: "target_pos_exceeds_limit",
    5: "joint_communication_error",
    6: "joint_brake_not_released",
    7: "collision_occurred",
    8: "overspeed_during_teaching_drag",
    9: "joint_status_error",
    10: "other_error",
    11: "teaching_record",
    12: "teaching_execution",
    13: "teaching_pause",
    14: "main_controller_over_temperature",
    15: "release_resistor_over_temperature",
}


def arm_status_name(code):
    return ARM_STATUS_NAMES.get(int(code), f"unknown({code})")


def load_yaml(path):
    import yaml
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def pose_to_matrix(msg):
    pose = msg.pose
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat([
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w,
    ]).as_matrix()
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def matrix_to_pose_msg(transform, frame_id):
    transform = np.asarray(transform, dtype=float)
    rotation = Rotation.from_matrix(transform[:3, :3])
    quaternion = rotation.as_quat()
    rpy = rotation.as_euler("xyz", degrees=True)
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(transform[0, 3])
    msg.pose.position.y = float(transform[1, 3])
    msg.pose.position.z = float(transform[2, 3])
    msg.pose.orientation.x = float(quaternion[0])
    msg.pose.orientation.y = float(quaternion[1])
    msg.pose.orientation.z = float(quaternion[2])
    msg.pose.orientation.w = float(quaternion[3])
    return msg, rpy


def rotation_error_rad(first, second):
    cosine = 0.5 * (np.trace(np.asarray(first).T @ np.asarray(second)) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def image_to_rgb(msg):
    height, width = int(msg.height), int(msg.width)
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, msg.step)
    encoding = msg.encoding.lower()
    if encoding in ("rgb8", "bgr8"):
        image = row[:, :width * 3].reshape(height, width, 3).copy()
        if encoding == "bgr8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    if encoding in ("rgba8", "bgra8"):
        image = row[:, :width * 4].reshape(height, width, 4)[..., :3].copy()
        if encoding == "bgra8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    if encoding == "mono8":
        image = row[:, :width].reshape(height, width)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"unsupported color encoding {msg.encoding!r}")


def image_to_depth(msg, scale_m):
    height, width = int(msg.height), int(msg.width)
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, msg.step)
    encoding = msg.encoding.lower()
    if encoding in ("16uc1", "mono16"):
        depth = row[:, :width * 2].copy().view(np.uint16).reshape(height, width)
        return depth.astype(np.float32) * float(scale_m)
    if encoding == "32fc1":
        return row[:, :width * 4].copy().view(np.float32).reshape(height, width)
    raise ValueError(f"unsupported depth encoding {msg.encoding!r}")


class PiperRosBridge(Node):
    def __init__(self, config):
        super().__init__("piperx_ros_bridge")
        self.config = config
        self.lock = threading.Lock()
        self.latest_joint_state = None
        self.latest_pose = None
        self.latest_arm_status = None
        self.latest_gripper_status = None
        self.latest_color = None
        self.latest_depth = None
        self.latest_color_info = None
        self.latest_depth_info = None
        self.color_stamp = 0.0
        self.pose_stamp = 0.0
        self.state_stamp = 0.0

        self.flange_tcp = self._lookup("arm", "flange_tcp_xyz_m")
        tcp_xyz = np.asarray(self.flange_tcp, dtype=float)
        tcp_rpy = np.asarray(self._lookup("arm", "flange_tcp_rpy_deg"), dtype=float)
        self.T_flange_tcp = np.eye(4)
        self.T_flange_tcp[:3, :3] = Rotation.from_euler(
            "xyz", tcp_rpy, degrees=True).as_matrix()
        self.T_flange_tcp[:3, 3] = tcp_xyz
        self.T_flange_cam = self._load_handeye(
            self._lookup("camera", "handeye_json"))

        self.create_subscription(
            JointState, "/feedback/joint_states", self._on_joints, 10)
        self.create_subscription(
            PoseStamped, "/feedback/tcp_pose", self._on_pose, 10)
        if AgxArmStatus is not None:
            self.create_subscription(
                AgxArmStatus, "/feedback/arm_status", self._on_arm_status, 10)
        if GripperStatus is not None:
            self.create_subscription(
                GripperStatus, "/feedback/gripper_status",
                self._on_gripper_status, 10)

        camera = self._lookup("camera")
        self.create_subscription(
            Image, camera["color_topic"], self._on_color, 10)
        self.create_subscription(
            Image, camera["depth_topic"], self._on_depth, 10)
        self.create_subscription(
            CameraInfo, camera["color_info_topic"], self._on_color_info, 10)
        self.create_subscription(
            CameraInfo, camera["depth_info_topic"], self._on_depth_info, 10)

        self.pub_j = self.create_publisher(JointState, "/control/move_j", 1)
        self.pub_p = self.create_publisher(PoseStamped, "/control/move_p", 1)
        self.pub_l = self.create_publisher(PoseStamped, "/control/move_l", 1)
        self.gate_client = self.create_client(SetBool, "/control_enable")
        self.estop_client = self.create_client(Empty, "/emergency_stop")
        self.get_logger().info("PiperX ROS bridge ready")

    def _lookup(self, *keys):
        value = self.config
        for key in keys:
            value = value[key]
        return value

    def _load_handeye(self, configured):
        path = Path(configured)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        with path.open() as stream:
            data = json.load(stream)
        return np.asarray(data["result"]["matrix4x4"], dtype=float)

    def _on_joints(self, msg):
        with self.lock:
            self.latest_joint_state = msg
            self.state_stamp = time.monotonic()

    def _on_pose(self, msg):
        with self.lock:
            self.latest_pose = msg
            self.pose_stamp = time.monotonic()
            self.state_stamp = self.pose_stamp

    def _on_arm_status(self, msg):
        with self.lock:
            self.latest_arm_status = msg
            self.state_stamp = time.monotonic()

    def _on_gripper_status(self, msg):
        with self.lock:
            self.latest_gripper_status = msg

    def _on_color(self, msg):
        with self.lock:
            self.latest_color = image_to_rgb(msg)
            self.color_stamp = time.monotonic()

    def _on_depth(self, msg):
        with self.lock:
            self.latest_depth = image_to_depth(
                msg, self._lookup("camera", "depth_scale_m"))

    def _on_color_info(self, msg):
        with self.lock:
            self.latest_color_info = msg

    def _on_depth_info(self, msg):
        with self.lock:
            self.latest_depth_info = msg

    def _state_snapshot(self, require_pose=True):
        with self.lock:
            now = time.monotonic()
            timeout = float(self._lookup("bridge", "state_timeout_s"))
            if require_pose and self.latest_pose is None:
                raise RuntimeError("no /feedback/tcp_pose received")
            if require_pose and now - self.pose_stamp > timeout:
                raise RuntimeError("stale /feedback/tcp_pose")
            joints = {}
            efforts = {}
            if self.latest_joint_state is not None:
                joints = dict(zip(
                    self.latest_joint_state.name,
                    self.latest_joint_state.position,
                ))
                efforts = dict(zip(
                    self.latest_joint_state.name,
                    self.latest_joint_state.effort,
                ))
            T_flange = pose_to_matrix(self.latest_pose) if self.latest_pose else None
            T_tcp = T_flange @ self.T_flange_tcp if T_flange is not None else None
            arm_status = None
            if self.latest_arm_status is not None:
                arm_status = {
                    "arm_status": int(self.latest_arm_status.arm_status),
                    "motion_status": int(self.latest_arm_status.motion_status),
                    "mode_feedback": int(self.latest_arm_status.mode_feedback),
                    "ctrl_mode": int(self.latest_arm_status.ctrl_mode),
                }
            gripper_width = None
            gripper_force = None
            if "gripper" in joints:
                gripper_width = float(abs(joints["gripper"]))
                gripper_force = float(abs(efforts.get("gripper", 0.0)))
            if self.latest_gripper_status is not None:
                gripper_width = float(self.latest_gripper_status.width)
                gripper_force = float(self.latest_gripper_status.force)
            return {
                "T_base_flange": None if T_flange is None else T_flange.tolist(),
                "T_base_tcp": None if T_tcp is None else T_tcp.tolist(),
                "joints": {name: float(value) for name, value in joints.items()},
                "arm_rad": {
                    f"joint{i}": float(joints[f"joint{i}"])
                    for i in range(1, 7) if f"joint{i}" in joints
                },
                "gripper_width_m": gripper_width,
                "gripper_force_n": gripper_force,
                "arm_status": arm_status,
                "pose_age_s": float(now - self.pose_stamp) if self.pose_stamp else None,
            }

    def _camera_snapshot(self):
        with self.lock:
            now = time.monotonic()
            max_age = float(self._lookup("camera", "max_frame_age_s"))
            if self.latest_color is None or self.latest_depth is None:
                raise RuntimeError("no aligned RGB-D frame received")
            if now - self.color_stamp > max_age:
                raise RuntimeError("stale RGB-D frame")
            if self.latest_color_info is None:
                raise RuntimeError("no color CameraInfo received")
            if self.latest_pose is None:
                raise RuntimeError("no /feedback/tcp_pose for camera pose")
            rgb = self.latest_color.copy()
            depth = self.latest_depth.copy()
            info = self.latest_color_info
            T_flange = pose_to_matrix(self.latest_pose)
            T_tcp = T_flange @ self.T_flange_tcp
            T_camera = T_flange @ self.T_flange_cam
            if depth.shape != rgb.shape[:2]:
                depth = cv2.resize(
                    depth, (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
            k = np.asarray(info.k, dtype=float).reshape(3, 3)
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise RuntimeError("failed to encode RGB frame")
            return {
                "rgb_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "depth_zlib_b64": base64.b64encode(
                    zlib.compress(depth.astype(np.float32).tobytes(), 3)
                ).decode("ascii"),
                "depth_dtype": "float32",
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "ppx": float(k[0, 2]),
                "ppy": float(k[1, 2]),
                "depth_scale_m": 1.0,
                "T_flange_cam": self.T_flange_cam.tolist(),
                "T_flange_tcp": self.T_flange_tcp.tolist(),
                "T_base_flange": T_flange.tolist(),
                "T_base_tcp": T_tcp.tolist(),
                "T_base_camera": T_camera.tolist(),
                "frame_age_s": float(now - self.color_stamp),
            }

    def _call_service(self, client, request, timeout):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {client.srv_name} is not available")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise RuntimeError(f"service {client.srv_name} timed out")
        if future.exception() is not None:
            raise RuntimeError(str(future.exception()))
        return future.result()

    def _open_gate(self, enable):
        request = SetBool.Request()
        request.data = bool(enable)
        response = self._call_service(self.gate_client, request, 3.0)
        return {"success": bool(response.success), "message": response.message}

    def _estop(self):
        response = self._call_service(self.estop_client, Empty.Request(), 3.0)
        return {"success": True, "response": str(response)}

    def _workspace_ok(self, position):
        workspace = self._lookup("workspace")
        x_min, x_max = map(float, workspace["x"])
        y_min, y_max = map(float, workspace["y"])
        z_min, z_max = map(float, workspace["table_z"])
        if not (x_min <= position[0] <= x_max and
                y_min <= position[1] <= y_max and
                z_min <= position[2] <= z_max):
            return False
        reach_radius = workspace.get("reach_radius_m")
        if reach_radius is None:
            return True
        center_xy = workspace.get("reach_center_xy", (0.0, 0.0))
        center = np.array([
            float(center_xy[0]), float(center_xy[1]),
            float(workspace.get("reach_center_z", 0.123)),
        ], dtype=float)
        distance = float(np.linalg.norm(
            np.asarray(position, dtype=float) - center))
        min_radius = float(workspace.get("reach_min_radius_m", 0.0))
        return min_radius <= distance <= float(reach_radius)

    def _publish_pose(self, T_flange_target, linear):
        msg, _rpy = matrix_to_pose_msg(T_flange_target, "base_link")
        msg.header.stamp = self.get_clock().now().to_msg()
        publisher = self.pub_l if linear else self.pub_p
        for _ in range(5):
            publisher.publish(msg)
            time.sleep(0.02)

    def move_tcp(self, request):
        target = np.asarray(request["tcp_pose"], dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("tcp_pose must be a finite 4x4 matrix")
        if not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("tcp_pose rotation is not orthonormal")
        if not self._workspace_ok(target[:3, 3]):
            raise ValueError(f"target outside workspace: {target[:3, 3].tolist()}")

        current = self._state_snapshot()
        current_tcp = np.asarray(current["T_base_tcp"], dtype=float)
        distance = float(np.linalg.norm(target[:3, 3] - current_tcp[:3, 3]))
        angle = rotation_error_rad(current_tcp[:3, :3], target[:3, :3])
        if distance > float(self._lookup("arm", "max_command_distance_m")):
            raise ValueError(f"target distance {distance:.3f} m exceeds limit")
        max_angle = math.radians(float(self._lookup("arm", "max_command_rotation_deg")))
        if angle > max_angle:
            raise ValueError(f"target rotation {angle:.3f} rad exceeds limit")
        status = current.get("arm_status")
        if status is not None:
            code = int(status["arm_status"])
            # Codes 2/3/4 describe a rejected target, not a latched fault:
            # a new valid target must still be allowed, otherwise the arm
            # cannot recover without restarting the driver.
            if code not in (0, 2, 3, 4):
                raise RuntimeError(
                    f"arm_status={code} ({arm_status_name(code)}) is not ready")

        T_flange_target = target @ np.linalg.inv(self.T_flange_tcp)
        if request.get("dry_run", False):
            return {
                "dry_run": True,
                "target_tcp_pose": target.tolist(),
                "target_flange_pose": T_flange_target.tolist(),
                "distance_m": distance,
                "rotation_rad": angle,
            }

        self._publish_pose(T_flange_target, bool(
            request.get("linear", self._lookup("arm", "linear_moves"))))
        timeout = float(request.get("timeout_s", 30.0))
        pos_tol = float(request.get("position_tolerance", 0.01))
        angle_tol = float(request.get("angle_tolerance", 0.03))
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        initial_position = current_tcp[:3, 3].copy()
        last = current
        pos_err = float("inf")
        angle_err = float("inf")
        while time.monotonic() < deadline:
            time.sleep(0.05)
            last = self._state_snapshot()
            actual = np.asarray(last["T_base_tcp"], dtype=float)
            pos_err = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            angle_err = rotation_error_rad(actual[:3, :3], target[:3, :3])
            status = last.get("arm_status")
            if status is not None:
                code = int(status["arm_status"])
                if code not in (0, 2, 3, 4):
                    raise RuntimeError(
                        f"move aborted: arm_status={code} "
                        f"({arm_status_name(code)})")
                # Give the firmware a few cycles to replace the previous
                # command's status with this command's result.  A stale code
                # 2/3/4 must not abort a move that has already started; only
                # abort when the TCP has not moved at all.
                moved = float(np.linalg.norm(
                    actual[:3, 3] - initial_position))
                if (code in (2, 3, 4) and
                        time.monotonic() - started > 1.5 and moved < 0.001):
                    raise RuntimeError(
                        f"move rejected: arm_status={code} "
                        f"({arm_status_name(code)}); target pose is not "
                        "reachable from the current joint state. Check that "
                        "the current joints, especially joint4, are inside "
                        "the +/-89 deg software limits")
            if (pos_err <= pos_tol and angle_err <= angle_tol and
                    (status is None or status["motion_status"] == 0)):
                return {
                    "dry_run": False,
                    "target_tcp_pose": target.tolist(),
                    "final_tcp_pose": actual.tolist(),
                    "position_error_m": pos_err,
                    "rotation_error_rad": angle_err,
                    "state": last,
                }
        raise RuntimeError(
            f"move timed out: position_error={pos_err:.4f} m, "
            f"rotation_error={angle_err:.4f} rad")

    def move_joints(self, request):
        joints = np.asarray(request.get("joints"), dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("joints must contain six finite radians")
        limits = self._lookup("arm", "joint_limits_deg")
        if limits:
            if len(limits) != 6:
                raise ValueError("arm.joint_limits_deg must contain six pairs")
            for index, (value, bounds) in enumerate(zip(joints, limits), start=1):
                low, high = np.radians(bounds)
                if value < low - 1e-6 or value > high + 1e-6:
                    raise ValueError(
                        f"joint{index}={np.degrees(value):.2f} deg outside "
                        f"[{np.degrees(low):.1f}, {np.degrees(high):.1f}]")

        current = self._state_snapshot()
        current_q = np.asarray([
            current["arm_rad"][f"joint{index}"] for index in range(1, 7)])
        if request.get("dry_run", False):
            return {
                "dry_run": True,
                "target_joints": joints.tolist(),
                "current_joints": current_q.tolist(),
                "max_delta_rad": float(np.max(np.abs(joints - current_q))),
            }

        message = JointState()
        message.name = [f"joint{index}" for index in range(1, 7)]
        message.position = joints.tolist()
        for _ in range(5):
            self.pub_j.publish(message)
            time.sleep(0.02)

        started = time.monotonic()
        deadline = started + float(request.get("timeout_s", 30.0))
        tolerance = float(request.get("joint_tolerance_rad", 0.01))
        last = current
        error = float("inf")
        while time.monotonic() < deadline:
            time.sleep(0.05)
            last = self._state_snapshot()
            actual = np.asarray([
                last["arm_rad"][f"joint{index}"] for index in range(1, 7)])
            error = float(np.max(np.abs(actual - joints)))
            status = last.get("arm_status")
            if status is not None:
                code = int(status["arm_status"])
                if code not in (0, 2, 3, 4):
                    raise RuntimeError(
                        f"joint move aborted: arm_status={code} "
                        f"({arm_status_name(code)})")
                moved = float(np.max(np.abs(actual - current_q)))
                if (code in (2, 3, 4) and
                        time.monotonic() - started > 1.5 and moved < 0.005):
                    raise RuntimeError(
                        f"joint move rejected: arm_status={code} "
                        f"({arm_status_name(code)})")
            if error <= tolerance and (
                    status is None or status["motion_status"] == 0):
                return {
                    "dry_run": False,
                    "target_joints": joints.tolist(),
                    "final_joints": actual.tolist(),
                    "max_joint_error_rad": error,
                    "state": last,
                }
        raise RuntimeError(
            f"joint move timed out: max_error={error:.4f} rad")

    def move_gripper(self, request):
        width = float(request["width_m"])
        force = float(request["force_n"])
        max_width = float(self._lookup("gripper", "max_width_m"))
        force_min = float(self._lookup("gripper", "force_min_n"))
        force_max = float(self._lookup("gripper", "force_max_n"))
        if not 0.0 <= width <= max_width:
            raise ValueError(f"gripper width must be 0..{max_width} m")
        if not force_min <= force <= force_max:
            raise ValueError(f"gripper force must be {force_min}..{force_max} N")
        if request.get("dry_run", False):
            return {"dry_run": True, "width_m": width, "force_n": force}
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["gripper"]
        msg.position = [width]
        msg.effort = [force]
        for _ in range(5):
            self.pub_j.publish(msg)
            time.sleep(0.03)
        deadline = time.monotonic() + float(request.get("timeout_s", 5.0))
        last = None
        while time.monotonic() < deadline:
            time.sleep(0.05)
            last = self._state_snapshot()
            actual = last.get("gripper_width_m")
            if actual is not None and abs(actual - width) <= 0.002:
                break
        if last is None:
            last = self._state_snapshot()
        return {
            "dry_run": False,
            "target_width_m": width,
            "force_n": force,
            "state": last,
        }

    def process_request(self, request):
        command = request.get("command")
        if command == "ping":
            return {"ok": True, "data": {"name": self.get_name()}}
        if command == "shutdown":
            return {"ok": True, "data": {"shutdown": True}}
        if command == "state":
            return {"ok": True, "data": self._state_snapshot()}
        if command == "camera":
            return {"ok": True, "data": self._camera_snapshot()}
        if command == "move_tcp":
            return {"ok": True, "data": self.move_tcp(request)}
        if command == "move_joints":
            return {"ok": True, "data": self.move_joints(request)}
        if command == "gripper":
            return {"ok": True, "data": self.move_gripper(request)}
        if command == "open_gate":
            return {"ok": True, "data": self._open_gate(
                request.get("enable", True))}
        if command == "estop":
            return {"ok": True, "data": self._estop()}
        raise ValueError(f"unknown command {command!r}")


def serve(bridge, socket_path):
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    server.settimeout(1.0)
    bridge.get_logger().info(f"JSON bridge listening on {path}")
    stop = False
    try:
        while not stop:
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                try:
                    stream = connection.makefile("rwb")
                    while True:
                        line = stream.readline()
                        if not line:
                            break
                        request = {}
                        try:
                            request = json.loads(line)
                            response = bridge.process_request(request)
                        except Exception as error:
                            response = {"ok": False, "error": str(error)}
                        try:
                            stream.write(
                                (json.dumps(response, separators=(",", ":")) +
                                 "\n").encode())
                            stream.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            break
                        if request.get("command") == "shutdown":
                            stop = True
                            break
                except (BrokenPipeError, ConnectionResetError, OSError) as error:
                    # A client that times out and disconnects must not kill
                    # the bridge; the next request reconnects to this server.
                    bridge.get_logger().warning(
                        f"bridge client disconnected: {error}")
    finally:
        server.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/real_piperx.yaml"))
    args = parser.parse_args()
    config = load_yaml(args.config)
    rclpy.init()
    bridge = PiperRosBridge(config)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        serve(bridge, config["bridge"]["socket_path"])
    finally:
        executor.shutdown()
        bridge.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
