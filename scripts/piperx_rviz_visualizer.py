#!/usr/bin/env python3
"""Publish real PiperX state, wrist camera and grasp pose for RViz2.

The node is intentionally independent of the vision process.  It republishes
the driver feedback as the joint names expected by the official URDF, adds the
hand-eye camera transform, relays the D435i image into a camera frame attached
to base_link, and renders the newest ``grasp_pose.json`` as markers.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import (
    CameraInfo, Image, JointState, PointCloud2, PointField,
)
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


# D435i official Min-Z is about 0.105 m; keep a small margin for the display
# near clip.  This only affects RViz cloud/depth rendering -- the grasp
# pipeline consumes the raw aligned depth and does not apply this cutoff.
MIN_DISPLAY_DEPTH_M = 0.11
MAX_DISPLAY_DEPTH_M = 1.0


ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def load_yaml(path):
    import yaml
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def load_handeye(path):
    with Path(path).open() as stream:
        data = json.load(stream)
    transform = np.asarray(data["result"]["matrix4x4"], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("hand-eye matrix must be 4x4")
    return transform


def pose_matrix(msg: PoseStamped):
    pose = msg.pose
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat([
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w,
    ]).as_matrix()
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def transform_msg(transform, parent, child, stamp):
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(transform[0, 3])
    msg.transform.translation.y = float(transform[1, 3])
    msg.transform.translation.z = float(transform[2, 3])
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    msg.transform.rotation.x = float(quat[0])
    msg.transform.rotation.y = float(quat[1])
    msg.transform.rotation.z = float(quat[2])
    msg.transform.rotation.w = float(quat[3])
    return msg


def marker_axes(frame, stamp, tcp):
    markers = []
    colors = [(1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.2, 1.0)]
    origin = tcp[:3, 3]
    for index, color in enumerate(colors):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns = "piperx_grasp_axes"
        marker.id = index
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.004
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (*color, 1.0)
        marker.points = [
            _point(origin),
            _point(origin + tcp[:3, index] * 0.06),
        ]
        markers.append(marker)
    return markers


def marker_gripper(frame, stamp, tcp, width):
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.ns = "piperx_grasp_gripper"
    marker.id = 10
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.003
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.9, 0.9, 1.0
    half = max(float(width or 0.08) / 2.0, 0.005)
    points = []
    for y in (-half, half):
        points.extend([
            _point(tcp[:3, 3] + tcp[:3, :3] @ np.array([0.0, y, -0.015])),
            _point(tcp[:3, 3] + tcp[:3, :3] @ np.array([0.0, y, 0.045])),
        ])
    points.extend([
        _point(tcp[:3, 3] + tcp[:3, :3] @ np.array([0.0, -half, -0.015])),
        _point(tcp[:3, 3] + tcp[:3, :3] @ np.array([0.0, half, -0.015])),
    ])
    marker.points = points
    return marker


def marker_approach(frame, stamp, tcp):
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.ns = "piperx_grasp_approach"
    marker.id = 20
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.008
    marker.scale.y = 0.014
    marker.scale.z = 0.014
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.9, 0.0, 1.0
    marker.points = [
        _point(tcp[:3, 3]),
        _point(tcp[:3, 3] + tcp[:3, 2] * 0.08),
    ]
    return marker


def marker_text(frame, stamp, tcp, rank, width):
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.ns = "piperx_grasp_text"
    marker.id = 30
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position = _point(tcp[:3, 3] + tcp[:3, :3] @ np.array([0.0, 0.0, 0.06]))
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.018
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 1.0, 1.0, 1.0
    marker.text = f"rank {rank}  width {float(width or 0) * 1000:.0f} mm"
    return marker


def _point(xyz):
    from geometry_msgs.msg import Point
    point = Point()
    point.x, point.y, point.z = [float(v) for v in xyz]
    return point


def camera_body_markers(frame, stamp, transform):
    markers = []
    body = Marker()
    body.header.frame_id = frame
    body.header.stamp = stamp
    body.ns = "piperx_camera_body"
    body.id = 0
    body.type = Marker.CUBE
    body.action = Marker.ADD
    body.pose.position = _point(transform[:3, 3])
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    body.pose.orientation.x = float(quat[0])
    body.pose.orientation.y = float(quat[1])
    body.pose.orientation.z = float(quat[2])
    body.pose.orientation.w = float(quat[3])
    body.scale.x, body.scale.y, body.scale.z = 0.09, 0.025, 0.025
    body.color.r, body.color.g, body.color.b, body.color.a = 0.55, 0.58, 0.62, 0.9
    markers.append(body)
    return markers


def camera_frustum_markers(frame, stamp, transform, info, depth=0.25):
    if info is not None and len(info.k) >= 6:
        k = list(info.k)
        fx, fy, ppx, ppy = k[0], k[4], k[2], k[5]
        width, height = int(info.width), int(info.height)
    else:
        fx, fy, ppx, ppy = 905.14, 903.11, 644.13, 377.80
        width, height = 1280, 720
    corners = []
    for u, v in ((0, 0), (width, 0), (width, height), (0, height)):
        ray = np.array([(u - ppx) / fx, (v - ppy) / fy, 1.0]) * depth
        corners.append(transform[:3, :3] @ ray + transform[:3, 3])
    origin = transform[:3, 3]
    points = []
    for corner in corners:
        points.extend([_point(origin), _point(corner)])
    for index in range(4):
        points.extend([_point(corners[index]), _point(corners[(index + 1) % 4])])
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.ns = "piperx_camera_frustum"
    marker.id = 10
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.002
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 1.0, 0.2, 0.8
    marker.points = points
    return [marker]


def color_to_bgr(msg):
    height, width = int(msg.height), int(msg.width)
    encoding = msg.encoding.lower()
    if encoding in ("rgb8", "bgr8"):
        row = np.frombuffer(msg.data, np.uint8).reshape(
            height, msg.step)[:, :width * 3].reshape(height, width, 3)
        return row.copy() if encoding == "bgr8" else cv2.cvtColor(
            row, cv2.COLOR_RGB2BGR)
    if encoding in ("rgba8", "bgra8"):
        row = np.frombuffer(msg.data, np.uint8).reshape(
            height, msg.step)[:, :width * 4].reshape(
                height, width, 4)[..., :3]
        return row.copy() if encoding == "bgra8" else cv2.cvtColor(
            row, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported color encoding {msg.encoding!r}")


def draw_grasp_overlay(image, T_cam_tcp, grasp, fx, fy, ppx, ppy):
    def project(point):
        point = np.asarray(point, dtype=float)
        camera = T_cam_tcp[:3, :3] @ point + T_cam_tcp[:3, 3]
        if camera[2] <= 0.01:
            return None
        u = fx * camera[0] / camera[2] + ppx
        v = fy * camera[1] / camera[2] + ppy
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return int(round(u)), int(round(v))

    origin = project([0.0, 0.0, 0.0])
    if origin is None:
        return image
    for axis, color in (
        ([0.06, 0.0, 0.0], (0, 0, 255)),
        ([0.0, 0.06, 0.0], (0, 255, 0)),
        ([0.0, 0.0, 0.06], (255, 0, 0)),
    ):
        end = project(axis)
        if end is not None:
            cv2.line(image, origin, end, color, 2, cv2.LINE_AA)
    width = float(grasp.get("width") or 0.08)
    half = max(width / 2.0, 0.005)
    outline = (
        ([0.0, -half, -0.015], [0.0, -half, 0.045]),
        ([0.0, half, -0.015], [0.0, half, 0.045]),
        ([0.0, -half, -0.015], [0.0, half, -0.015]),
        ([0.0, -half, 0.045], [0.0, half, 0.045]),
    )
    for start, end in outline:
        p_start, p_end = project(start), project(end)
        if p_start is not None and p_end is not None:
            cv2.line(image, p_start, p_end, (255, 255, 0), 2, cv2.LINE_AA)
    approach = project([0.0, 0.0, 0.08])
    if approach is not None:
        cv2.arrowedLine(image, origin, approach, (0, 255, 255), 2,
                        tipLength=0.2, line_type=cv2.LINE_AA)
    text = f"rank {grasp.get('rank')}  w={width * 1000:.0f} mm"
    cv2.putText(image, text, (max(origin[0] + 6, 6), max(origin[1] - 6, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    return image


class PiperXRvizNode(Node):
    def __init__(self, config_path, grasp_glob):
        super().__init__("piperx_rviz_visualizer")
        self.config = load_yaml(config_path)
        handeye = Path(self.config["camera"]["handeye_json"])
        if not handeye.is_absolute():
            handeye = (ROOT / handeye).resolve()
        self.T_flange_cam = load_handeye(handeye)
        self.camera_frame = "piperx_camera_optical"
        self.grasp_glob = str(grasp_glob)
        self.last_grasp = None
        self.last_grasp_mtime = None
        self.latest_tcp = None
        self.latest_info = None
        self.latest_color = None

        self.joint_pub = self.create_publisher(JointState, "/piperx/joint_states", 10)
        self.image_pub = self.create_publisher(
            Image, "/piperx/camera/color/image_raw", qos_profile_sensor_data)
        self.info_pub = self.create_publisher(
            CameraInfo, "/piperx/camera/color/camera_info", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/piperx/grasp_markers", 10)
        self.camera_marker_pub = self.create_publisher(
            MarkerArray, "/piperx/camera_markers", 10)
        self.depth_cloud_pub = self.create_publisher(
            PointCloud2, "/piperx/camera/depth_cloud", qos_profile_sensor_data)
        self.rgbd_cloud_pub = self.create_publisher(
            PointCloud2, "/piperx/camera/rgbd_cloud", qos_profile_sensor_data)
        self.depth_image_pub = self.create_publisher(
            Image, "/piperx/camera/depth/image_raw", qos_profile_sensor_data)
        self.rgbd_pub = self.create_publisher(
            Image, "/piperx/camera/rgbd_image", qos_profile_sensor_data)
        self.last_depth_publish = 0.0
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            JointState, "/feedback/joint_states", self.on_joints, 10)
        self.create_subscription(
            PoseStamped, "/feedback/tcp_pose", self.on_tcp, 10)
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.on_image,
            qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self.on_info, 10)
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self.on_depth, qos_profile_sensor_data)
        self.create_timer(1.0, self.on_timer)
        self.get_logger().info(
            f"RViz2 visualizer ready; camera frame={self.camera_frame}; "
            f"grasp glob={self.grasp_glob}")

    def on_joints(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions for name in ARM_JOINTS):
            return
        width = float(positions.get("gripper", 0.0))
        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = list(ARM_JOINTS) + ["gripper", "gripper_joint1", "gripper_joint2"]
        out.position = (
            [float(positions[name]) for name in ARM_JOINTS]
            + [width, width / 2.0, -width / 2.0]
        )
        out.velocity = []
        out.effort = []
        self.joint_pub.publish(out)

    def on_tcp(self, msg):
        self.latest_tcp = msg

    def on_image(self, msg):
        self.latest_color = msg
        msg.header.frame_id = self.camera_frame
        self.image_pub.publish(msg)

    def on_info(self, msg):
        self.latest_info = msg
        msg.header.frame_id = self.camera_frame
        self.info_pub.publish(msg)

    def on_depth(self, msg):
        now = time.monotonic()
        if now - self.last_depth_publish < 0.2:
            return
        if self.latest_info is None or len(self.latest_info.k) < 6:
            return
        height, width = int(msg.height), int(msg.width)
        encoding = msg.encoding.lower()
        if encoding in ("16uc1", "mono16"):
            raw = np.frombuffer(msg.data, np.uint16).reshape(
                height, msg.step // 2)[:, :width]
            depth = raw.astype(np.float32) * 0.001
        elif encoding == "32fc1":
            depth = np.frombuffer(msg.data, np.float32).reshape(
                height, msg.step // 4)[:, :width].copy()
        else:
            return
        self.last_depth_publish = now
        fx = float(self.latest_info.k[0])
        fy = float(self.latest_info.k[4])
        ppx = float(self.latest_info.k[2])
        ppy = float(self.latest_info.k[5])
        workspace = self.config.get("workspace", {})
        x_min, x_max = workspace.get("x", [0.0, 0.80])
        y_min, y_max = workspace.get("y", [-0.80, 0.80])
        z_min = float(workspace.get("table_top_z", 0.0)) - 0.02
        z_max = float(workspace.get("object_z", [0.002, 0.25])[1])
        reach_radius = workspace.get("reach_radius_m")
        T_base_camera = None
        if self.latest_tcp is not None:
            T_base_camera = pose_matrix(self.latest_tcp) @ self.T_flange_cam
        try:
            median = cv2.medianBlur(depth, 5)
            depth_consistent = np.abs(depth - median) < 0.03
        except cv2.error:
            depth_consistent = np.ones_like(depth, dtype=bool)
        u_full, v_full = np.meshgrid(
            np.arange(width), np.arange(height))
        valid_full = (np.isfinite(depth) &
                      (depth > MIN_DISPLAY_DEPTH_M) &
                      (depth < MAX_DISPLAY_DEPTH_M) &
                      depth_consistent)
        if T_base_camera is not None:
            x_full = (u_full - ppx) * depth / fx
            y_full = (v_full - ppy) * depth / fy
            base = (np.stack([x_full, y_full, depth], axis=-1)
                    @ T_base_camera[:3, :3].T + T_base_camera[:3, 3])
            valid_full &= (
                (base[..., 0] >= x_min) & (base[..., 0] <= x_max) &
                (base[..., 1] >= y_min) & (base[..., 1] <= y_max) &
                (base[..., 2] >= z_min) & (base[..., 2] <= z_max))
            if reach_radius is not None:
                center_xy = workspace.get("reach_center_xy", (0.0, 0.0))
                center = np.array([
                    float(center_xy[0]), float(center_xy[1]),
                    float(workspace.get("reach_center_z", 0.123)),
                ])
                distance_sq = np.sum((base - center) ** 2, axis=-1)
                radius = float(reach_radius)
                valid_full &= distance_sq <= radius * radius
                min_radius = float(workspace.get("reach_min_radius_m", 0.0))
                if min_radius > 0.0:
                    valid_full &= distance_sq >= min_radius * min_radius
        vis = np.clip((depth - 0.05) / (1.5 - 0.05), 0.0, 1.0)
        vis = (vis * 255.0).astype(np.uint8)
        vis[~valid_full] = 0
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        colored[~valid_full] = 0
        image = Image()
        image.header.stamp = msg.header.stamp
        image.header.frame_id = self.camera_frame
        image.height = height
        image.width = width
        image.encoding = "bgr8"
        image.is_bigendian = 0
        image.step = width * 3
        image.data = colored.tobytes()
        self.depth_image_pub.publish(image)

        color = None
        if self.latest_color is not None:
            try:
                color = color_to_bgr(self.latest_color)
            except Exception:
                color = None
        if color is not None:
            if color.shape[:2] != colored.shape[:2]:
                color = cv2.resize(
                    color, (colored.shape[1], colored.shape[0]))
            if (self.last_grasp is not None and
                    self.latest_tcp is not None and
                    self.last_grasp_mtime is not None and
                    time.time() - self.last_grasp_mtime < 60.0 and
                    self.last_grasp.get("world_tcp")):
                T_base_flange = pose_matrix(self.latest_tcp)
                T_base_camera = T_base_flange @ self.T_flange_cam
                T_base_tcp = np.asarray(
                    self.last_grasp["world_tcp"], dtype=float)
                T_cam_tcp = np.linalg.inv(T_base_camera) @ T_base_tcp
                draw_grasp_overlay(
                    color, T_cam_tcp, self.last_grasp,
                    fx, fy, ppx, ppy)
                draw_grasp_overlay(
                    colored, T_cam_tcp, self.last_grasp,
                    fx, fy, ppx, ppy)
            combined = np.hstack([color, colored])
            cv2.putText(combined, "RGB", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)
            cv2.putText(combined, "DEPTH",
                        (color.shape[1] + 12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)
            rgbd = Image()
            rgbd.header.stamp = msg.header.stamp
            rgbd.header.frame_id = self.camera_frame
            rgbd.height = combined.shape[0]
            rgbd.width = combined.shape[1]
            rgbd.encoding = "bgr8"
            rgbd.is_bigendian = 0
            rgbd.step = combined.shape[1] * 3
            rgbd.data = combined.tobytes()
            self.rgbd_pub.publish(rgbd)

        step = 2
        z = depth[::step, ::step]
        u, v = np.meshgrid(
            np.arange(0, width, step), np.arange(0, height, step))
        valid = valid_full[::step, ::step]
        x = (u - ppx) * z / fx
        y = (v - ppy) * z / fy
        points = np.stack([x[valid], y[valid], z[valid], z[valid]], axis=1)
        if len(points) == 0:
            return
        header = msg.header
        header.frame_id = self.camera_frame
        fields = [
            PointField(name="x", offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12,
                       datatype=PointField.FLOAT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(header, fields, points.tolist())
        self.depth_cloud_pub.publish(cloud)

        if color is not None:
            if color.shape[:2] != (height, width):
                color = cv2.resize(color, (width, height))
            blue = color[v[valid], u[valid], 0].astype(np.uint32)
            green = color[v[valid], u[valid], 1].astype(np.uint32)
            red = color[v[valid], u[valid], 2].astype(np.uint32)
            packed = (red << 16) | (green << 8) | blue
            rgb_float = packed.astype(np.uint32).view(np.float32)
            points_rgb = np.stack(
                [x[valid], y[valid], z[valid], rgb_float], axis=1)
            fields_rgb = [
                PointField(name="x", offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12,
                           datatype=PointField.FLOAT32, count=1),
            ]
            cloud_rgb = point_cloud2.create_cloud(
                header, fields_rgb, points_rgb.tolist())
            self.rgbd_cloud_pub.publish(cloud_rgb)

    def on_timer(self):
        stamp = self.get_clock().now().to_msg()
        if self.latest_tcp is not None:
            T_base_flange = pose_matrix(self.latest_tcp)
            T_base_camera = T_base_flange @ self.T_flange_cam
            self.tf_broadcaster.sendTransform(transform_msg(
                T_base_camera, "base_link", self.camera_frame, stamp))
            self.publish_camera_markers(stamp, T_base_camera)
        self.publish_markers(stamp)

    def publish_camera_markers(self, stamp, T_base_camera):
        markers = camera_body_markers("base_link", stamp, T_base_camera)
        markers += camera_frustum_markers(
            "base_link", stamp, T_base_camera, self.latest_info)
        self.camera_marker_pub.publish(MarkerArray(markers=markers))

    def _newest_grasp(self):
        paths = glob.glob(self.grasp_glob)
        if not paths:
            return None
        path = max(paths, key=lambda item: Path(item).stat().st_mtime)
        mtime = Path(path).stat().st_mtime
        if mtime != self.last_grasp_mtime:
            try:
                with Path(path).open() as stream:
                    self.last_grasp = json.load(stream)
                self.last_grasp_mtime = mtime
                self.get_logger().info(f"Loaded grasp pose: {path}")
            except Exception as error:
                self.get_logger().warn(f"failed to read {path}: {error}")
        return self.last_grasp

    def publish_markers(self, stamp):
        data = self._newest_grasp()
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        if data and data.get("world_tcp"):
            tcp = np.asarray(data["world_tcp"], dtype=float)
            if tcp.shape == (4, 4):
                width = data.get("width")
                rank = data.get("rank")
                array.markers.extend(marker_axes("base_link", stamp, tcp))
                array.markers.append(marker_gripper("base_link", stamp, tcp, width))
                array.markers.append(marker_approach("base_link", stamp, tcp))
        self.marker_pub.publish(array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/real_piperx.yaml"))
    parser.add_argument(
        "--grasp-glob",
        default=str(ROOT / "docs/vision_runs/*/cycle_*/final/grasp_pose.json"))
    args = parser.parse_args()
    rclpy.init()
    node = PiperXRvizNode(args.config, args.grasp_glob)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
