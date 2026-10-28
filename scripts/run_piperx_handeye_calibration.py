#!/usr/bin/env python3
"""Automated eye-in-hand calibration for the real Piper X.

Run with ROS 2 Humble sourced and the arm driver, RGB camera and control gate
already running.  The script starts aruco_detect, moves the arm through small
joint-space offsets around the current board-visible pose, drives the existing
calib.py collector, and saves the result in piper_control/tools/calib_results.
It never disables the arm; on any failure it returns to the initial joints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pexpect
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool

try:
    from agx_arm_msgs.msg import AgxArmStatus
except ImportError:
    AgxArmStatus = None

ROOT = Path(__file__).resolve().parents[1]
PIPER = ROOT.parent / "piper_control"
TOOLS = PIPER / "tools"
DETECTOR = TOOLS / "aruco_detect.sh"
CALIB = TOOLS / "calib.py"
RESULT_DIR = TOOLS / "calib_results"
LOG_DIR = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp"

JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
LIMITS_DEG = np.array([
    [-150.0, 150.0], [0.0, 180.0], [-170.0, 0.0],
    [-89.0, 89.0], [-89.0, 89.0], [-180.0, 180.0],
])
MIN_SAMPLES = 20
MAX_SAMPLES = 22

# Small offsets (degrees) around the current board-visible pose.  Positive J1
# yaw, shoulder/elbow translations, wrist roll/pitch and J6 roll give both
# position and orientation diversity without leaving the ~87x58 deg FOV.
OFFSETS_DEG = [
    [0, 0, 0, 0, 0, 0],
    [12, 0, 0, 0, 0, 0], [-12, 0, 0, 0, 0, 0],
    [18, 0, 0, 0, 0, 0], [-18, 0, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0], [0, -8, 0, 0, 0, 0],
    [0, 0, 8, 0, 0, 0], [0, 0, -8, 0, 0, 0],
    [12, 8, 0, 0, 0, 0], [-12, -8, 0, 0, 0, 0],
    [12, -8, 0, 0, 0, 0], [-12, 8, 0, 0, 0, 0],
    [0, 8, -8, 0, 0, 0], [0, -8, 8, 0, 0, 0],
    [0, 0, 0, 8, 0, 0], [0, 0, 0, -8, 0, 0],
    [0, 0, 0, -14, 0, 0],
    [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, -8, 0],
    [0, 0, 0, 0, 0, 15], [0, 0, 0, 0, 0, -15],
    [10, 8, -8, -8, 6, 12], [-10, -8, 8, 8, -6, -12],
    [16, -6, 6, 0, 0, -10], [-16, 6, -6, 0, 0, 10],
    [0, 0, 0, 8, 10, -12], [0, 0, 0, -8, -10, 12],
]


def log(message):
    print(message, flush=True)


class ArmClient(Node):
    def __init__(self):
        super().__init__("piperx_handeye_arm")
        self.q = None
        self.tcp = None
        self.status = None
        self.aruco = None
        self.aruco_time = 0.0
        self.create_subscription(JointState, "/feedback/joint_states", self._on_joints, 10)
        self.create_subscription(PoseStamped, "/feedback/tcp_pose", self._on_tcp, 10)
        self.create_subscription(PoseStamped, "/aruco_single/pose", self._on_aruco, 10)
        if AgxArmStatus is not None:
            self.create_subscription(AgxArmStatus, "/feedback/arm_status", self._on_status, 10)
        self.publisher = self.create_publisher(JointState, "/control/move_j", 1)
        self.gate = self.create_client(SetBool, "/control_enable")

    def _on_joints(self, msg):
        self.q = np.asarray(msg.position[:6], dtype=float)

    def _on_tcp(self, msg):
        self.tcp = msg

    def _on_aruco(self, msg):
        self.aruco = msg
        self.aruco_time = time.monotonic()

    def _on_status(self, msg):
        self.status = msg

    def spin_for(self, duration):
        end = time.monotonic() + max(0.0, duration)
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, end - time.monotonic())))

    def wait_for(self, predicate, timeout, what):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.spin_for(0.05)
            if predicate():
                return
        raise RuntimeError(f"timeout waiting for {what}")

    def ensure_gate(self):
        if not self.gate.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/control_enable service not available")
        request = SetBool.Request()
        request.data = True
        future = self.gate.call_async(request)
        self.wait_for(lambda: future.done(), 5.0, "control gate")
        result = future.result()
        log(f"control gate: {result.message}")

    def board_distance(self):
        if self.aruco is None or time.monotonic() - self.aruco_time > 0.8:
            return None
        p = self.aruco.pose.position
        return float(np.linalg.norm([p.x, p.y, p.z]))

    def move_joints(self, target_deg):
        target = np.radians(np.asarray(target_deg, dtype=float))
        if target.shape != (6,):
            raise ValueError("target must contain six joints")
        margin = np.radians(1.0)
        if np.any(target < np.radians(LIMITS_DEG[:, 0]) + margin) or \
                np.any(target > np.radians(LIMITS_DEG[:, 1]) - margin):
            raise RuntimeError(f"target outside joint limits: {np.round(target_deg, 2).tolist()}")
        message = JointState()
        message.name = list(JOINT_NAMES)
        message.position = target.tolist()
        for _ in range(3):
            self.publisher.publish(message)
            self.spin_for(0.02)
        deadline = time.monotonic() + 20.0
        stable_since = None
        while time.monotonic() < deadline:
            self.spin_for(0.05)
            if self.q is None or self.status is None:
                continue
            code = int(self.status.arm_status)
            if code not in (0, 2, 3, 4):
                raise RuntimeError(f"arm_status={code} during joint move")
            error = float(np.max(np.abs(self.q - target)))
            if error < np.radians(0.3):
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since > 0.5:
                    return
            else:
                stable_since = None
        raise RuntimeError(
            f"joint move timeout: target={np.round(target_deg, 2).tolist()} "
            f"current={np.round(np.degrees(self.q), 2).tolist()}")


def start_detector(log_path):
    stream = log_path.open("w")
    process = subprocess.Popen(
        [
            str(DETECTOR), "--board", "charuco",
            "--dictionary", "DICT_4X4_50",
            "--squares-x", "7", "--squares-y", "5",
            "--square-length", "0.03", "--marker-length", "0.022",
            "--blur", "3",
        ],
        stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy(),
    )
    return process, stream


def start_calib(log_path, min_samples):
    stream = log_path.open("w")
    child = pexpect.spawn(
        "/usr/bin/python3",
        [str(CALIB), "--min", str(min_samples), "--result-dir", str(RESULT_DIR)],
        encoding="utf-8", timeout=30.0, env=os.environ.copy(),
    )
    child.logfile = stream
    child.expect("Enter-采集", timeout=60.0)
    return child, stream


def capture_sample(child):
    child.sendline("")
    index = child.expect(
        ["已采集", "⚠️", "已退出", pexpect.EOF, pexpect.TIMEOUT], timeout=20.0)
    if index in (0, 1):
        try:
            child.expect("Enter-采集", timeout=15.0)
        except (pexpect.EOF, pexpect.TIMEOUT):
            pass
        return index == 0
    return False


def newest_result():
    files = sorted(RESULT_DIR.glob("*_handeye.json"),
                   key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    detector_log = LOG_DIR / f"handeye_detector_{stamp}.log"
    calib_log = LOG_DIR / f"handeye_calib_{stamp}.log"

    rclpy.init()
    node = ArmClient()
    detector = None
    detector_stream = None
    child = None
    calib_stream = None
    home = None
    try:
        node.wait_for(lambda: node.q is not None and node.status is not None,
                      10.0, "joint state and arm status")
        home = node.q.copy()
        log(f"home joints (deg): {np.round(np.degrees(home), 2).tolist()}")
        node.ensure_gate()
        detector, detector_stream = start_detector(detector_log)
        node.wait_for(lambda: node.board_distance() is not None,
                      20.0, "aruco board pose")
        log(f"board visible, distance={node.board_distance():.3f} m")
        child, calib_stream = start_calib(calib_log, args.min_samples)
        log(f"calib collector ready; detector log={detector_log}; calib log={calib_log}")

        samples = 0
        # Capture the current home pose first.
        if node.board_distance() is not None and capture_sample(child):
            samples += 1
            log(f"[home] sample {samples}")
        else:
            log("[home] capture failed")

        for attempt, offset in enumerate(OFFSETS_DEG[1:], start=1):
            if samples >= args.max_samples:
                break
            target = np.degrees(home) + np.asarray(offset, dtype=float)
            try:
                node.move_joints(target)
            except Exception as error:
                log(f"[{attempt}] move rejected: {error}; returning home")
                try:
                    node.move_joints(np.degrees(home))
                except Exception:
                    pass
                continue
            node.spin_for(0.6)
            distance = node.board_distance()
            if distance is None:
                log(f"[{attempt}] board lost at {np.round(target, 1).tolist()}; returning home")
                try:
                    node.move_joints(np.degrees(home))
                except Exception:
                    pass
                continue
            ok = capture_sample(child)
            if ok:
                samples += 1
                current = np.degrees(node.q)
                log(f"[{attempt}] sample {samples} at joints "
                    f"{np.round(current, 1).tolist()}, board {distance:.3f} m")
            else:
                log(f"[{attempt}] capture failed at board {distance:.3f} m; returning home")
                try:
                    node.move_joints(np.degrees(home))
                except Exception:
                    pass

        if samples < args.min_samples:
            raise RuntimeError(
                f"only {samples} samples collected; need {args.min_samples}")
        child.sendline("q")
        child.expect("已保存", timeout=180.0)
        child.expect(pexpect.EOF, timeout=60.0)
        result_path = newest_result()
        log(f"calibration saved: {result_path}")
        if result_path is not None:
            data = json.loads(result_path.read_text())
            result = data.get("result", {})
            log("result: " + json.dumps({
                "method": result.get("method"),
                "num_samples": data.get("num_samples"),
                "translation_rms_mm": result.get("translation_rms_mm"),
                "translation_max_mm": result.get("translation_max_mm"),
                "rotation_rms_deg": result.get("rotation_rms_deg"),
                "rotation_max_deg": result.get("rotation_max_deg"),
            }, ensure_ascii=False))
        return 0
    finally:
        if home is not None:
            try:
                node.move_joints(np.degrees(home))
                log("returned to home joints")
            except Exception as error:
                log(f"return home failed: {error}")
        if child is not None:
            try:
                child.close(force=True)
            except Exception:
                pass
        if calib_stream is not None:
            calib_stream.close()
        if detector is not None:
            detector.terminate()
            try:
                detector.wait(timeout=5.0)
            except Exception:
                detector.kill()
        if detector_stream is not None:
            detector_stream.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
