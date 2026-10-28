"""Real bridge client tests without ROS, CUDA or hardware."""
import base64
import json
import unittest
import zlib

import cv2
import numpy as np

from utils.piperx_real_env import RealPiperXEnv
from utils.piperx_vision_control import TOP_DOWN


class FakeBridge:
    socket_path = "/tmp/fake-piperx-bridge.sock"

    def __init__(self):
        self.requests = []
        self.width = 0.10
        self.force = 0.0
        self.tcp = np.eye(4)
        self.tcp[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        self.tcp[:3, 3] = [0.30, 0.0, 0.20]

    def request(self, command, **payload):
        self.requests.append((command, payload))
        if command == "ping":
            return {"name": "fake"}
        if command == "state":
            joints = {
                f"joint{i}": 0.01 * i for i in range(1, 7)
            }
            return {
                "T_base_tcp": self.tcp.tolist(),
                "T_base_flange": self.tcp.tolist(),
                "arm_rad": joints,
                "gripper_width_m": self.width,
                "gripper_force_n": self.force,
                "arm_status": {
                    "arm_status": 0, "motion_status": 0,
                    "mode_feedback": 0, "ctrl_mode": 0,
                },
            }
        if command == "camera":
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb[..., 0] = 200
            ok, encoded = cv2.imencode(".jpg", rgb[..., ::-1])
            if not ok:
                raise RuntimeError("encode failed")
            depth = np.full((8, 8), 0.25, dtype=np.float32)
            return {
                "rgb_jpeg_b64": base64.b64encode(encoded.tobytes()).decode(),
                "depth_zlib_b64": base64.b64encode(
                    zlib.compress(depth.tobytes(), 3)).decode(),
                "depth_dtype": "float32",
                "width": 8,
                "height": 8,
                "fx": 6.0,
                "fy": 6.0,
                "ppx": 4.0,
                "ppy": 4.0,
                "depth_scale_m": 1.0,
                "T_base_flange": self.tcp.tolist(),
                "T_base_tcp": self.tcp.tolist(),
                "T_base_camera": self.tcp.tolist(),
                "frame_age_s": 0.0,
            }
        if command in ("move_tcp", "gripper", "open_gate", "estop"):
            if command == "move_tcp" and not payload.get("dry_run"):
                self.tcp = np.asarray(payload["tcp_pose"], dtype=float)
            if command == "gripper" and not payload.get("dry_run"):
                self.width = float(payload["width_m"])
            return {"dry_run": bool(payload.get("dry_run", False))}
        raise AssertionError(command)

    def close(self):
        pass


class RealEnvTests(unittest.TestCase):
    def setUp(self):
        self.bridge = FakeBridge()
        self.env = RealPiperXEnv(
            config_path="real_piperx.yaml",
            execute=False,
            log=lambda *_args: None,
            client=self.bridge,
        )

    def tearDown(self):
        self.env.close()

    def test_tcp_frames_are_consistent(self):
        np.testing.assert_allclose(
            self.env.T_flange_tcp[:3, 3], [0.0, 0.0, 0.1295], atol=1e-9)
        np.testing.assert_allclose(
            self.env.T_flange_tcp[:3, 2], [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(
            self.env.camera_offset_from_tcp(), self.env.T_tcp_cam[:3, 3])
        self.assertTrue(np.isfinite(self.env.T_tcp_cam).all())

    def test_render_decodes_d435i_snapshot(self):
        frame = self.env.render()
        self.assertEqual(frame["img"].shape, (8, 8, 3))
        self.assertEqual(frame["depth"].shape, (8, 8))
        camera = self.env.camera_parameters()
        self.assertEqual(camera["width"], 8)
        self.assertAlmostEqual(camera["fx"], 6.0)
        np.testing.assert_allclose(
            camera["T_world_camera"], self.bridge.tcp)

    def test_plan_and_dry_run_do_not_send_motion(self):
        target = np.array([0.31, 0.01, 0.20])
        plan = self.env.plan_to(target, TOP_DOWN)
        self.assertEqual(plan["target_tcp"][:3, 3].tolist(), target.tolist())
        result = self.env.execute_plan(plan)
        self.assertTrue(result["dry_run"])
        move_requests = [
            payload for command, payload in self.bridge.requests
            if command == "move_tcp"
        ]
        self.assertTrue(move_requests[-1]["dry_run"])

    def test_outside_workspace_is_rejected(self):
        with self.assertRaises(ValueError):
            self.env.plan_to([-1.0, 0.0, 0.2], TOP_DOWN)
        with self.assertRaises(ValueError):
            self.env.plan_to([0.3, 0.0, -0.1], TOP_DOWN)

    def test_urdf_reach_sphere_rejects_table_corner(self):
        np.testing.assert_allclose(self.env.reach_center, [0.0, 0.0, 0.123])
        self.assertAlmostEqual(self.env.reach_radius_m, 0.80)
        # Bottle from the 2026-10-28 run: inside the table and the reach ball.
        self.env._check_pose(np.array([0.11, 0.40, 0.03]), TOP_DOWN)
        # Inside x/y bounds, but farther than the URDF chain can reach.
        with self.assertRaisesRegex(ValueError, "reach sphere"):
            self.env._check_pose(np.array([0.65, 0.55, 0.03]), TOP_DOWN)

    def test_gripper_force_is_only_a_proxy(self):
        self.bridge.force = 0.0
        self.bridge.width = 0.03
        self.assertFalse(self.env.finger_contact_state()["bilateral"])
        self.bridge.force = 0.5
        state = self.env.finger_contact_state()
        self.assertTrue(state["bilateral"])
        self.assertEqual(state["source"], "gripper_force_proxy")

    def test_gripper_dry_run(self):
        result = self.env.gripper_control(0.05)
        self.assertTrue(result["dry_run"])


if __name__ == "__main__":
    unittest.main()
