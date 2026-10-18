"""Eye-in-hand calibration follows the arm; old frames retain their own pose."""
import unittest

import numpy as np

from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv, TOP_DOWN


class WristCameraTests(unittest.TestCase):
    def setUp(self):
        self.env = PiperXGraspEnv(realtime=False)
        self.addCleanup(self.env.close)

    def test_camera_follows_gripper_with_fixed_relative_transform(self):
        env = self.env
        camera_id = env.model.camera("wrist").id
        body = int(env.model.cam_bodyid[camera_id])
        ancestors = []
        while body:
            ancestors.append(body)
            body = int(env.model.body_parentid[body])
        self.assertIn(env.model.body("gripper_base").id, ancestors)
        before = env.camera_parameters()["T_world_camera"]
        relative = np.linalg.inv(env.tcp_pose) @ before
        env.move_to([0.36, -0.12, 0.87])
        after = env.camera_parameters()["T_world_camera"]
        self.assertGreater(np.linalg.norm(before[:3, 3] - after[:3, 3]), 0.02)
        np.testing.assert_allclose(np.linalg.inv(env.tcp_pose) @ after, relative, atol=1e-8)

    def test_saved_frame_pose_is_used_after_camera_moves(self):
        env = self.env
        old = env.camera_parameters()["T_world_camera"].copy()
        point = np.array([0.42, -0.12, 0.776])
        camera_point = old[:3, :3].T @ (point - old[:3, 3])
        grasp_world = np.column_stack((TOP_DOWN[:, 2], TOP_DOWN[:, 1], -TOP_DOWN[:, 0]))
        camera_rotation = old[:3, :3].T @ grasp_world
        env.move_to([0.36, -0.12, 0.87])
        restored = env.camera_grasp_to_world(camera_point, camera_rotation, T_world_camera=old)
        np.testing.assert_allclose(restored[:3, 3], point, atol=1e-9)
        np.testing.assert_allclose(restored[:3, :3], TOP_DOWN, atol=1e-9)
        live = env.camera_grasp_to_world(camera_point, camera_rotation)
        self.assertGreater(np.linalg.norm(live[:3, 3] - point), 0.02)

    def test_pickup_and_drop_regions_are_inside_initial_field_of_view(self):
        camera = self.env.camera_parameters("wrist", 1280, 960)
        transform = camera["T_world_camera"]
        for point in [[0.42, -0.12, 0.776], [0.30, 0.08, 0.776], [0.42, 0.16, 0.776]]:
            optical = transform[:3, :3].T @ (np.asarray(point) - transform[:3, 3])
            self.assertGreater(optical[2], 0)
            u = camera["ppx"] + camera["fx"] * optical[0] / optical[2]
            v = camera["ppy"] + camera["fy"] * optical[1] / optical[2]
            self.assertTrue(0 <= u < camera["width"] and 0 <= v < camera["height"], (point, u, v))


if __name__ == "__main__":
    unittest.main()
