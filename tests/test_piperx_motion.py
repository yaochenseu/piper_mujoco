"""Trajectory preflight must be side-effect free and reject stale commands."""
import unittest

import mujoco
import numpy as np

from manipulator_grasp.env.piperx_grasp_env import IKError, PiperXGraspEnv, TOP_DOWN


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.env = PiperXGraspEnv(realtime=False, target="ycb_banana")
        self.addCleanup(self.env.close)

    def test_preflight_chains_without_changing_live_state(self):
        env = self.env
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time]
        first = env.plan_to([0.4, -0.1, 0.85])
        second = env.plan_to([0.4, -0.1, 0.78], start_joint=first["goals"][-1])
        np.testing.assert_array_equal(first["goals"][-1], second["start_joint"])
        np.testing.assert_allclose(env._fk_pose(second["goals"][-1])[:3, 3], [0.4, -0.1, 0.78], atol=0.0002)
        for saved, actual in zip(before, [env.data.qpos, env.data.qvel, env.data.ctrl, env.data.time]):
            np.testing.assert_array_equal(saved, actual)

    def test_stale_plan_rejected_without_motion(self):
        env = self.env
        plan = env.plan_to([0.4, -0.1, 0.85])
        env.data.qpos[env.q_indices[0]] += 0.06
        mujoco.mj_forward(env.model, env.data)
        q, ctrl, time = env.data.qpos.copy(), env.data.ctrl.copy(), env.data.time
        with self.assertRaisesRegex(IKError, "stale"):
            env.execute_plan(plan)
        np.testing.assert_array_equal(env.data.qpos, q)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
        self.assertEqual(env.data.time, time)

    def test_shortest_rotation_interpolation_is_continuous(self):
        # +179 -> -179 degrees must pass through 180, not zero.
        def rz(angle):
            a = np.deg2rad(angle)
            return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        middle = self.env._slerp(rz(179), rz(-179), 0.5)
        np.testing.assert_allclose(middle, rz(180), atol=1e-9)
        np.testing.assert_allclose(self.env._slerp(TOP_DOWN, TOP_DOWN, 0.5), TOP_DOWN, atol=1e-9)

    def test_current_tactile_contact_clears_after_opening(self):
        env = self.env
        env.wait(0.4)
        point = env.object_position + [0, 0, 0.003]  # Independent fixture setup.
        env.move_to(point + [0, 0, 0.1])
        env.move_to(point)
        env.gripper_control(0)
        self.assertTrue(env.finger_contact_state()["bilateral"])
        env.move_to(point + [0, 0, 0.1])
        env.gripper_control(0.1)
        self.assertFalse(env.finger_contact_state()["bilateral"])
        self.assertTrue(env.bilateral_contacts)  # Historical flag is intentionally still true.


if __name__ == "__main__":
    unittest.main()
