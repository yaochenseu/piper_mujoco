"""Physics and kinematic regression checks; no viewer or ML dependencies."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from main_piperx import run_demo
from manipulator_grasp.env.piperx_grasp_env import IKError, PiperXGraspEnv, TOP_DOWN


class PiperXSimulationTests(unittest.TestCase):
    def setUp(self):
        self.env = PiperXGraspEnv(object_type="cube")
        self.addCleanup(self.env.close)

    def test_official_joint_limits_and_physical_dofs(self):
        source = Path(__file__).resolve().parents[1] / "external/agx_arm_urdf/piper_x/urdf/piper_x_description.urdf"
        root = ET.parse(source).getroot()
        expected = [
            [float(root.find(f"joint[@name='joint{i}']/limit").get(key)) for key in ("lower", "upper")]
            for i in range(1, 7)
        ]
        np.testing.assert_allclose(self.env.limits, expected)
        self.assertEqual(self.env.model.nu, 7)
        # Six arm joints + two fingers + four free objects (six velocities each).
        self.assertEqual(self.env.model.nv, 32)
        self.assertEqual(self.env.model.neq, 1)
        self.assertEqual(int(self.env.model.eq_type[0]), int(mujoco.mjtEq.mjEQ_JOINT))

    def test_gripper_command_is_total_width(self):
        self.env.gripper_control(0.06)
        self.assertAlmostEqual(self.env.gripper_width, 0.06, delta=0.001)
        fingers = self.env.data.qpos[self.env.finger_indices]
        self.assertAlmostEqual(float(fingers.sum()), 0, delta=1e-4)

    def test_ik_checks_residual_without_changing_simulation(self):
        before = self.env.data.qpos.copy()
        target = np.array([0.40, -0.10, 0.85])
        q = self.env.solve_ik(target)
        np.testing.assert_array_equal(self.env.data.qpos, before)
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[:] = before
        scratch.qpos[self.env.q_indices] = q
        mujoco.mj_forward(self.env.model, scratch)
        np.testing.assert_allclose(scratch.site("tcp").xpos, target, atol=0.0002)
        np.testing.assert_allclose(scratch.site("tcp").xmat.reshape(3, 3), TOP_DOWN, atol=0.002)
        with self.assertRaises(IKError):
            self.env.move_to([2, 0, 2])
        np.testing.assert_array_equal(self.env.data.qpos, before)

    def test_camera_grasp_axes_and_translation(self):
        camera = self.env.camera_parameters()
        position = np.array([0.42, -0.12, 0.76])
        world_camera = camera["T_world_camera"]
        optical_point = world_camera[:3, :3].T @ (position - world_camera[:3, 3])
        self.assertGreater(optical_point[2], 0)
        # GraspNet X=approach, Y=jaw direction, Z=X cross Y.
        grasp_world_rotation = np.column_stack((TOP_DOWN[:, 2], TOP_DOWN[:, 1], -TOP_DOWN[:, 0]))
        grasp_camera_rotation = world_camera[:3, :3].T @ grasp_world_rotation
        result = self.env.camera_grasp_to_world(optical_point, grasp_camera_rotation)
        np.testing.assert_allclose(result[:3, 3], position, atol=1e-9)
        np.testing.assert_allclose(result[:3, :3], TOP_DOWN, atol=1e-9)

    def check_pick_place(self):
        unexpected_contacts = set()
        step = self.env.step

        def checked_step():
            step()
            robot = self.env.model.body_rootid[self.env.model.body("base_link").id]
            for contact in self.env.data.contact:
                names = (self.env.model.geom(contact.geom1).name, self.env.model.geom(contact.geom2).name)
                roots = self.env.model.body_rootid[
                    self.env.model.geom_bodyid[[contact.geom1, contact.geom2]]]
                # The extra free objects must contact the table to stand on it.
                # Keep rejecting robot contacts with anything except its target.
                if robot in roots and "target_geom" not in names:
                    unexpected_contacts.add(tuple(sorted(names)))

        self.env.step = checked_step
        report = run_demo(self.env, log=lambda _: None)
        self.assertTrue(report["success"])
        self.assertTrue(report["bilateral_contact"])
        self.assertGreater(report["lift_height_m"], 0.08)
        self.assertLess(report["placement_error_m"], 0.01)
        self.assertFalse(unexpected_contacts, unexpected_contacts)
        self.assertFalse(self.env.data.warning.number.any())

    def test_contact_pick_and_place(self):
        self.check_pick_place()

    def test_pick_and_place_with_offset_and_rotated_object(self):
        address = self.env.model.joint("target_free").qposadr[0]
        self.env.data.qpos[address:address + 3] = [0.395, -0.105, 0.76]
        self.env.data.qpos[address + 3:address + 7] = [np.cos(0.1), 0, 0, np.sin(0.1)]
        mujoco.mj_forward(self.env.model, self.env.data)
        self.check_pick_place()


if __name__ == "__main__":
    unittest.main()
