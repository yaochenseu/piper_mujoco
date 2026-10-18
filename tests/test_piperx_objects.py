"""Rigid bottle physics and preservation of the original cube option."""
import unittest

import mujoco
import numpy as np

from main_piperx import run_demo
from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv
from utils.piperx_bottle_grasp import fit_bottle_body


class ObjectTests(unittest.TestCase):
    def test_bottle_is_one_free_rigid_body(self):
        with PiperXGraspEnv(realtime=False, object_type="bottle") as env:
            target = env.model.body("target")
            self.assertAlmostEqual(float(target.mass[0]), 0.04)
            self.assertEqual(env.model.nv, 32)  # Arm/fingers + four free objects.
            self.assertEqual(env.model.neq, 1)  # Only the finger coupling.
            self.assertEqual(int(env.model.joint("target_free").type[0]),
                             int(mujoco.mjtJoint.mjJNT_FREE))
            for name in ("target_geom", "target_shoulder", "target_neck", "target_cap"):
                geom = env.model.geom(name)
                self.assertEqual(int(geom.bodyid[0]), target.id)
                self.assertNotEqual(int(geom.contype[0]), 0)
            self.assertEqual(int(env.model.geom("target_label").contype[0]), 0)

    def test_bottle_pick_place_uses_contacts(self):
        with PiperXGraspEnv(realtime=False, object_type="bottle") as env:
            report = run_demo(env, log=lambda _: None)
            self.assertTrue(report["success"])
            self.assertTrue(report["bilateral_contact"])
            self.assertGreater(report["lift_height_m"], 0.08)
            self.assertFalse(env.data.warning.number.any())

    def test_cube_replaces_all_bottle_parts(self):
        with PiperXGraspEnv(realtime=False, object_type="cube") as env:
            self.assertEqual(env.object_type, "cube")
            self.assertEqual(int(env.model.geom("target_geom").type[0]),
                             int(mujoco.mjtGeom.mjGEOM_BOX))
            self.assertEqual(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM,
                                             "target_neck"), -1)
            self.assertAlmostEqual(float(env.model.body("target").mass[0]), 0.04)

    def test_body_center_from_partially_visible_wall(self):
        angle, z = np.meshgrid(np.linspace(-0.9, 0.9, 80), np.linspace(0.744, 0.782, 50))
        points = np.column_stack((0.42 + 0.017 * np.cos(angle.ravel()),
                                  -0.12 + 0.017 * np.sin(angle.ravel()), z.ravel()))
        points += np.random.default_rng(4).normal(0, 0.00005, points.shape)
        fit = fit_bottle_body(points)
        np.testing.assert_allclose(fit["center"][:2], [0.42, -0.12], atol=0.0005)
        self.assertAlmostEqual(fit["radius_m"], 0.017, delta=0.0005)
        self.assertTrue(0.75 < fit["center"][2] < 0.78)

    def test_body_fit_rejects_insufficient_geometry(self):
        with self.assertRaises(RuntimeError):
            fit_bottle_body(np.ones((10, 3)))
        # A single vertical line cannot determine a bottle axis/radius.
        with self.assertRaises(RuntimeError):
            fit_bottle_body(np.column_stack((np.zeros(100), np.zeros(100),
                                            np.linspace(0.744, 0.81, 100))))


if __name__ == "__main__":
    unittest.main()
