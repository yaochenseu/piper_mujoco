"""Scene manifest loading, free-body enumeration and point-cloud matching."""
import unittest

import numpy as np

from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv
from utils.config import load_config
from utils.piperx_scene import (
    OBJECT_CLEARANCE_Z,
    TABLE_TOP_Z,
    resolve_scene_body,
    scene_bodies,
)


EXPECTED_BODIES = {
    "ycb_banana",
    "ycb_pear",
    "ycb_strawberry",
    "ycb_tennis_ball",
    "ycb_tomato_soup_can",
}


class SceneManifestTests(unittest.TestCase):
    def test_yaml_manifest_matches_runtime_constants(self):
        config = load_config("piperx_scene.yaml")
        self.assertAlmostEqual(float(config["table_top_z"]), TABLE_TOP_Z)
        self.assertAlmostEqual(float(config["object_clearance_z"]), OBJECT_CLEARANCE_Z)
        bodies = [entry["body"] for entry in config["objects"]]
        self.assertEqual(set(bodies), EXPECTED_BODIES)
        self.assertEqual(len(bodies), len(set(bodies)))


class SceneBodyTests(unittest.TestCase):
    def setUp(self):
        self.env = PiperXGraspEnv(realtime=False)
        self.addCleanup(self.env.close)

    def test_scene_bodies_are_the_free_joint_objects(self):
        bodies = scene_bodies(self.env)
        self.assertEqual(set(bodies), EXPECTED_BODIES)
        for body, joint in bodies.items():
            self.assertEqual(joint, f"{body}_free")

    def test_each_ycb_body_matches_its_own_cloud(self):
        rng = np.random.default_rng(20261019)
        for body in scene_bodies(self.env):
            body_id = self.env.model.body(body).id
            centre = self.env.data.xipos[body_id]
            points = centre + rng.normal(0, 0.008, (200, 3))
            matched, distance, report = resolve_scene_body(self.env, points)
            self.assertEqual(matched, body)
            self.assertLess(distance, 0.02)
            distances = [entry["distance_m"] for entry in report["candidates"]]
            self.assertEqual(distances, sorted(distances))

    def test_far_cloud_matches_nothing(self):
        points = np.tile([-0.5, -0.5, 1.2], (100, 1))
        matched, distance, report = resolve_scene_body(self.env, points)
        self.assertIsNone(matched)
        self.assertGreater(distance, report["max_distance_m"])

    def test_empty_cloud_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_scene_body(self.env, np.empty((0, 3)))

    def test_set_target_accepts_any_scene_body_name(self):
        self.env.set_target("ycb_tennis_ball")
        self.assertEqual(self.env.target_key, "ycb_tennis_ball")
        self.assertEqual(self.env.target_label, "ycb_tennis_ball")
        self.assertEqual(self.env.target_body, "ycb_tennis_ball")
        body_id = self.env.model.body("ycb_tennis_ball").id
        np.testing.assert_allclose(self.env.object_position, self.env.data.xipos[body_id])

    def test_set_target_accepts_object_with_body_attribute(self):
        class Spec:
            body = "ycb_pear"
            key = "pear"
            label = "pear"

        self.env.set_target(Spec())
        self.assertEqual(self.env.target_key, "pear")
        self.assertEqual(self.env.target_label, "pear")
        self.assertEqual(self.env.target_body, "ycb_pear")

    def test_unknown_body_reports_candidates(self):
        with self.assertRaises(ValueError) as context:
            self.env.set_target("no_such_body")
        message = str(context.exception)
        self.assertIn("no_such_body", message)
        for body in EXPECTED_BODIES:
            self.assertIn(body, message)

    def test_none_clears_target_and_blocks_object_position(self):
        self.env.set_target("ycb_banana")
        self.env.set_target(None)
        self.assertEqual(self.env.target_body, "")
        self.assertEqual(self.env.target_body_id, -1)
        with self.assertRaises(RuntimeError):
            _ = self.env.object_position
        self.env.step()
        self.assertEqual(len(self.env.target_geom_ids), 0)


if __name__ == "__main__":
    unittest.main()
