"""Lightweight unit tests for run_closed_loop / held_object (no ML, GPU, MuJoCo)."""
import unittest
from unittest import mock

import numpy as np

from utils import piperx_vision_control as vc

DROP = np.array([0.30, -0.10])


class TargetPointSamplingTests(unittest.TestCase):
    def test_grasp_cloud_excludes_workspace_points_outside_sam_mask(self):
        workspace = np.zeros((10, 10), dtype=bool)
        workspace[:8, :8] = True
        object_mask = np.zeros_like(workspace)
        object_mask[:3, :10] = True  # 30 target points.
        roi = np.ones_like(workspace)
        sampled, report = vc.sample_target_grasp_points(
            object_mask, roi, workspace, 60, rng=np.random.default_rng(7))
        self.assertEqual(report["target_points_in_roi"], 24)
        self.assertEqual(report["workspace_points_in_roi"], 64)
        self.assertEqual(report["sampled_unique_points"], 24)
        self.assertTrue(report["sampled_with_replacement"])
        self.assertTrue(object_mask.ravel()[sampled].all())

    def test_roi_crop_cannot_remove_target_or_add_neighbours(self):
        workspace = np.ones((5, 5), dtype=bool)
        object_mask = np.zeros_like(workspace)
        object_mask[2, 2] = True
        roi = np.zeros_like(workspace)
        roi[2, 1:4] = True
        with self.assertRaises(RuntimeError):
            vc.sample_target_grasp_points(object_mask, roi, workspace, 20, rng=np.random.default_rng(1))

    def test_no_replacement_when_target_has_enough_points(self):
        workspace = np.ones((8, 8), dtype=bool)
        object_mask = np.zeros_like(workspace)
        object_mask[:6, :6] = True  # 36 target points.
        sampled, report = vc.sample_target_grasp_points(
            object_mask, np.ones_like(workspace), workspace, 20, rng=np.random.default_rng(3))
        self.assertFalse(report["sampled_with_replacement"])
        self.assertEqual(report["sampled_unique_points"], 20)
        self.assertTrue(object_mask.ravel()[sampled].all())


class AnalyticTopDownTests(unittest.TestCase):
    @staticmethod
    def box_cloud(*, outliers=False, table_noise=False):
        rng = np.random.default_rng(11)
        points = rng.uniform([0.30, -0.10, 0.742], [0.34, -0.06, 0.780],
                             size=(600, 3))
        if outliers:
            points = np.vstack((points, [[5.0, 5.0, 0.75], [-5.0, -5.0, 0.75]]))
        if table_noise:
            table = rng.uniform([0.10, -0.30, 0.738], [0.50, 0.20, 0.7405],
                                size=(1000, 3))
            points = np.vstack((points, table))
        return points

    def test_candidates_use_robust_box_centre_and_top_down_approach(self):
        candidates = vc.top_down_grasp_candidates(self.box_cloud())
        self.assertGreaterEqual(len(candidates), 4)
        first = candidates[0]
        self.assertEqual(first["source"], "analytic_top_down")
        self.assertEqual(first["rank"], -1)
        self.assertAlmostEqual(first["world_position"][0], 0.32, delta=0.004)
        self.assertAlmostEqual(first["world_position"][1], -0.08, delta=0.004)
        self.assertAlmostEqual(first["world_position"][2], 0.762, delta=0.004)
        self.assertAlmostEqual(first["approach_cos"], 1.0, places=6)
        self.assertLessEqual(first["width"], 0.095)
        self.assertTrue(all(first["checks"].values()))

    def test_outliers_and_table_points_do_not_shift_candidate(self):
        points = self.box_cloud(outliers=True, table_noise=True)
        candidates = vc.top_down_grasp_candidates(points)
        first = candidates[0]
        self.assertAlmostEqual(first["world_position"][0], 0.32, delta=0.006)
        self.assertAlmostEqual(first["world_position"][1], -0.08, delta=0.006)

    def test_too_few_or_only_table_points_returns_no_candidate(self):
        self.assertEqual(vc.top_down_grasp_candidates(np.zeros((19, 3))), [])
        table_only = np.column_stack((
            np.linspace(0.1, 0.5, 100), np.linspace(-0.2, 0.2, 100),
            np.full(100, 0.7395)))
        self.assertEqual(vc.top_down_grasp_candidates(table_only), [])


class JawSymmetryTests(unittest.TestCase):
    def test_flip_is_180_degrees_about_approach_axis(self):
        variants = vc.parallel_jaw_rotation_variants(vc.TOP_DOWN, reference=vc.TOP_DOWN)
        self.assertEqual([item["name"] for item in variants], ["original", "flipped"])
        self.assertEqual([item["branch"] for item in variants], ["near", "far"])
        np.testing.assert_allclose(variants[1]["rotation"], vc.TOP_DOWN @ vc.JAW_FLIP)
        np.testing.assert_allclose(variants[1]["rotation"][:, 2], vc.TOP_DOWN[:, 2])
        self.assertAlmostEqual(np.rad2deg(variants[1]["angle_rad"]), 180.0, places=4)

    def test_near_branch_tracks_reference_pose(self):
        reference = vc.TOP_DOWN @ vc.JAW_FLIP
        variants = vc.parallel_jaw_rotation_variants(vc.TOP_DOWN, reference=reference)
        self.assertEqual(variants[0]["name"], "flipped")
        self.assertEqual(variants[0]["branch"], "near")


class GraspInsertionTests(unittest.TestCase):
    def test_insertion_moves_contact_centre_along_approach(self):
        tcp = np.eye(4)
        tcp[:3, :3] = vc.TOP_DOWN
        tcp[:3, 3] = [0.33, -0.14, 0.78]
        result = vc.bounded_grasp_insertion(tcp, 0.010, table_top_z=0.740)
        self.assertAlmostEqual(result["applied_depth_m"], 0.010)
        self.assertFalse(result["table_limited"])
        np.testing.assert_allclose(result["tcp"][:3, 3], [0.33, -0.14, 0.770])

    def test_insertion_is_clipped_before_fingertip_hits_table(self):
        tcp = np.eye(4)
        tcp[:3, :3] = vc.TOP_DOWN
        tcp[:3, 3] = [0.33, -0.14, 0.756]
        result = vc.bounded_grasp_insertion(tcp, 0.010, table_top_z=0.740,
                                            tip_ahead=0.013, table_clearance=0.002)
        self.assertEqual(result["table_limited"], True)
        self.assertAlmostEqual(result["applied_depth_m"], 0.001)
        self.assertAlmostEqual(result["tcp"][2, 3], 0.755)

    def test_requested_depth_must_be_nonnegative(self):
        with self.assertRaises(ValueError):
            vc.bounded_grasp_insertion(np.eye(4), -0.001)


class TwoFingerSupportTests(unittest.TestCase):
    @staticmethod
    def _cloud(y_values):
        x, y = np.meshgrid(np.linspace(-0.012, 0.012, 12), y_values)
        return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    def test_two_sided_object_passes(self):
        points = self._cloud(np.linspace(-0.03, 0.03, 12))
        result = vc.two_finger_support(points, np.eye(4), width=0.06)
        self.assertTrue(result["supported"])
        self.assertGreater(result["left_points"], 0)
        self.assertGreater(result["right_points"], 0)

    def test_endpoint_grasp_with_one_empty_side_fails(self):
        points = self._cloud(np.linspace(0.005, 0.05, 12))
        result = vc.two_finger_support(points, np.eye(4), width=0.06)
        self.assertFalse(result["supported"])
        self.assertIn("right_side_sparse", result["reason"])

    def test_object_wider_than_jaw_fails(self):
        points = self._cloud(np.linspace(-0.05, 0.05, 20))
        result = vc.two_finger_support(points, np.eye(4), width=0.06)
        self.assertFalse(result["supported"])
        self.assertIn("object_too_wide", result["reason"])


class FakeEnv:
    """Only implements what the loop touches; truth-like attributes raise."""

    def __init__(self, held_sequence=None):
        self.tcp_pose = np.eye(4)
        self.tcp_pose[:3, 3] = [0.20, -0.10, 0.70]
        self.executed = []
        self.gripper = []
        # simulate tactile sensor: list of bilateral flags consumed per held_object() call
        self._held = list(held_sequence or [])

    def __getattr__(self, name):
        if name in ("object_position", "bilateral_contacts", "max_object_height"):
            raise AssertionError(f"loop must not read env.{name}")
        raise AttributeError(name)

    def gripper_control(self, value):
        self.gripper.append(value)

    def finger_contact_state(self):
        bilateral = self._held.pop(0) if self._held else True
        return {"bilateral": bilateral, "width": 0.03 if bilateral else 0.001}

    def wait(self, _t):
        pass

    def plan_to(self, *a, **k):
        return {"name": "support_retreat", "goals": [np.zeros(6)],
                "start_joint": np.zeros(6), "duration": 1.0}

    def execute_plan(self, plan, **_kwargs):
        self.executed.append(plan["name"])
        if plan["name"] == "descend":
            self.tcp_pose[2, 3] = 0.78
        elif plan["name"] == "lift_probe":
            self.tcp_pose[2, 3] += 0.02
        elif plan["name"] == "lift":
            self.tcp_pose[2, 3] = 0.88

    def _plan(self, name):
        return {"name": name, "goals": [np.zeros(6)]}


def make_observe():
    calls = []

    def observe(cycle, phase):
        calls.append((cycle, phase))
        c = f"cycle_{cycle}/{phase}"
        centroid_by_phase = {"pickup": [0.30, -0.10, 0.785],
                             "pregrasp": [0.30, -0.10, 0.785],
                             "pregrasp_retry": [0.30, -0.10, 0.785],
                             "placement": [0.30, -0.10, 0.78]}
        return {"id": c, "centroid": np.array(centroid_by_phase[phase]),
                "object_points": np.zeros((0, 3))}

    observe.calls = calls
    return observe


def make_propose():
    calls = []

    def propose(observation, cycle, home_pose):
        calls.append(observation["id"])
        plans = {k: {"name": k, "goals": [np.zeros(6)]}
                 for k in ("approach", "descend", "lift_probe", "lift", "transfer",
                           "place", "retreat", "observe")}
        return {"rank": 0, "score": 1.0, "plans": plans}

    propose.calls = calls
    return propose


class ClosedLoopTests(unittest.TestCase):
    def test_success_without_reading_object_truth(self):
        env = FakeEnv(held_sequence=[True] * 12)  # lift(6) + transfer(6)
        with mock.patch.object(vc, "held_object",
                               side_effect=lambda e, samples=6: {"held": True, "contact_fraction": 1.0,
                                                                 "width": 0.03, "current_bilateral": True}):
            out = vc.run_closed_loop(env, make_observe(), make_propose(),
                                     drop_xy=DROP, max_attempts=1, log=lambda *_: None)
        self.assertTrue(out["policy_success"])
        self.assertIn("approach", env.executed)

    def test_first_tactile_failure_then_fresh_observation_and_plan(self):
        env = FakeEnv()
        observe, propose = make_observe(), make_propose()
        held = [False, True, True, True, True]  # cycle0 lift fails, cycle1 lift+transfer ok
        with mock.patch.object(vc, "held_object",
                               side_effect=lambda e, samples=6: {"held": held.pop(0),
                                                                 "contact_fraction": 0.0, "width": 0.03,
                                                                 "current_bilateral": False}):
            with mock.patch.object(vc, "recover") as recover:
                out = vc.run_closed_loop(env, observe, propose, drop_xy=DROP,
                                         max_attempts=2, log=lambda *_: None)
        self.assertTrue(out["policy_success"])
        recover.assert_called_once()
        self.assertEqual(observe.calls[:3], [(0, "pickup"), (1, "pickup"), (1, "placement")])
        # propose fed with observations from distinct pickup calls (fresh, not reused)
        self.assertEqual(len(set(propose.calls)), len(propose.calls))
        self.assertEqual(len(propose.calls), 2)

    def test_historical_contact_but_current_false_fails(self):
        env = FakeEnv(held_sequence=[True] * 5 + [False])
        reading = vc.held_object(env)
        self.assertFalse(reading["held"])
        self.assertFalse(reading["current_bilateral"])

    def test_empty_jaws_cannot_confirm_holding(self):
        env = FakeEnv(held_sequence=[False] * 6)
        self.assertFalse(vc.held_object(env)["held"])

    def test_max_attempts_one_does_not_retry(self):
        env = FakeEnv()
        observe, propose = make_observe(), make_propose()
        with mock.patch.object(vc, "held_object", return_value={"held": False, "contact_fraction": 0.0,
                                                                "width": 0.03, "current_bilateral": False}):
            with mock.patch.object(vc, "recover") as recover:
                out = vc.run_closed_loop(env, observe, propose, drop_xy=DROP,
                                         max_attempts=1, log=lambda *_: None)
        self.assertFalse(out["policy_success"])
        self.assertEqual(len(out["attempts"]), 1)
        self.assertEqual(observe.calls, [(0, "pickup")])
        recover.assert_not_called()

    def test_reorientation_segment_runs_after_lift_and_is_checked(self):
        env = FakeEnv()
        observe = make_observe()
        base_propose = make_propose()

        def propose_with_reorient(observation, cycle, home_pose):
            result = base_propose(observation, cycle, home_pose)
            result["plans"]["reorient"] = {"name": "reorient", "goals": [np.zeros(6)]}
            return result

        held = {"held": True, "contact_fraction": 1.0, "width": 0.03,
                "current_bilateral": True}
        with mock.patch.object(vc, "held_object", return_value=held):
            out = vc.run_closed_loop(env, observe, propose_with_reorient, drop_xy=DROP,
                                     max_attempts=1, log=lambda *_: None)
        self.assertTrue(out["policy_success"])
        self.assertIn("reorient", env.executed)
        self.assertLess(env.executed.index("lift"), env.executed.index("reorient"))
        self.assertLess(env.executed.index("reorient"), env.executed.index("transfer"))


class TwoStageLoopTests(unittest.TestCase):
    def test_pregrasp_then_final_and_support_retry(self):
        env = FakeEnv()
        observe = make_observe()
        final_calls = []

        def full_plans():
            return {name: {"name": name, "goals": [np.zeros(6)]}
                    for name in ("approach", "descend", "lift_probe", "lift", "transfer",
                                 "place", "retreat", "observe")}

        def propose_pregrasp(observation, cycle, home_pose):
            plans = full_plans()
            plans["approach"] = {"name": "pregrasp_approach", "goals": [np.zeros(6)]}
            return {"rank": 2, "score": 0.9, "width": 0.04,
                    "world_tcp": np.eye(4), "plans": plans}

        def propose_final(observation, cycle, home_pose, *, exclude=None):
            final_calls.append(set(exclude or ()))
            rank = 3 if not exclude else 4
            return {"rank": rank, "score": 0.8 - rank * 0.01, "width": 0.04,
                    "world_tcp": np.eye(4), "jaw_variant": "original",
                    "jaw_branch": "near", "plans": full_plans()}

        support = [
            {"supported": False, "reason": "left_side_sparse"},
            {"supported": True, "reason": None},
        ]
        held = {"held": True, "contact_fraction": 1.0, "width": 0.03,
                "current_bilateral": True}
        with mock.patch.object(vc, "two_finger_support", side_effect=support), \
                mock.patch.object(vc, "held_object", return_value=held):
            out = vc.run_closed_loop(
                env, observe, make_propose(), drop_xy=DROP, max_attempts=1,
                log=lambda *_: None, two_stage=True,
                propose_pregrasp=propose_pregrasp, propose_final=propose_final,
                support_check=True, max_support_trials=2,
                reobserve_final=lambda cycle: observe(cycle, "pregrasp_retry"))
        self.assertTrue(out["policy_success"])
        self.assertEqual(out["selected"]["rank"], 4)
        self.assertEqual(final_calls, [set(), {3}])
        self.assertEqual(observe.calls,
                         [(0, "pickup"), (0, "pregrasp"), (0, "pregrasp_retry"),
                          (0, "placement")])
        self.assertEqual(env.executed.count("pregrasp_approach"), 1)
        self.assertEqual(env.executed.count("approach"), 2)
        retreat_index = env.executed.index("support_retreat")
        self.assertEqual(env.executed[:retreat_index].count("approach"), 1)
        self.assertLess(retreat_index, env.executed.index("descend"))
        checks = out["attempts"][0]["support_checks"]
        self.assertEqual(len(checks), 2)
        self.assertFalse(checks[0]["supported"])
        self.assertTrue(checks[1]["supported"])
        retreats = out["attempts"][0]["support_retreats"]
        self.assertEqual(len(retreats), 1)
        self.assertEqual(retreats[0]["rank"], 3)
        self.assertEqual(retreats[0]["status"], "returned")
        self.assertLess(env.executed.index("approach"), env.executed.index("descend"))

    def test_post_close_lift_probe_rejects_and_retries_next_candidate(self):
        env = FakeEnv()
        observe = make_observe()
        final_calls = []

        def full_plans():
            return {name: {"name": name, "goals": [np.zeros(6)]}
                    for name in ("approach", "descend", "lift_probe", "lift", "transfer",
                                 "place", "retreat", "observe")}

        def propose_pregrasp(observation, cycle, home_pose):
            plans = full_plans()
            plans["approach"] = {"name": "pregrasp_approach", "goals": [np.zeros(6)]}
            return {"rank": 2, "score": 0.9, "width": 0.04,
                    "world_tcp": np.eye(4), "plans": plans}

        def propose_final(observation, cycle, home_pose, *, exclude=None):
            final_calls.append(set(exclude or ()))
            rank = 3 if not exclude else 4
            return {"rank": rank, "score": 0.8 - rank * 0.01, "width": 0.04,
                    "world_tcp": np.eye(4), "jaw_variant": "original",
                    "jaw_branch": "near", "plans": full_plans()}

        lost = {"held": False, "contact_fraction": 0.0, "width": 0.0,
                "current_bilateral": False}
        held = {"held": True, "contact_fraction": 1.0, "width": 0.03,
                "current_bilateral": True}
        with mock.patch.object(vc, "two_finger_support",
                               return_value={"supported": True, "reason": None}), \
                mock.patch.object(vc, "held_object", side_effect=[lost, held, held, held]):
            out = vc.run_closed_loop(
                env, observe, make_propose(), drop_xy=DROP, max_attempts=1,
                log=lambda *_: None, two_stage=True,
                propose_pregrasp=propose_pregrasp, propose_final=propose_final,
                support_check=True, max_support_trials=2)
        self.assertTrue(out["policy_success"])
        self.assertEqual(out["selected"]["rank"], 4)
        self.assertEqual(final_calls, [set(), {3}])
        retreats = out["attempts"][0]["prelift_retreats"]
        self.assertEqual(len(retreats), 1)
        self.assertEqual(retreats[0]["rank"], 3)
        self.assertEqual(retreats[0]["status"], "returned")
        self.assertGreaterEqual(env.executed.count("lift_probe"), 2)


if __name__ == "__main__":
    unittest.main()
