"""CPU regressions for the banana diagnostic scene."""
import unittest

import numpy as np

from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv
from utils.piperx_vision_control import (
    parallel_jaw_rotation_variants,
    plan_pick_place,
)


RANK9_TCP = np.array([
    [0.8367962274961306, -0.13217447233830684, 0.5313210272881281, 0.39131099838980116],
    [-0.2577935935023355, -0.9512388664925872, 0.1693725980062355, -0.12720214743680325],
    [0.48302643748828045, -0.2787014905579829, -0.8300668711908721, 0.7735038763382263],
    [0.0, 0.0, 0.0, 1.0],
])

RANK6_TCP = np.array([
    [-0.8484122325185213, -0.0473676193096319, -0.5272124338457836, 0.37614931559304715],
    [0.11902611809951844, 0.9534107183324798, -0.27720142842722195, -0.10761742095625022],
    [0.5157804016910388, -0.29793313973654467, -0.8032474213162774, 0.775966145330008],
    [0.0, 0.0, 0.0, 1.0],
])

RANK7_TCP = np.array([
    [0.8304638990109446, -0.21169943333914149, -0.5152797494181542, 0.40580595672416564],
    [-0.09870600034481139, -0.9662611351647181, 0.23790031894269723, -0.12852165464283238],
    [-0.5482581837713169, -0.14670643048107054, -0.8233409703947518, 0.773448420379936],
    [0.0, 0.0, 0.0, 1.0],
])


class GraspSymmetryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.env = PiperXGraspEnv(realtime=False, viewer=False)
        self.addCleanup(self.env.close)
        self.home = self.env.tcp_pose.copy()

    def test_rank9_is_rescued_by_the_equivalent_jaw_attitude(self):
        variants = parallel_jaw_rotation_variants(
            RANK9_TCP[:3, :3], reference=self.home[:3, :3])
        near, far = variants
        self.assertTrue(near["flipped"])
        near_tcp = RANK9_TCP.copy()
        near_tcp[:3, :3] = near["rotation"]
        plans = plan_pick_place(self.env, near_tcp, (0.22, -0.34), self.home)
        self.assertEqual(set(plans), {"approach", "descend", "lift_probe", "lift", "transfer",
                                      "place", "retreat", "observe"})
        far_tcp = RANK9_TCP.copy()
        far_tcp[:3, :3] = far["rotation"]
        with self.assertRaises(RuntimeError):
            plan_pick_place(self.env, far_tcp, (0.22, -0.34), self.home)

    def test_rank6_and_rank7_remain_unreachable_after_symmetry_resolution(self):
        for tcp in (RANK6_TCP, RANK7_TCP):
            for variant in parallel_jaw_rotation_variants(
                    tcp[:3, :3], reference=self.home[:3, :3]):
                candidate = tcp.copy()
                candidate[:3, :3] = variant["rotation"]
                with self.assertRaises(RuntimeError):
                    plan_pick_place(self.env, candidate, (0.22, -0.34), self.home)


if __name__ == "__main__":
    unittest.main()
