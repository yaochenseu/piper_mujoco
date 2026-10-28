"""Model IK preflight for real grasp poses."""
import unittest

import numpy as np

from utils.piperx_reachability import Reachability
from utils.piperx_vision_control import APPROACH_HEIGHT


SUCCESS_TCP = np.array([
    [-0.35385435407055454, -0.9345650641971059, -0.03708329582698677,
     0.13083917394282618],
    [-0.8383556031283406, 0.2993494780244661, 0.45557626212770286,
     0.4281427359142468],
    [-0.4146647455833616, 0.19229665179844974, -0.8894240952044978,
     0.14811647791505306],
    [0.0, 0.0, 0.0, 1.0],
])

FAILED_TCP = np.array([
    [-0.5787211232235832, 0.6909016535962053, 0.4332859191465037,
     0.13125957242410044],
    [0.5882428674153175, 0.7216299297344277, -0.36499405211859914,
     0.41980758723520617],
    [-0.5648470560509486, 0.043647547462430214, -0.8240404553947674,
     0.15223356304274296],
    [0.0, 0.0, 0.0, 1.0],
])


def with_pregrasp(tcp):
    pregrasp = tcp.copy()
    pregrasp[:3, 3] = tcp[:3, 3] - tcp[:3, :3][:, 2] * APPROACH_HEIGHT
    return pregrasp


class ReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = Reachability()

    @classmethod
    def tearDownClass(cls):
        cls.checker.close()

    def test_successful_bottle_pose_is_reachable(self):
        result = self.checker.feasible(SUCCESS_TCP, with_pregrasp(SUCCESS_TCP))
        self.assertTrue(result["feasible"], result)

    def test_unreachable_bottle_pose_is_rejected(self):
        result = self.checker.feasible(FAILED_TCP, with_pregrasp(FAILED_TCP))
        self.assertFalse(result["feasible"])
        self.assertIn(result["failed_pose"], ("pregrasp", "final"))


if __name__ == "__main__":
    unittest.main()
