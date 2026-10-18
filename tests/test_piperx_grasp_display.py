"""Check grasp marker frame conventions and rendering without modifying physics."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np

from utils.piperx_grasp_display import GraspPoseDisplay, append_grasp_geoms
from utils.piperx_grasp_geometry import grasp_segments


def selection(pose):
    return {"world_tcp": pose.tolist(), "width": 0.046, "rank": 1, "score": 0.4}


class GraspDisplayTests(unittest.TestCase):
    def test_axes_and_opening_follow_world_tcp(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [0.4, -0.1, 0.77]
        segments = grasp_segments(pose, 0.046)
        axes = [s for s in segments if s["kind"] == "axis"]
        for i, segment in enumerate(axes):
            np.testing.assert_allclose(segment["start"], pose[:3, 3])
            np.testing.assert_allclose(segment["end"] - segment["start"], 0.05 * pose[:3, i])
        fingers = [s for s in segments if s["label"].startswith("finger_")]
        np.testing.assert_allclose(fingers[1]["start"] - fingers[0]["start"],
                                   0.046 * pose[:3, 1])
        arrow = segments[-1]
        self.assertGreater(np.dot(arrow["end"] - arrow["start"], pose[:3, 2]), 0)

    def test_rgb_projection_uses_camera_frame_and_preserves_source(self):
        pose = np.eye(4)
        pose[2, 3] = 1
        display = GraspPoseDisplay(SimpleNamespace(viewer=None))
        display.set_grasp(selection(pose))
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        camera = {"T_world_camera": np.eye(4), "fx": 200, "fy": 200, "ppx": 100, "ppy": 100}
        result = display.annotate_rgb(image, camera)
        self.assertFalse(image.any())
        self.assertTrue(result.any())
        # TCP +X projects to the right of image centre and is colored red.
        self.assertGreater(int(result[100, 105, 0]), int(result[100, 105, 1]))
        display.clear()
        np.testing.assert_array_equal(display.annotate_rgb(image, camera), image)

    def test_viewer_marker_replaced_on_new_selection_and_cleared(self):
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        scene = mujoco.MjvScene(model, maxgeom=64)
        viewer = SimpleNamespace(user_scn=scene, is_running=lambda: True, lock=nullcontext,
                                 set_texts=Mock(), clear_texts=Mock(), sync=Mock())
        display = GraspPoseDisplay(SimpleNamespace(viewer=viewer))
        pose = np.eye(4)
        display.set_grasp(selection(pose))
        count = scene.ngeom
        self.assertGreater(count, 0)
        pose[:3, 3] = [0.3, 0.1, 0.76]
        display.set_grasp(selection(pose), cycle=1)
        self.assertEqual(scene.ngeom, count)
        self.assertEqual(display.info["cycle"], 1)
        np.testing.assert_allclose(display.info["position_m"], pose[:3, 3])
        display.clear()
        self.assertEqual(scene.ngeom, 0)
        self.assertIsNone(display.pose)

    def test_geometry_is_visual_only_and_preserves_existing_scene(self):
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/>"
                                                "<geom type='sphere' size='.02' mass='.04'/>"
                                                "</body></worldbody></mujoco>")
        data = mujoco.MjData(model)
        before = data.qpos.copy()
        scene = mujoco.MjvScene(model, maxgeom=64)
        scene.ngeom = 1
        scene.geoms[0].rgba[:] = [0.2, 0.3, 0.4, 1]
        pose = np.eye(4)
        append_grasp_geoms(scene, grasp_segments(pose, 0.046), pose[:3, 3])
        np.testing.assert_allclose(scene.geoms[0].rgba, [0.2, 0.3, 0.4, 1])
        self.assertEqual(model.ngeom, 1)
        np.testing.assert_array_equal(data.qpos, before)
        for i in range(1, scene.ngeom):
            self.assertEqual(scene.geoms[i].category, int(mujoco.mjtCatBit.mjCAT_DECOR))


if __name__ == "__main__":
    unittest.main()
