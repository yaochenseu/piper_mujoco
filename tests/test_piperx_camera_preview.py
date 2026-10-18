"""Display updates use fresh frames, fit the viewport and release GL resources."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from utils.piperx_camera_preview import WristCameraPreview, depth_rgb


class PreviewTests(unittest.TestCase):
    def test_depth_scale_handles_invalid_values(self):
        depth = np.array([[0, np.nan, np.inf, -1, 0.1, 0.8]], dtype=float)
        image = depth_rgb(depth)
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[0, :4], 0)
        self.assertFalse(np.array_equal(image[0, 4], image[0, 5]))
        np.testing.assert_array_equal(depth_rgb(np.array([[0.1]]))[0, 0], image[0, 4])

    def test_live_updates_resize_throttle_and_close(self):
        viewer = Mock()
        viewer.is_running.return_value = True
        viewer.viewport = mujoco.MjrRect(270, 0, 740, 720)
        env = SimpleNamespace(model=object(), data=SimpleNamespace(time=0), viewer=viewer)
        renderer = Mock()
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.full_like(frame1, 180)
        depth = np.full((240, 320), 0.3)
        renderer.render.side_effect = [frame1, depth, frame2, depth]
        with patch("utils.piperx_camera_preview.mujoco.Renderer", return_value=renderer):
            preview = WristCameraPreview(env)
            try:
                self.assertTrue(preview.update())
                env.data.time = 0.02
                self.assertFalse(preview.update())
                viewer.viewport = mujoco.MjrRect(0, 0, 360, 240)
                env.data.time = 0.2
                self.assertTrue(preview.update())
                self.assertEqual(preview.frame_count, 2)
                np.testing.assert_array_equal(preview.last_rgb, frame2)
                rect, image = viewer.set_images.call_args.args[0][0]
                self.assertEqual(image.shape[:2], (rect.height, rect.width))
                self.assertGreaterEqual(rect.left, 0)
                self.assertGreaterEqual(rect.bottom, 0)
                self.assertLessEqual(rect.left + rect.width, 360)
                self.assertLessEqual(rect.bottom + rect.height, 240)
            finally:
                preview.close()
            preview.close()
            renderer.close.assert_called_once()
            viewer.clear_images.assert_called_once()
            self.assertFalse(preview.update(force=True))


if __name__ == "__main__":
    unittest.main()
