"""Live RGB and metric depth insets inside the MuJoCo viewer."""
import cv2
import mujoco
import numpy as np


def depth_rgb(depth, near=0.05, far=1.0):
    """Fixed metric Turbo color scale; invalid depth is black."""
    valid = np.isfinite(depth) & (depth > 0)
    level = np.clip((np.where(valid, depth, far) - near) / (far - near), 0, 1)
    colored = cv2.applyColorMap(np.uint8(level * 255), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


class WristCameraPreview:
    def __init__(self, env, fps=10, grasp_display=None):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        self.env = env
        self.grasp_display = grasp_display
        self.period = 1 / fps
        self.next_time = -np.inf
        self.renderer = mujoco.Renderer(env.model, height=240, width=320)
        self.frame_count = 0
        self.last_rgb = self.last_depth = None
        self.last_panel = None

    def update(self, force=False):
        viewer = self.env.viewer
        if self.renderer is None or viewer is None or not viewer.is_running():
            return False
        if not force and self.env.data.time < self.next_time:
            return False
        viewport = viewer.viewport
        if viewport is None or viewport.width < 200 or viewport.height < 120:
            return False
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.env.data, camera="wrist")
        self.last_rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self.last_depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        rgb = self.last_rgb
        if self.grasp_display is not None:
            camera = self.env.camera_parameters("wrist", 320, 240)
            rgb = self.grasp_display.annotate_rgb(rgb, camera)
        panel = np.concatenate((rgb, depth_rgb(self.last_depth)), axis=1)
        for x, label in ((6, "Wrist RGB"), (326, "Depth [m]: 0.05 - 1.00")):
            cv2.rectangle(panel, (x - 4, 0), (x + 306, 25), (0, 0, 0), -1)
            cv2.putText(panel, label, (x, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        width = min(640, int(viewport.width * 0.65),
                    int(viewport.height * 0.4 * 640 / 240))
        height = max(1, round(width * 240 / 640))
        self.last_panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
        rect = mujoco.MjrRect(viewport.left + viewport.width - width - 10,
                             viewport.bottom + viewport.height - height - 10,
                             width, height)
        viewer.set_images([(rect, self.last_panel)])
        self.frame_count += 1
        self.next_time = float(self.env.data.time) + self.period
        return True

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            if self.env.viewer is not None and self.env.viewer.is_running():
                self.env.viewer.clear_images()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
