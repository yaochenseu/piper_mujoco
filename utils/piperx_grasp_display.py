"""Show the selected pickup TCP in world coordinates and wrist RGB images."""
import cv2
import mujoco
import numpy as np

from utils.piperx_grasp_geometry import grasp_segments


def append_grasp_geoms(scene, segments, position):
    """Append visual-only geoms, preserving the scene's existing geometry."""
    if scene.ngeom + len(segments) + 1 > scene.maxgeom:
        raise RuntimeError("Insufficient visual scene capacity for grasp marker")
    for segment in segments:
        geom = scene.geoms[scene.ngeom]
        arrow = segment["kind"] != "gripper"
        kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.asarray(segment["rgba"], dtype=np.float32))
        mujoco.mjv_connector(geom, kind, 0.002 if arrow else 0.0015,
                             segment["start"], segment["end"])
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.003),
                       np.asarray(position), np.eye(3).ravel(), np.array([1., 1., 1., 1.], dtype=np.float32))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


class GraspPoseDisplay:
    def __init__(self, env):
        self.env = env
        self.pose = None
        self.segments = []
        self.info = None

    def set_grasp(self, selected, cycle=0):
        pose = np.asarray(selected["world_tcp"], dtype=float)
        segments = grasp_segments(pose, selected["width"])
        self.pose, self.segments = pose.copy(), segments
        rotation = pose[:3, :3]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        if quaternion[0] < 0:
            quaternion *= -1
        cy = np.hypot(rotation[0, 0], rotation[1, 0])
        pitch = np.arctan2(-rotation[2, 0], cy)
        if cy > 1e-8:
            roll, yaw = np.arctan2(rotation[2, 1], rotation[2, 2]), np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            roll, yaw = np.arctan2(-rotation[1, 2], rotation[1, 1]), 0.
        self.info = {"frame": "world", "target": "selected pickup TCP (fixed)", "cycle": cycle,
                     "position_m": pose[:3, 3].tolist(),
                     "quaternion_wxyz": quaternion.tolist(),
                     "rpy_degrees": np.rad2deg([roll, pitch, yaw]).tolist(),
                     "rpy_convention": "R = Rz(yaw) Ry(pitch) Rx(roll)",
                     "width_m": float(selected["width"]), "rank": int(selected["rank"]),
                     "model_candidate_score": float(selected["score"]), "world_tcp": pose.tolist()}
        viewer = self.env.viewer
        if viewer is not None and viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                append_grasp_geoms(viewer.user_scn, segments, pose[:3, 3])
            xyz = " ".join(f"{x:+.4f}" for x in self.info["position_m"])
            rpy = " ".join(f"{x:+.1f}" for x in self.info["rpy_degrees"])
            text = (f"SELECTED PICKUP TCP / WORLD / cycle {cycle}\n"
                    f"XYZ [m]: {xyz}\nRPY [deg, RzRyRx]: {rpy}\n"
                    f"Opening: {selected['width']*1000:.1f} mm / rank {selected['rank']}\n"
                    "X red / Y green (opening) / Z blue (approach)\n"
                    "Cyan: gripper outline / yellow: approach")
            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL,
                              mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, text, ""))
            viewer.sync()

    def annotate_rgb(self, image, camera):
        annotated = image.copy()
        if self.pose is None:
            return annotated
        transform = np.asarray(camera["T_world_camera"])
        height, width = image.shape[:2]
        def project(point):
            optical = transform[:3, :3].T @ (point - transform[:3, 3])
            if optical[2] <= 0.005:
                return None
            pixel = [camera["ppx"] + camera["fx"] * optical[0] / optical[2],
                     camera["ppy"] + camera["fy"] * optical[1] / optical[2]]
            return tuple(np.rint(np.clip(pixel, -100000, 100000)).astype(int))
        for segment in self.segments:
            start, end = project(segment["start"]), project(segment["end"])
            if start is None or end is None:
                continue
            inside, start, end = cv2.clipLine((0, 0, width, height), start, end)
            if not inside:
                continue
            color = tuple(int(c * 255) for c in segment["rgba"][:3])
            if segment["kind"] == "gripper":
                cv2.line(annotated, start, end, color, 2, cv2.LINE_AA)
            else:
                cv2.arrowedLine(annotated, start, end, color, 2, cv2.LINE_AA, tipLength=0.15)
                if segment["kind"] == "axis":
                    cv2.putText(annotated, segment["label"], end, cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, color, 1, cv2.LINE_AA)
        return annotated

    def render_overview(self, width=1280, height=960, closeup=False):
        camera = "overview"
        if closeup and self.pose is not None:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self.pose[:3, 3] + [0, 0, 0.025]
            camera.distance = 0.32
            camera.azimuth = 135
            camera.elevation = -25
        with mujoco.Renderer(self.env.model, width=width, height=height) as renderer:
            renderer.update_scene(self.env.data, camera=camera)
            if self.pose is not None:
                append_grasp_geoms(renderer.scene, self.segments, self.pose[:3, 3])
            image = renderer.render().copy()
        if self.info is not None:
            lines = ["Selected pickup TCP / world frame",
                     "XYZ [m]: " + " ".join(f"{x:+.4f}" for x in self.info["position_m"]),
                     "RPY [deg, RzRyRx]: " + " ".join(f"{x:+.1f}" for x in self.info["rpy_degrees"]),
                     f"Opening: {self.info['width_m']*1000:.1f} mm; candidate rank {self.info['rank']}",
                     "X red / Y green / Z blue / approach yellow / gripper cyan"]
            cv2.rectangle(image, (8, 8), (min(width-1, 880), 24*len(lines)+16), (0, 0, 0), -1)
            for i, line in enumerate(lines):
                cv2.putText(image, line, (16, 30 + 24*i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def clear(self):
        self.pose, self.info, self.segments = None, None, []
        viewer = self.env.viewer
        if viewer is not None and viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
            viewer.clear_texts()
            viewer.sync()

    def close(self):
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
