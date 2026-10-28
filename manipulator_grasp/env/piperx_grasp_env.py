"""Piper X simulation and Jacobian IK, requiring only NumPy and MuJoCo."""
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from utils.piperx_scene import scene_bodies


SCENE = Path(__file__).resolve().parents[1] / "assets/agilex_piper_x/scene.xml"
# TCP +Z points into the object; fingers open along TCP Y.
TOP_DOWN = np.diag([-1.0, 1.0, -1.0])
# Real-robot baseline recorded on 2026-10-24.  Joint 4 is 89.38 deg on the
# real arm, but the simulated wrist range is +/-89 deg and that pose sits on a
# wrist singularity: the first Cartesian departure has no local IK solution.
# Use 87 deg in simulation (same posture, 2.4 deg inside the real value).
HOME_JOINTS_DEG = np.array([5.51, 32.98, -74.92, 87.0, -8.68, 1.28])


class IKError(RuntimeError):
    pass


class PiperXGraspEnv:
    def __init__(self, *, viewer=False, realtime=True, scene=SCENE, render_size=None,
                 target=None):
        if not Path(scene).is_file() or not Path(scene).with_name("piperx.xml").is_file():
            raise FileNotFoundError("Run python scripts/build_piperx_model.py first")
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        # The viewer allocates its offscreen framebuffer at launch. Configure
        # the vision resolution before it creates that rendering context.
        if render_size is not None:
            width, height = render_size
            if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
                raise ValueError("render_size must contain positive integer width/height")
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 7)])
        self.q_indices = self.model.jnt_qposadr[self.joint_ids]
        self.v_indices = self.model.jnt_dofadr[self.joint_ids]
        self.arm_actuators = np.array([self.model.actuator(f"joint{i}_position").id for i in range(1, 7)])
        self.gripper_actuator = self.model.actuator("gripper_position").id
        self.finger_indices = np.array([self.model.joint(f"gripper_joint{i}").qposadr[0] for i in (1, 2)])
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        self.tcp_id = self.model.site("tcp").id
        self.target_key = ""
        self.target_label = ""
        self.target_body = ""
        self.target_body_id = -1
        self.target_free_joint_name = ""
        self.target_geom_ids = np.empty(0, dtype=int)
        self.set_target(target)
        self.viewer = None
        self._viewer_threads = []
        self.realtime = realtime
        self._next_sync = 0.0
        self._start_wall = time.monotonic()
        self.max_object_height = 0.0
        self.bilateral_contacts = False
        self.reset()
        if viewer:
            from mujoco import viewer as mj_viewer
            threads_before = set(threading.enumerate())
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            # launch_passive starts daemon threads but exposes no join API.
            # Keep the threads started during this launch alive through close
            # so GLFW is not terminated while their render loop is unwinding.
            self._viewer_threads = [thread for thread in threading.enumerate()
                                    if thread not in threads_before]
            self.viewer.cam.lookat[:] = [0.25, 0, 0.95]
            self.viewer.cam.distance = 1.5
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25

    def set_target(self, target):
        """Select any free-body name; None means no target is selected."""
        if target is None:
            self.target_key = ""
            self.target_label = ""
            self.target_body = ""
            self.target_body_id = -1
            self.target_free_joint_name = ""
            self.target_geom_ids = np.empty(0, dtype=int)
            self.max_object_height = 0.0
            self.bilateral_contacts = False
            return
        if hasattr(target, "body"):
            body_name = str(target.body)
            key = str(getattr(target, "key", body_name))
            label = str(getattr(target, "label", body_name))
        else:
            body_name = str(target)
            key = label = body_name
        bodies = scene_bodies(self)
        if body_name not in bodies:
            choices = ", ".join(sorted(bodies))
            raise ValueError(f"Unknown target body {body_name!r}; choose one of: {choices}")
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_name = bodies[body_name]
        geom_start = int(self.model.body_geomadr[body_id])
        geom_count = int(self.model.body_geomnum[body_id])
        self.target_key = key
        self.target_label = label
        self.target_body = body_name
        self.target_body_id = int(body_id)
        self.target_free_joint_name = joint_name
        self.target_geom_ids = np.arange(geom_start, geom_start + geom_count, dtype=int)
        if hasattr(self, "data"):
            self.max_object_height = float(self.object_position[2])
            self.bilateral_contacts = False

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.finger_indices] = [0.05, -0.05]
        mujoco.mj_forward(self.model, self.data)
        # Start from the recorded real-robot baseline, with both jaws open.
        # This is the only place arm qpos is assigned directly; motions below
        # use actuators.
        home = np.clip(np.radians(HOME_JOINTS_DEG),
                       self.limits[:, 0], self.limits[:, 1])
        self.data.qpos[self.q_indices] = home
        self.data.ctrl[self.arm_actuators] = home
        self.data.ctrl[self.gripper_actuator] = 0.1
        mujoco.mj_forward(self.model, self.data)
        self.max_object_height = (float(self.object_position[2])
                                  if self.target_body_id >= 0 else 0.0)
        self.bilateral_contacts = False
        self._start_wall = time.monotonic()
        self._next_sync = 0.0

    @property
    def object_position(self):
        if self.target_body_id < 0:
            raise RuntimeError("No target body selected; call set_target() first")
        return self.data.xipos[self.target_body_id].copy()

    @property
    def tcp_pose(self):
        transform = np.eye(4)
        transform[:3, :3] = self.data.site_xmat[self.tcp_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[self.tcp_id]
        return transform

    @property
    def gripper_width(self):
        return float(self.data.qpos[self.finger_indices[0]] - self.data.qpos[self.finger_indices[1]])

    def get_joint(self):
        return self.data.qpos[self.q_indices].copy()

    def finger_contact_state(self):
        """Current tactile contacts with external bodies; never a latched flag.

        This observes contact pairs, not the target's name or true position.
        Both jaws must touch the same external body, excluding world/robot.
        """
        robot_root = self.model.body_rootid[self.model.body("base_link").id]
        touched = [set(), set()]
        for contact in self.data.contact:
            if contact.dist > 0.001:
                continue
            for finger_geom, other_geom in ((contact.geom1, contact.geom2),
                                            (contact.geom2, contact.geom1)):
                name = self.model.geom(finger_geom).name or ""
                other_root = int(self.model.body_rootid[self.model.geom_bodyid[other_geom]])
                if other_root in (0, robot_root):
                    continue
                for index in (0, 1):
                    if name.startswith(f"gripper_link{index + 1}_"):
                        touched[index].add(other_root)
        return {"left": bool(touched[0]), "right": bool(touched[1]),
                "bilateral": bool(touched[0] & touched[1]), "width": self.gripper_width}

    def solve_ik(self, position, rotation=TOP_DOWN, *, seed=None, position_tolerance=0.0002):
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("Expected a 3D position and 3x3 rotation")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Pose must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
            raise ValueError("Invalid rotation matrix")
        scratch = self.ik_data
        scratch.qpos[:] = self.data.qpos
        q = np.clip(self.get_joint() if seed is None else np.asarray(seed, dtype=float),
                    self.limits[:, 0] + 1e-5, self.limits[:, 1] - 1e-5)
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, rotation.reshape(-1))
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        error_r = np.empty(3)
        current_quat, inverse_quat, difference = np.empty(4), np.empty(4), np.empty(4)
        for _ in range(500):
            scratch.qpos[self.q_indices] = q
            mujoco.mj_forward(self.model, scratch)
            error_p = position - scratch.site_xpos[self.tcp_id]
            mujoco.mju_mat2Quat(current_quat, scratch.site_xmat[self.tcp_id])
            mujoco.mju_negQuat(inverse_quat, current_quat)
            mujoco.mju_mulQuat(difference, target_quat, inverse_quat)
            if difference[0] < 0:
                difference *= -1
            mujoco.mju_quat2Vel(error_r, difference, 1.0)
            if np.linalg.norm(error_p) < position_tolerance and np.linalg.norm(error_r) < 0.002:
                return q.copy()
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.tcp_id)
            jac = np.vstack((jacp[:, self.v_indices], 0.25 * jacr[:, self.v_indices]))
            error = np.concatenate((error_p, 0.25 * error_r))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
            delta *= min(1.0, 0.12 / max(np.max(np.abs(delta)), 1e-12))
            q = np.clip(q + delta, self.limits[:, 0] + 1e-5, self.limits[:, 1] - 1e-5)
        raise IKError(f"Unreachable pose {position.tolist()}: position error={np.linalg.norm(error_p):.4f} m, "
                      f"rotation error={np.linalg.norm(error_r):.4f} rad")

    def step(self):
        if self.viewer is not None and not self.viewer.is_running():
            raise KeyboardInterrupt
        # Feed-forward compensation for arm gravity/Coriolis; the object and
        # fingers remain subject to gravity and all contact forces.
        self.data.qfrc_applied[self.v_indices] = self.data.qfrc_bias[self.v_indices]
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Simulation became non-finite")
        if self.target_body_id >= 0:
            self.max_object_height = max(self.max_object_height, float(self.object_position[2]))
        finger_contacts = set()
        target_geoms = self.target_geom_ids
        for contact in self.data.contact:
            if contact.geom1 in target_geoms or contact.geom2 in target_geoms:
                other = contact.geom2 if contact.geom1 in target_geoms else contact.geom1
                name = self.model.geom(other).name
                for finger in ("gripper_link1", "gripper_link2"):
                    if name.startswith(finger):
                        finger_contacts.add(finger)
        self.bilateral_contacts |= len(finger_contacts) == 2
        if self.viewer is not None and self.data.time >= self._next_sync:
            self.viewer.sync()
            self._next_sync = self.data.time + 1 / 30
            if self.realtime:
                delay = self._start_wall + self.data.time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

    def camera_parameters(self, camera="wrist", width=640, height=480):
        """Return intrinsics and optical-camera-to-world pose for RGB-D data.

        Optical axes are X right, Y down, Z forward. MuJoCo camera axes have
        Y up and look along -Z, so both axes must be flipped for GraspNet.
        """
        camera_id = self.model.camera(camera).id
        focal = height / (2 * np.tan(np.deg2rad(self.model.cam_fovy[camera_id]) / 2))
        transform = np.eye(4)
        transform[:3, :3] = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        transform[:3, 3] = self.data.cam_xpos[camera_id]
        return {"name": camera, "parent_body": self.model.body(self.model.cam_bodyid[camera_id]).name,
                "width": width, "height": height, "fx": focal, "fy": focal,
                "ppx": width / 2, "ppy": height / 2, "scale": 1.0,
                "T_world_camera": transform}

    def camera_rotation_from_tcp(self, camera="wrist"):
        """Return the fixed camera optical rotation expressed in the TCP frame."""
        camera_id = self.model.camera(camera).id
        return self.tcp_pose[:3, :3].T @ self.data.cam_xmat[camera_id].reshape(3, 3)

    def camera_grasp_to_world(self, translation, rotation, camera="wrist", *, T_world_camera=None):
        """Convert using the camera pose at exposure, before the wrist moves.

        Pass a saved T_world_camera for a grasp inferred from an earlier frame.
        The live pose is only appropriate when the camera has not moved.
        """
        camera_grasp = np.eye(4)
        camera_grasp[:3, :3] = np.asarray(rotation)
        camera_grasp[:3, 3] = np.asarray(translation)
        grasp_tcp = np.eye(4)
        grasp_tcp[:3, :3] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        transform = (self.camera_parameters(camera)["T_world_camera"] if T_world_camera is None
                     else np.asarray(T_world_camera, dtype=float))
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Expected a finite 4x4 camera-to-world transform")
        return transform @ camera_grasp @ grasp_tcp

    def wait(self, duration):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive and finite")
        for _ in range(max(1, round(duration / self.model.opt.timestep))):
            self.step()

    def move_joints(self, goal, duration=2.0):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive and finite")
        start = self.get_joint()
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (6,) or not np.isfinite(goal).all():
            raise ValueError("Expected six finite joint angles")
        if np.any(goal < self.limits[:, 0]) or np.any(goal > self.limits[:, 1]):
            raise ValueError("Joint goal exceeds model limits")
        count = max(1, round(duration / self.model.opt.timestep))
        for k in range(1, count + 1):
            t = k / count
            s = 10*t**3 - 15*t**4 + 6*t**5
            self.data.ctrl[self.arm_actuators] = start + s * (goal - start)
            self.step()
        self.wait(0.15)

    def _fk_pose(self, joint):
        """Forward kinematics for `joint` in a scratch MjData; real data untouched."""
        joint = np.asarray(joint, dtype=float)
        if joint.shape != (6,) or not np.isfinite(joint).all():
            raise ValueError("Expected six finite joint angles")
        if np.any(joint < self.limits[:, 0]) or np.any(joint > self.limits[:, 1]):
            raise ValueError("Joint pose exceeds model limits")
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.q_indices] = joint
        mujoco.mj_forward(self.model, scratch)
        transform = np.eye(4)
        transform[:3, :3] = scratch.site_xmat[self.tcp_id].reshape(3, 3)
        transform[:3, 3] = scratch.site_xpos[self.tcp_id]
        return transform

    @staticmethod
    def _slerp(start_rotation, end_rotation, s):
        """Shortest-path spherical interpolation between two rotations."""
        start_quat, end_quat, out = np.empty(4), np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(start_quat, np.asarray(start_rotation, dtype=float).reshape(-1))
        mujoco.mju_mat2Quat(end_quat, np.asarray(end_rotation, dtype=float).reshape(-1))
        dot = float(start_quat @ end_quat)
        if dot < 0:
            end_quat *= -1
            dot = -dot
        angle = np.arccos(np.clip(dot, -1., 1.))
        if angle < 1e-6:
            out[:] = (1 - s) * start_quat + s * end_quat
        else:
            out[:] = (np.sin((1 - s) * angle) * start_quat + np.sin(s * angle) * end_quat) / np.sin(angle)
        out /= np.linalg.norm(out)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, out)
        return rotation.reshape(3, 3)

    def plan_to(self, position, rotation=TOP_DOWN, duration=2.0, *, start_joint=None):
        """Precompute a continuous Cartesian segment without touching the sim.

        With start_joint=None the true current joint configuration is used;
        otherwise FK is evaluated on an independent MjData so cached plans can
        be validated against a previous plan without moving the robot.
        """
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive and finite")
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("Expected a 3D position and 3x3 rotation")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Pose must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
            raise ValueError("Invalid rotation matrix")
        if start_joint is None:
            start_joint = self.get_joint()
        else:
            start_joint = np.asarray(start_joint, dtype=float)
            if start_joint.shape != (6,) or not np.isfinite(start_joint).all():
                raise ValueError("Expected six finite joint angles")
            if np.any(start_joint < self.limits[:, 0]) or np.any(start_joint > self.limits[:, 1]):
                raise ValueError("Joint pose exceeds model limits")
            start_joint = start_joint.copy()
        start_pose = self._fk_pose(start_joint)
        count = max(2, round(duration / 0.04))
        goals = []
        seed = start_joint.copy()
        for k in range(1, count + 1):
            t = k / count
            s = 10*t**3 - 15*t**4 + 6*t**5
            target_position = start_pose[:3, 3] + s * (position - start_pose[:3, 3])
            target_rotation = self._slerp(start_pose[:3, :3], rotation, s)
            seed = self.solve_ik(
                target_position, target_rotation, seed=seed,
                position_tolerance=0.0002 if k == count else 0.0008)
            if np.max(np.abs(seed - (goals[-1] if goals else start_joint))) > 0.25:
                raise IKError("Discontinuous joint branch in Cartesian trajectory")
            self._check_arm_collision(self.ik_data)
            goals.append(seed.copy())
        return {"start_joint": start_joint.copy(), "goals": np.asarray(goals),
                "position": position.copy(), "rotation": rotation.copy(), "duration": float(duration)}

    def _check_arm_collision(self, data):
        robot = self.model.body_rootid[self.model.body("base_link").id]
        for contact in data.contact:
            roots = self.model.body_rootid[self.model.geom_bodyid[[contact.geom1, contact.geom2]]]
            if contact.dist < -0.0001 and ((roots == robot).all() or
                                           (robot in roots and 0 in roots)):
                names = [self.model.geom(i).name for i in (contact.geom1, contact.geom2)]
                raise IKError(f"Arm trajectory collision: {names}")

    def execute_plan(self, plan, *, position_tolerance=0.01, angle_tolerance=0.03):
        """Execute a prevalidated segment only while its start is still current."""
        goals = np.asarray(plan["goals"])
        duration = float(plan["duration"])
        if goals.ndim != 2 or goals.shape[1] != 6 or len(goals) == 0 or not np.isfinite(goals).all():
            raise ValueError("Invalid planned joint trajectory")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive and finite")
        if np.any(goals < self.limits[:, 0]) or np.any(goals > self.limits[:, 1]):
            raise ValueError("Plan exceeds joint limits")
        previous = self.get_joint()
        if np.max(np.abs(previous - plan["start_joint"])) > 0.04:
            raise IKError("Robot moved since planning; discard this stale plan")
        for goal in goals:
            steps = max(1, round(duration / len(goals) / self.model.opt.timestep))
            for k in range(1, steps + 1):
                self.data.ctrl[self.arm_actuators] = previous + (k / steps) * (goal - previous)
                self.step()
            previous = goal
        self.wait(0.2)
        current = self.tcp_pose
        position_error = np.linalg.norm(current[:3, 3] - plan["position"])
        angle = np.arccos(np.clip((np.trace(current[:3, :3].T @ plan["rotation"]) - 1) / 2, -1., 1.))
        if position_error > position_tolerance or angle > angle_tolerance:
            raise RuntimeError(f"TCP tracking error: {position_error:.4f} m, {angle:.4f} rad")

    def move_to(self, position, rotation=TOP_DOWN, duration=2.0, *,
                position_tolerance=0.01, angle_tolerance=0.03):
        """Validate the entire segment, then execute it with continuous attitude."""
        plan = self.plan_to(position, rotation, duration)
        self.execute_plan(plan, position_tolerance=position_tolerance,
                          angle_tolerance=angle_tolerance)
        return plan

    def gripper_control(self, width, duration=1.5):
        """Ramp the total jaw opening smoothly, then give the servo time to settle.

        The position actuator is fast enough that assigning a new width in one
        step produces an almost instantaneous jaw motion.  Interpolating the
        position command makes both opening and closing deliberately slower
        and avoids slamming the fingers into the object.
        """
        width = float(width)
        if not 0 <= width <= 0.1:
            raise ValueError("Gripper width must be between 0 and 0.10 m")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Gripper duration must be positive and finite")
        start = float(self.gripper_width)
        steps = max(1, round(duration / self.model.opt.timestep))
        for k in range(1, steps + 1):
            t = k / steps
            s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
            self.data.ctrl[self.gripper_actuator] = start + s * (width - start)
            self.step()
        self.wait(0.1)

    def render(self, camera="wrist", width=640, height=480):
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=camera)
            depth = renderer.render().copy()
        return {"img": rgb, "depth": depth}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            for thread in self._viewer_threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        raise RuntimeError("MuJoCo viewer thread did not shut down")
            self.viewer = None
            self._viewer_threads = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
