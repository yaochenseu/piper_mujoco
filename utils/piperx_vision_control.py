"""Visual/tactile pick-place policy, deliberately independent of object truth."""
import numpy as np


class TargetNotFoundError(RuntimeError):
    """No usable instance of the requested target in the current camera view."""

    def __init__(self, message, *, search_yaw_deg=None):
        super().__init__(message)
        self.search_yaw_deg = search_yaw_deg


class NoUsableGraspError(TargetNotFoundError):
    """The target is visible, but this view yields no collision-free grasp."""


APPROACH_HEIGHT = 0.06
MIN_LIFT_HEIGHT = 0.04
LIFT_PROBE_HEIGHT = 0.020
MIN_LIFT_PROBE_HEIGHT = 0.015
# GraspNet translations sit on the visible surface.  A 0.03 m insertion
# reaches the mid-section of common 5-8 cm bottles/cans while the table-top
# clip still prevents the fingertips from going below the table.  Thin or
# elevated objects can override this with --grasp-insertion-depth.
GRASP_INSERTION_DEPTH = 0.03
# Fingertip extent ahead of the pad-centre TCP, measured from the AGX finger
# collision meshes.  Used only to bound insertion against the tabletop.
GRIPPER_TIP_AHEAD_M = 0.013
GRASP_TABLE_CLEARANCE = 0.002
PLACE_CLEARANCE = 0.004
# Same TCP convention as manipulator_grasp.env.piperx_grasp_env.TOP_DOWN.
# Kept here so the pure geometry helpers do not import MuJoCo.
TOP_DOWN = np.diag([-1.0, 1.0, -1.0])
# A parallel-jaw gripper is symmetric under a 180-degree rotation about its
# approach axis (TCP +Z).  Both attitudes describe the same physical grasp.
JAW_FLIP = np.diag([-1.0, -1.0, 1.0])
ANALYTIC_YAW_DEGREES = (0.0, 45.0, 90.0, 135.0)


def inside_reach_sphere(point, center, radius, min_radius=0.0):
    """Return whether ``point`` is inside the arm's spherical reach envelope.

    Piper X reach is a ball around the shoulder (base_link + 0.123 m), with the
    URDF chain limiting its radius to about 0.80 m.  Keep this helper pure so
    the perception, real-env and test layers can share the same convention.
    """
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    if point.shape != (3,) or center.shape != (3,):
        raise ValueError("Expected a 3D point and a 3D sphere centre")
    if radius < 0.0 or min_radius < 0.0:
        raise ValueError("Reach radii must be non-negative")
    distance = float(np.linalg.norm(point - center))
    return bool(min_radius <= distance <= radius)


def yaw_delta_to_center(target_world, camera, *, max_abs_deg=45.0,
                        step_deg=0.5, tolerance_px=None):
    """Return the J1 yaw that projects ``target_world`` near the image centre.

    J1 rotates the whole arm (and the wrist camera) about the base Z axis, so
    the camera transform after a candidate yaw is ``Rz(yaw) @ T_world_camera``.
    Searching that one-dimensional family is much more reliable than using the
    VLM's qualitative "left/right" hint, and it costs no extra perception.
    Only horizontal centring is possible with J1; the vertical position is left
    unchanged.
    """
    target = np.asarray(target_world, dtype=float)
    transform = np.asarray(camera["T_world_camera"], dtype=float)
    if target.shape != (3,) or transform.shape != (4, 4):
        raise ValueError("Expected a 3D target and a 4x4 camera transform")
    if not np.isfinite(target).all() or not np.isfinite(transform).all():
        raise ValueError("Target and camera transform must be finite")
    fx = float(camera["fx"])
    ppx = float(camera["ppx"])
    width = float(camera.get("width", 2.0 * ppx))
    if tolerance_px is None:
        tolerance_px = max(8.0, 0.05 * width)
    if fx <= 0.0 or width <= 0.0:
        raise ValueError("Camera fx and width must be positive")
    if max_abs_deg <= 0.0 or step_deg <= 0.0:
        raise ValueError("max_abs_deg and step_deg must be positive")

    def project_u(delta_deg):
        angle = np.radians(float(delta_deg))
        cosine, sine = np.cos(angle), np.sin(angle)
        rotate = np.eye(4)
        rotate[:3, :3] = [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
        rotated = rotate @ transform
        point = rotated[:3, :3].T @ (target - rotated[:3, 3])
        if point[2] <= 1e-6:
            return None
        return fx * point[0] / point[2] + ppx

    current_u = project_u(0.0)
    if current_u is None:
        return {"centered": False, "solution_found": False,
                "delta_deg": 0.0, "current_u": None, "predicted_u": None,
                "tolerance_px": float(tolerance_px)}
    if abs(current_u - ppx) <= tolerance_px:
        return {"centered": True, "solution_found": True, "delta_deg": 0.0,
                "current_u": float(current_u), "predicted_u": float(current_u),
                "tolerance_px": float(tolerance_px)}

    samples = []
    for delta in np.arange(-max_abs_deg, max_abs_deg + step_deg * 0.5,
                           step_deg):
        u = project_u(float(delta))
        if u is not None:
            samples.append((float(delta), float(u)))
    if not samples:
        return {"centered": False, "delta_deg": 0.0,
                "current_u": float(current_u), "predicted_u": None,
                "tolerance_px": float(tolerance_px)}

    roots = []
    for (left_deg, left_u), (right_deg, right_u) in zip(samples, samples[1:]):
        left_error = left_u - ppx
        right_error = right_u - ppx
        if left_error == 0.0:
            roots.append(left_deg)
        elif left_error * right_error < 0.0:
            span = right_u - left_u
            root = (left_deg if abs(span) < 1e-12 else
                    left_deg + (ppx - left_u) / span * (right_deg - left_deg))
            roots.append(float(np.clip(root, -max_abs_deg, max_abs_deg)))
    if roots:
        delta = min(roots, key=abs)
        return {"centered": False, "solution_found": True,
                "delta_deg": float(delta), "current_u": float(current_u),
                "predicted_u": float(ppx),
                "tolerance_px": float(tolerance_px)}

    delta, predicted = min(samples, key=lambda item: abs(item[1] - ppx))
    return {"centered": False, "solution_found": False,
            "delta_deg": float(delta), "current_u": float(current_u),
            "predicted_u": float(predicted),
            "tolerance_px": float(tolerance_px)}


def rotation_angle_radians(first, second):
    """Return the geodesic angle between two proper rotation matrices."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != (3, 3) or second.shape != (3, 3):
        raise ValueError("Expected two 3x3 rotation matrices")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Rotation matrices must be finite")
    cosine = 0.5 * (np.trace(first.T @ second) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def parallel_jaw_rotation_variants(rotation, reference=None):
    """Return the two equivalent parallel-jaw attitudes, nearest first.

    ``R`` and ``R @ diag([-1, -1, 1])`` differ by a 180-degree rotation about
    the approach axis and therefore represent the same finger placement.  The
    near/far ordering only chooses which IK branch to try first; it must never
    be used to rank the physical grasp itself.
    """
    rotation = np.asarray(rotation, dtype=float)
    reference = rotation if reference is None else np.asarray(reference, dtype=float)
    if rotation.shape != (3, 3) or reference.shape != (3, 3):
        raise ValueError("Expected 3x3 rotation matrices")
    if (not np.isfinite(rotation).all() or not np.isfinite(reference).all() or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
            not np.allclose(reference.T @ reference, np.eye(3), atol=1e-5) or
            np.linalg.det(rotation) < 0 or np.linalg.det(reference) < 0):
        raise ValueError("Invalid rotation matrix")
    variants = [
        {"name": "original", "flipped": False, "rotation": rotation.copy(),
         "angle_rad": rotation_angle_radians(reference, rotation)},
        {"name": "flipped", "flipped": True, "rotation": rotation @ JAW_FLIP,
         "angle_rad": rotation_angle_radians(reference, rotation @ JAW_FLIP)},
    ]
    variants.sort(key=lambda item: (item["angle_rad"], item["name"] != "original"))
    for index, variant in enumerate(variants):
        variant["branch"] = "near" if index == 0 else "far"
    return variants


def observation_pregrasp_pose(env, centroid, *, camera="wrist", camera_height=0.955,
                              clearance=0.12, rotation=TOP_DOWN):
    """Place the wrist camera above the target without choosing a grasp.

    The second perception pass should see the whole object from a safe,
    repeatable viewpoint.  A GraspNet approach pose is a grasp pose, not an
    observation pose; using one here can put the camera at an object tip and
    make the final grasp proposals unreachable.  This helper computes the TCP
    position that centres the named wrist camera over the detected centroid.
    """
    centroid = np.asarray(centroid, dtype=float)
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError("centroid must be a finite 3D point")
    if not np.isfinite(camera_height) or not np.isfinite(clearance):
        raise ValueError("camera_height and clearance must be finite")
    tcp = np.asarray(env.tcp_pose, dtype=float)
    camera_offset = getattr(env, "camera_offset_from_tcp", None)
    if camera_offset is not None:
        offset_tcp = np.asarray(camera_offset(camera), dtype=float)
    else:
        camera_id = env.model.camera(camera).id
        camera_position = np.asarray(env.data.cam_xpos[camera_id], dtype=float)
        offset_tcp = tcp[:3, :3].T @ (camera_position - tcp[:3, 3])
    camera_rotation = getattr(env, "camera_rotation_from_tcp", None)
    if camera_rotation is not None:
        R_tcp_cam = np.asarray(camera_rotation(camera), dtype=float)
    elif hasattr(env, "model"):
        camera_id = env.model.camera(camera).id
        R_tcp_cam = tcp[:3, :3].T @ np.asarray(
            env.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    else:
        R_tcp_cam = np.eye(3)
    desired_rotation = np.asarray(rotation, dtype=float)
    optical_axis = desired_rotation @ R_tcp_cam[:, 2]
    z_camera = max(float(camera_height), float(centroid[2]) + float(clearance))
    # Point the optical axis at the object instead of merely centring the
    # camera body above it.  A wrist camera is usually pitched relative to the
    # TCP; ignoring that pitch pushes the object to the image border.
    if optical_axis[2] < -1e-3:
        distance = (float(centroid[2]) - z_camera) / float(optical_axis[2])
        desired_camera = np.asarray(centroid, dtype=float) - optical_axis * distance
    else:
        desired_camera = np.array([centroid[0], centroid[1], z_camera])
    position = desired_camera - desired_rotation @ offset_tcp
    return position, desired_rotation.copy()


def bounded_grasp_insertion(tcp, requested_depth=GRASP_INSERTION_DEPTH, *,
                            table_top_z=0.740, tip_ahead=GRIPPER_TIP_AHEAD_M,
                            table_clearance=GRASP_TABLE_CLEARANCE):
    """Move a grasp contact frame into the object along its approach axis.

    GraspNet translations often sit on the visible surface.  A small insertion
    puts the pad centre inside the object without changing the robot TCP
    convention.  The requested depth is clipped so the fingertip cannot be
    driven below the tabletop on steep downward approaches.
    """
    tcp = np.asarray(tcp, dtype=float)
    if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
        raise ValueError("Expected a finite 4x4 TCP transform")
    for name, value in (("requested_depth", requested_depth),
                        ("table_top_z", table_top_z), ("tip_ahead", tip_ahead),
                        ("table_clearance", table_clearance)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    approach = tcp[:3, 2]
    if requested_depth == 0:
        applied = 0.0
        limit = None
    elif approach[2] < -1e-6:
        tip_z = tcp[2, 3] + approach[2] * tip_ahead
        limit = max(0.0, (tip_z - table_top_z - table_clearance) / -approach[2])
        applied = min(float(requested_depth), limit)
    else:
        limit = None
        applied = float(requested_depth)
    inserted = tcp.copy()
    inserted[:3, 3] += applied * approach
    return {
        "tcp": inserted,
        "requested_depth_m": float(requested_depth),
        "applied_depth_m": float(applied),
        "table_limited": bool(limit is not None and applied < requested_depth),
        "table_limit_m": None if limit is None else float(limit),
    }


def sample_target_grasp_points(object_mask, roi_mask, workspace_mask, num_points, *, rng=None):
    """Sample GraspNet input only from the segmented target object.

    The scene/workspace cloud remains the source for collision checking.  This
    helper deliberately intersects with object_mask so a rectangular ROI can
    never pull neighbouring objects or the robot into grasp proposal input.
    """
    object_mask = np.asarray(object_mask, dtype=bool)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    workspace_mask = np.asarray(workspace_mask, dtype=bool)
    if object_mask.shape != roi_mask.shape or object_mask.shape != workspace_mask.shape:
        raise ValueError("Object, ROI and workspace masks must have the same shape")
    if not isinstance(num_points, int) or num_points < 1:
        raise ValueError("num_points must be a positive integer")
    target_indices = np.flatnonzero(workspace_mask & object_mask & roi_mask)
    if len(target_indices) < 20:
        raise RuntimeError(f"Segmented target has too few grasp points ({len(target_indices)})")
    rng = np.random.default_rng() if rng is None else rng
    replace = len(target_indices) < num_points
    if replace:
        # Match GraspNet's demo loader: keep every unique target point first,
        # then repeat a random subset only to reach the fixed network input
        # size.  Random sampling with replacement from the whole set would
        # silently discard valid target geometry.
        extra = rng.choice(target_indices, num_points - len(target_indices), replace=True)
        sampled = np.concatenate((target_indices, extra))
    else:
        sampled = rng.choice(target_indices, num_points, replace=False)
    report = {
        "target_points_in_roi": int(len(target_indices)),
        "workspace_points_in_roi": int(np.count_nonzero(workspace_mask & roi_mask)),
        "sampled_unique_points": int(len(np.unique(sampled))),
        "sampled_with_replacement": bool(replace),
    }
    return sampled, report


def top_down_grasp_candidates(object_points, *, table_top_z=0.740,
                              yaw_degrees=ANALYTIC_YAW_DEGREES):
    """Build simple top-down grasps from the segmented target cloud.

    The tabletop objects in this scene are small enough for a parallel-jaw
    gripper to approach vertically.  Using the robust XY bounding-box centre
    avoids the visible-surface bias of a raw depth centroid, especially for
    tall objects such as the soup can.  Several fixed yaw candidates cover
    reachability differences of the wrist while remaining object-agnostic.

    These candidates are intentionally cheap and deterministic.  They are a
    fallback after GraspNet has failed to produce a fully reachable proposal.
    """
    points = np.asarray(object_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("object_points must have shape (N, 3)")
    if not np.isfinite(table_top_z):
        raise ValueError("table_top_z must be finite")
    points = points[np.isfinite(points).all(axis=1)]
    points = points[points[:, 2] > table_top_z + 0.001]
    if len(points) < 20:
        return []

    lower = np.percentile(points[:, :2], 2.0, axis=0)
    upper = np.percentile(points[:, :2], 98.0, axis=0)
    centre_xy = 0.5 * (lower + upper)
    # A 98th-percentile top is easily corrupted by a few percent of depth
    # outliers near the mask edge (e.g. a taller can behind a banana), which
    # pushes the analytic grasp height above the object and makes the support
    # slice empty.  The 95th percentile stays on the object while still
    # representing genuinely tall targets.
    top_z = float(np.percentile(points[:, 2], 95.0))
    grasp_z = max(table_top_z + 0.003, 0.5 * (table_top_z + top_z) + 0.002)

    band = points[np.abs(points[:, 2] - grasp_z) <= 0.020]
    if len(band) < 50:
        band = points[np.abs(points[:, 2] - grasp_z) <= 0.035]
    if len(band) < 10:
        band = points
    band_xy = band[:, :2]

    centred = band_xy - centre_xy
    covariance = np.cov(centred.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minor_axis = eigenvectors[:, int(np.argmin(eigenvalues))]
    pca_yaw = float(np.arctan2(minor_axis[1], minor_axis[0]) - np.pi / 2)

    yaws = []
    for degrees in yaw_degrees:
        yaw = (float(np.deg2rad(degrees)) + np.pi / 2) % np.pi - np.pi / 2
        if all(abs(np.angle(np.exp(1j * (yaw - existing)))) > np.deg2rad(10)
               for existing in yaws):
            yaws.append(yaw)
    for yaw in (pca_yaw, pca_yaw + np.pi / 2):
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if all(abs(np.angle(np.exp(1j * (yaw - existing)))) > np.deg2rad(10)
               for existing in yaws):
            yaws.append(yaw)

    candidates = []
    for index, yaw in enumerate(yaws):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine, 0.0],
                             [sine, cosine, 0.0],
                             [0.0, 0.0, 1.0]]) @ TOP_DOWN
        jaw_axis = rotation[:2, 1]
        projection = (band_xy - centre_xy) @ jaw_axis
        lower_jaw, upper_jaw = np.percentile(projection, [2.0, 98.0])
        width = float(np.clip(upper_jaw - lower_jaw + 0.012, 0.020, 0.095))
        tcp = np.eye(4)
        tcp[:3, :3] = rotation
        tcp[:3, 3] = [centre_xy[0], centre_xy[1], grasp_z]
        candidates.append({
            "source": "analytic_top_down",
            "rank": -1 - index,
            "score": float(1.0 / (index + 1)),
            "width": width,
            "world_tcp": tcp,
            "world_position": tcp[:3, 3].copy(),
            "yaw_deg": float(np.rad2deg(yaw)),
            "pca_yaw_deg": float(np.rad2deg(pca_yaw)),
            "approach_cos": float(rotation[:, 2] @ np.array([0.0, 0.0, -1.0])),
            "object_xy_distance": float(np.linalg.norm(
                points[:, :2] - centre_xy[None, :], axis=1).min()),
            "checks": {"width": bool(0.01 <= width <= 0.095),
                       "approach": True, "center": True, "object": True},
        })
    return candidates


def _plan_pick_place_segments(env, tcp, drop_xy, home_pose, *, reorient_rotation=None):
    """Plan one transport variant from a single continuous joint branch."""
    position, rotation = tcp[:3, 3], tcp[:3, :3]
    above = position + [0, 0, APPROACH_HEIGHT]
    destination = np.array([*drop_xy, above[2]])
    place = np.array([*drop_xy, position[2] + PLACE_CLEARANCE])
    targets = [
        ("approach", above, rotation, 2.0),
        ("descend", position, rotation, 1.5),
        # A short tactile probe rejects grips that look supported in the point
        # cloud but slip out as soon as the object is loaded.  The remaining
        # lift starts from the probe pose and keeps the same continuous branch.
        ("lift_probe", position + [0.0, 0.0, LIFT_PROBE_HEIGHT], rotation, 0.8),
        ("lift", above, rotation, 1.6),
    ]
    transport_rotation = rotation
    if reorient_rotation is not None:
        targets.append(("reorient", above, reorient_rotation, 1.5))
        transport_rotation = reorient_rotation
    # Verify from the transfer pose above the drop point: it is already
    # preflighted, and unlike the pre-grasp observation pose it keeps the drop
    # in view, so a successful placement cannot look like a large drop error.
    targets.extend([
        ("transfer", destination, transport_rotation, 2.0),
        ("place", place, transport_rotation, 1.5),
        ("retreat", place + [0, 0, APPROACH_HEIGHT], transport_rotation, 2.0),
        ("observe", destination, transport_rotation, 2.0),
    ])
    plans, seed = {}, env.get_joint()
    for name, point, attitude, duration in targets:
        try:
            plan = env.plan_to(point, attitude, duration, start_joint=seed)
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        plans[name] = plan
        seed = plan["goals"][-1]
    return plans


def plan_pick_place(env, tcp, drop_xy, home_pose, *, allow_reorient=True):
    """Preflight a pick/place path, retrying transport with an upright pose.

    GraspNet often returns a tilted grasp that is valid at the object but
    cannot maintain that attitude all the way to the drop pose.  Keeping the
    grasp attitude for approach/descend/lift is useful; forcing it on the
    transfer is unnecessarily restrictive.  If the direct path fails, retry
    after rotating to TOP_DOWN at the lift height.
    """
    tcp = np.asarray(tcp, dtype=float)
    if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
        raise ValueError("Expected a finite 4x4 TCP transform")
    try:
        return _plan_pick_place_segments(env, tcp, drop_xy, home_pose,
                                         reorient_rotation=None)
    except RuntimeError as direct_error:
        if not allow_reorient:
            raise
        errors = [f"direct: {direct_error}"]
        for variant in parallel_jaw_rotation_variants(TOP_DOWN, reference=tcp[:3, :3]):
            try:
                return _plan_pick_place_segments(
                    env, tcp, drop_xy, home_pose,
                    reorient_rotation=variant["rotation"])
            except RuntimeError as error:
                errors.append(f"reorient {variant['name']}: {error}")
        raise RuntimeError("All pick/place transport variants failed: " +
                           " | ".join(errors)) from direct_error


def plan_grasp_only(env, tcp):
    """Preflight only the approach/descend/lift-probe/lift segments.

    Used by ``--grasp-only``: the robot stops after the lift is confirmed and
    returns to its baseline while keeping the object in the closed gripper.
    No transfer, place, retreat or placement observation is planned, so a
    valid grasp is not rejected because the drop point is inconvenient.
    """
    tcp = np.asarray(tcp, dtype=float)
    if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
        raise ValueError("Expected a finite 4x4 TCP transform")
    position, rotation = tcp[:3, 3], tcp[:3, :3]
    # Approach from the side of the grasp axis, not always from world +Z.
    # GraspNet can propose diagonal/side grasps for flat objects such as a
    # screwdriver; the pre-grasp must lie opposite its approach axis.  The
    # lift itself is still vertical to clear the table.
    approach = position - rotation[:, 2] * APPROACH_HEIGHT
    lift = position + [0, 0, APPROACH_HEIGHT]
    targets = [
        ("approach", approach, rotation, 2.0),
        ("descend", position, rotation, 1.5),
        ("lift_probe", position + [0.0, 0.0, LIFT_PROBE_HEIGHT], rotation, 0.8),
        ("lift", lift, rotation, 1.6),
    ]
    plans, seed = {}, env.get_joint()
    for name, point, attitude, duration in targets:
        try:
            plan = env.plan_to(point, attitude, duration, start_joint=seed)
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        plans[name] = plan
        seed = plan["goals"][-1]
    return plans


def two_finger_support(object_points, tcp, *, width=None, x_half=0.016,
                       z_half=0.025, min_side_points=None, min_span=0.010,
                       max_center_offset=0.015):
    """Check that the target geometry spans both sides of the jaw opening.

    The check is deliberately performed in the TCP frame: local X spans the
    finger width, local Y is the opening direction, and local Z is the approach
    axis.  A single-sided endpoint grasp has almost no points on one side even
    when its nearest-point distance to the object is small.
    """
    points = np.asarray(object_points, dtype=float)
    tcp = np.asarray(tcp, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("object_points must have shape (N, 3)")
    if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
        raise ValueError("Expected a finite 4x4 TCP transform")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        return {"supported": False, "reason": "too_few_object_points",
                "object_points": int(len(points)), "left_points": 0,
                "right_points": 0, "points_in_contact_box": 0,
                "span_m": 0.0, "center_m": 0.0, "jaw_limit_m": 0.0}
    if min_side_points is None:
        min_side_points = max(8, min(30, int(0.002 * len(points))))
    if width is None:
        jaw_limit = 0.050
        max_span = 0.095
    else:
        # Stay close to the commanded jaw opening.  A large y-band can include
        # a neighbouring object and falsely turn an endpoint grasp into a
        # two-sided one.
        jaw_limit = float(min(0.060, max(0.012, 0.5 * float(width) + 0.004)))
        max_span = float(width) + 0.004
    local = (points - tcp[:3, 3]) @ tcp[:3, :3]
    slice_box = ((np.abs(local[:, 0]) <= x_half) &
                 (np.abs(local[:, 2]) <= z_half))
    slice_points = local[slice_box]
    band = slice_points[np.abs(slice_points[:, 1]) <= jaw_limit]
    left = band[band[:, 1] > 0.003]
    right = band[band[:, 1] < -0.003]
    if len(slice_points) >= 2:
        low, high = np.percentile(slice_points[:, 1], [2.0, 98.0])
        span = float(high - low)
        center = float(0.5 * (high + low))
    else:
        span = center = 0.0
    reasons = []
    if len(left) < min_side_points:
        reasons.append("left_side_sparse")
    if len(right) < min_side_points:
        reasons.append("right_side_sparse")
    if span < min_span:
        reasons.append("object_too_narrow")
    if span > max_span:
        reasons.append("object_too_wide")
    if abs(center) > max_center_offset:
        reasons.append("grasp_center_offset")
    return {
        "supported": not reasons,
        "reason": ",".join(reasons) if reasons else None,
        "object_points": int(len(points)),
        "slice_points": int(len(slice_points)),
        "points_in_contact_box": int(len(band)),
        "left_points": int(len(left)),
        "right_points": int(len(right)),
        "min_side_points": int(min_side_points),
        "span_m": span,
        "span_ratio": float(span / float(width)) if width else None,
        "center_m": center,
        "max_span_m": float(max_span),
        "jaw_limit_m": float(jaw_limit),
    }


def held_object(env, samples=6):
    """Require sustained current two-jaw contact and nonempty jaw opening."""
    readings = []
    for _ in range(samples):
        env.wait(0.02)
        readings.append(env.finger_contact_state())
    fraction = sum(x["bilateral"] and 0.008 < x["width"] < 0.095 for x in readings) / samples
    return {"held": bool(fraction >= 0.8 and readings[-1]["bilateral"]), "contact_fraction": fraction,
            "width": readings[-1]["width"], "current_bilateral": readings[-1]["bilateral"]}


def recover(env, home_pose, reference_z):
    """Set down a held object, open, retreat, and return for a fresh image."""
    current = env.tcp_pose.copy()
    if env.finger_contact_state()["bilateral"] and current[2, 3] > reference_z + 0.025:
        release = current[:3, 3].copy()
        release[2] = reference_z + 0.004
        env.move_to(release, current[:3, :3], duration=1.5)
    env.gripper_control(0.1)
    current = env.tcp_pose.copy()
    retreat = current[:3, 3].copy()
    retreat[2] = max(current[2, 3], min(home_pose[2, 3], reference_z + APPROACH_HEIGHT))
    # Preflight both recovery segments before moving away from this state.
    first = env.plan_to(retreat, current[:3, :3])
    second = env.plan_to(home_pose[:3, 3], home_pose[:3, :3], start_joint=first["goals"][-1])
    env.execute_plan(first)
    env.execute_plan(second)
    env.wait(0.3)


def run_closed_loop(env, observe, propose, *, drop_xy, max_attempts=3, log=print,
                    release_first=False, two_stage=False, propose_pregrasp=None,
                    propose_final=None, support_check=False, max_support_trials=12,
                    reobserve_final=None, grasp_only=False, retry=None):
    """Observe -> preflight -> execute -> observe, with fresh data on retries.

    observe(cycle, phase) returns an RGB-D observation with a world centroid;
    propose(observation, cycle, home_pose) returns a selected grasp and plans.
    With two_stage=True, propose_pregrasp selects and approaches a coarse
    pre-grasp pose, then a fresh pre-grasp observation and propose_final select
    the final grasp.  support_check is evaluated once at that final pre-grasp,
    immediately before descend and jaw closure.  A short post-close tactile
    lift probe is always checked before committing to the full lift.  If a
    candidate fails either check, the gripper opens, retreats to the final
    pre-grasp pose and the next candidate is requested.  reobserve_final, when
    provided, supplies a fresh RGB-D/final-grasp observation for that retry.
    retry(env, home_pose, reference_z, record), when provided, replaces the
    default inter-cycle recovery: it should make the arm safe, move to the next
    view, and return True; returning False stops the loop.  All old images,
    grasps and plans are discarded on every retry.
    With grasp_only=True, the loop stops as soon as the full lift is confirmed;
    the caller then returns to its baseline while the gripper stays closed.
    This function never reads a simulator object pose or a historical flag.
    """
    if two_stage and (propose_pregrasp is None or propose_final is None):
        raise ValueError("two_stage requires propose_pregrasp and propose_final")
    if max_support_trials < 1:
        raise ValueError("max_support_trials must be positive")
    home_pose = env.tcp_pose.copy()
    attempts, selected = [], {}
    for cycle in range(max_attempts):
        record = {"cycle": cycle, "phase": "perception"}
        reference_z = 0.78
        try:
            env.gripper_control(0.1)
            observation = observe(cycle, "pickup")
            reference_z = float(observation["centroid"][2])
            record["perception_id"] = observation["id"]
            record["observed_centroid"] = np.asarray(observation["centroid"]).tolist()
            record["phase"] = "planning"
            pregrasp_pose = None
            if two_stage:
                pregrasp = propose_pregrasp(observation, cycle, home_pose)
                pregrasp_plans = pregrasp["plans"]
                if "approach" not in pregrasp_plans:
                    raise RuntimeError("Pre-grasp proposal has no approach plan")
                record.update(
                    pregrasp_rank=pregrasp.get("rank"),
                    pregrasp_score=pregrasp.get("score"),
                    pregrasp_jaw_variant=pregrasp.get("jaw_variant"),
                    pregrasp_jaw_branch=pregrasp.get("jaw_branch"))
                record["phase"] = "pregrasp"
                env.execute_plan(pregrasp_plans["approach"])
                pregrasp_pose = np.asarray(env.tcp_pose).copy()
                record["pregrasp_pose"] = pregrasp_pose.tolist()
                observation = observe(cycle, "pregrasp")
                record["pregrasp_perception_id"] = observation["id"]
            excluded = set()
            support_checks = []

            def retreat_after_reject(rejected, reason, retreat_key, *, open_gripper):
                if open_gripper:
                    env.gripper_control(0.1)
                retreat = {
                    "rank": rejected["rank"],
                    "jaw_variant": rejected.get("jaw_variant"),
                    "reason": reason,
                    "from_pose": np.asarray(env.tcp_pose).tolist(),
                }
                try:
                    retreat_plan = env.plan_to(
                        pregrasp_pose[:3, 3], pregrasp_pose[:3, :3],
                        duration=2.0)
                    env.execute_plan(retreat_plan)
                except RuntimeError as retreat_error:
                    retreat["status"] = "failed"
                    retreat["error"] = str(retreat_error)
                    record.setdefault(retreat_key, []).append(retreat)
                    raise RuntimeError(
                        f"Candidate retreat failed for rank {rejected['rank']}: "
                        f"{retreat_error}") from retreat_error
                retreat["status"] = "returned"
                retreat["to_pose"] = np.asarray(env.tcp_pose).tolist()
                record.setdefault(retreat_key, []).append(retreat)
                log(f"Cycle {cycle}: returned to pre-grasp pose after "
                    f"rejecting rank {rejected['rank']} ({reason})")
                if reobserve_final is not None:
                    fresh = reobserve_final(cycle)
                    record["post_retreat_perception_id"] = fresh["id"]
                    return fresh
                return observation

            while True:
                if two_stage:
                    chosen = propose_final(observation, cycle, home_pose, exclude=excluded)
                else:
                    chosen = propose(observation, cycle, home_pose)
                selected = {key: value for key, value in chosen.items() if key != "plans"}
                plans = chosen["plans"]
                record.update(rank=chosen["rank"], score=chosen["score"],
                              jaw_variant=chosen.get("jaw_variant"),
                              jaw_branch=chosen.get("jaw_branch"),
                              jaw_reference_angle_rad=chosen.get("jaw_reference_angle_rad"),
                              grasp_insertion=chosen.get("grasp_insertion"),
                              reorient_required=chosen.get(
                                  "reorient_required", "reorient" in plans))
                record["phase"] = "grasp"
                env.execute_plan(plans["approach"])
                if support_check:
                    support = two_finger_support(
                        observation["object_points"],
                        np.asarray(chosen["world_tcp"], dtype=float),
                        width=chosen.get("width"))
                    support["rank"] = chosen["rank"]
                    support["jaw_variant"] = chosen.get("jaw_variant")
                    support_checks.append(support)
                    record["support_checks"] = support_checks
                    if not support["supported"]:
                        log(f"Cycle {cycle}: two-finger support failed for "
                            f"rank {chosen['rank']} ({support['reason']})")
                        if not two_stage:
                            raise RuntimeError(
                                f"Two-finger support check failed: {support['reason']}")
                        excluded.add(chosen["rank"])
                        observation = retreat_after_reject(
                            chosen, support["reason"], "support_retreats",
                            open_gripper=False)
                        if len(excluded) >= max_support_trials:
                            raise RuntimeError(
                                "Two-finger support check rejected every available candidate")
                        continue
                env.execute_plan(plans["descend"])
                env.gripper_control(0.0)
                if release_first and cycle == 0 and "injected_fault" not in record:
                    log("Fault injection: release first grasp to exercise recovery")
                    env.gripper_control(0.1)
                    record["injected_fault"] = "release_first_grasp"
                closed_z = float(env.tcp_pose[2, 3])
                env.execute_plan(plans["lift_probe"])
                record["probe_lift_m"] = float(env.tcp_pose[2, 3] - closed_z)
                record["prelift_sensor"] = held_object(env)
                prelift = {
                    "rank": chosen["rank"],
                    "jaw_variant": chosen.get("jaw_variant"),
                    "rise_m": record["probe_lift_m"],
                    **record["prelift_sensor"],
                }
                record.setdefault("prelift_checks", []).append(prelift)
                if (record["probe_lift_m"] < MIN_LIFT_PROBE_HEIGHT or
                        not record["prelift_sensor"]["held"]):
                    if not two_stage:
                        raise RuntimeError("Post-close lift probe lost the object")
                    record["phase"] = "prelift"
                    excluded.add(chosen["rank"])
                    observation = retreat_after_reject(
                        chosen, "lift_probe_lost", "prelift_retreats",
                        open_gripper=True)
                    if len(excluded) >= max_support_trials:
                        raise RuntimeError(
                            "Post-close lift probe rejected every available candidate")
                    continue
                break
            env.execute_plan(plans["lift"])
            record["tcp_lift_m"] = float(env.tcp_pose[2, 3] - closed_z)
            record["lift_sensor"] = held_object(env)
            if record["tcp_lift_m"] < MIN_LIFT_HEIGHT or not record["lift_sensor"]["held"]:
                raise RuntimeError("Lift not confirmed by current tactile contacts/jaw opening")
            log(f"Lift confirmed by tactile contact, TCP rise={record['tcp_lift_m']:.3f} m")
            if grasp_only:
                record.update(result="success", phase="complete",
                              grasp_only=True)
                attempts.append(record)
                return {
                    "policy_success": True,
                    "attempts": attempts,
                    "selected": selected,
                    "recovery_count": sum(
                        x.get("recovered", False) for x in attempts),
                }
            if "reorient" in plans:
                record["phase"] = "reorient"
                env.execute_plan(plans["reorient"])
                record["reorient_sensor"] = held_object(env)
                if not record["reorient_sensor"]["held"]:
                    raise RuntimeError("Object contact lost during reorientation")
            record["phase"] = "transfer"
            env.execute_plan(plans["transfer"])
            record["transfer_sensor"] = held_object(env)
            if not record["transfer_sensor"]["held"]:
                raise RuntimeError("Object contact lost during transfer")
            record["phase"] = "placement"
            # The commanded release point is intentionally above the table;
            # a few millimetres of loaded-servo sag is harmless here.
            env.execute_plan(plans["place"], position_tolerance=0.025,
                             angle_tolerance=0.06)
            env.gripper_control(0.1)
            env.execute_plan(plans["retreat"])
            env.execute_plan(plans["observe"])
            env.wait(0.3)
            placed = observe(cycle, "placement")
            centroid = np.asarray(placed["centroid"])
            record["placement_perception_id"] = placed["id"]
            record["placed_centroid"] = centroid.tolist()
            record["observed_placement_error_m"] = float(np.linalg.norm(centroid[:2] - drop_xy))
            record["observed_height_error_m"] = float(abs(centroid[2] - reference_z))
            if record["observed_placement_error_m"] > 0.04 or record["observed_height_error_m"] > 0.02:
                raise RuntimeError("Fresh RGB-D observation did not confirm placement")
            record.update(result="success", phase="complete")
            attempts.append(record)
            return {"policy_success": True, "attempts": attempts, "selected": selected,
                    "recovery_count": sum(x.get("recovered", False) for x in attempts)}
        except RuntimeError as error:
            record.update(result="failed", error=str(error),
                          error_type=type(error).__name__,
                          search_yaw_deg=getattr(error, "search_yaw_deg", None))
            log(f"Cycle {cycle}: {record['phase']} failed: {error}")
            attempts.append(record)
            if cycle + 1 == max_attempts:
                break
            if retry is not None:
                try:
                    if retry(env, home_pose, reference_z, record):
                        record["recovered"] = True
                        log("Search retry complete; all old images, grasps and plans discarded")
                        continue
                except Exception as retry_error:
                    record["retry_error"] = str(retry_error)
                    log(f"Search retry stopped: {retry_error}")
                    break
                break
            try:
                recover(env, home_pose, reference_z)
                record["recovered"] = True
                log("Recovery complete; all old images, grasps and plans discarded")
            except RuntimeError as recovery_error:
                record["recovery_error"] = str(recovery_error)
                log(f"Recovery stopped: {recovery_error}")
                break
    return {"policy_success": False, "attempts": attempts, "selected": selected,
            "recovery_count": sum(x.get("recovered", False) for x in attempts)}
