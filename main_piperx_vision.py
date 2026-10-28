"""Vision-driven Piper X pick/place in MuJoCo.

GroundingDINO locates the text prompt, SAM segments it, and GraspNet proposes
six-DoF grasps from the RGB-D cloud.  Execution uses contact physics only; the
object's true pose is read for final acceptance and never used to form a grasp.
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.path_setup import setup_graspnet_path  # noqa: E402
from utils.config import load_config  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

from utils.piperx_scene import (  # noqa: E402
    TABLE_TOP_Z,
    resolve_scene_body,
    scene_bodies,
)
from utils.piperx_vlm import (  # noqa: E402
    DEFAULT_DROP_XY,
    build_semantic_planner,
    drop_xy_for_plan,
    semantic_match_score,
)
from utils.piperx_vision_control import (  # noqa: E402
    GRASP_INSERTION_DEPTH,
    NoUsableGraspError,
    TargetNotFoundError,
    bounded_grasp_insertion,
    inside_reach_sphere,
    observation_pregrasp_pose,
    recover,
    yaw_delta_to_center,
)

setup_graspnet_path()

RENDER_WIDTH = 1280
RENDER_HEIGHT = 960
VISION_CAMERA = "wrist"
TABLE_X = (-0.3, 0.9)
TABLE_Y = (-0.45, 0.45)
TABLE_Z = (0.739, 0.90)
OBJECT_Z = (0.742, 0.90)
GRASP_Z = (0.742, 0.88)
NUM_POINT = 20000
VOXEL_SIZE = 0.01
APPROACH_DIST = 0.05
COLLISION_THRESH = 0.01
# TCP +Z is the approach axis. A top-down grasp points it along -Z world, so
# dot(R_tcp[:, 2], (0, 0, -1)) is close to +1 (the angle against +Z is ~pi).
# Full pick/place keeps this strict top-down requirement.  --grasp-only can
# also use diagonal/side GraspNet proposals (e.g. a screwdriver lying flat) as
# long as the pre-grasp point stays above the table and inside the workspace.
APPROACH_COS = 0.8
GRASP_ONLY_APPROACH_COS = 0.2
PREGRASP_TABLE_CLEARANCE = 0.005
OBJECT_XY_DIST = 0.04
MIN_WIDTH = 0.01
MAX_WIDTH = 0.095
BODY_MATCH_MAX_DISTANCE = 0.06


class Logger:
    def __init__(self, path):
        self.path = path
        self.path.write_text("")

    def __call__(self, message=""):
        print(message, flush=True)
        with self.path.open("a") as stream:
            stream.write(str(message) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",
                        help="GroundingDINO phrase, or a scene body name when placing with --obj-x/--obj-y")
    parser.add_argument("--instruction", help="Natural-language pick/place instruction")
    parser.add_argument("--text", help="Override the GroundingDINO target phrase")
    parser.add_argument("--semantic-backend", choices=("auto", "rules", "openai"), default="openai",
                        help="Instruction parser; openai requires PIPERX_VLM_BASE_URL/MODEL, "
                             "auto falls back to rules on VLM failure")
    parser.add_argument("--vlm-base-url", help="OpenAI-compatible VLM base URL")
    parser.add_argument("--vlm-model", help="Multimodal model name served by the VLM endpoint")
    parser.add_argument("--vlm-timeout", type=float, default=60.0)
    parser.add_argument("--vlm-max-tokens", type=int, default=2048,
                        help="Output budget; reasoning models need room before the JSON answer")
    parser.add_argument("--no-camera-preview", action="store_true",
                        help="Hide live wrist RGB/depth insets in the viewer")
    parser.add_argument("--grasp-preview-seconds", type=float, default=2.0,
                        help="Pause before each selected grasp for inspection (GUI only)")
    parser.add_argument("--headless", action="store_true", help="Run physics without opening a viewer")
    parser.add_argument("--fast", action="store_true", help="Do not synchronize viewer to real time")
    parser.add_argument("--hold", action="store_true", help="Keep viewer open after the grasp")
    parser.add_argument("--real", action="store_true",
                        help="Use the real Piper X ROS 2 bridge and D435i")
    parser.add_argument("--real-config", default="real_piperx.yaml",
                        help="Real robot bridge/camera configuration")
    parser.add_argument("--execute", action="store_true",
                        help="Send motion commands to the real robot; default is dry-run")
    parser.add_argument("--open-gate", action="store_true",
                        help="Open /control_enable when --execute is used")
    parser.add_argument("--perception-only", action="store_true",
                        help="Run one RGB-D/detection/segmentation pass without motion")
    parser.add_argument("--single-stage", action="store_true",
                        help="Skip the pregrasp observation move; generate one grasp "
                             "from the initial RGB-D and move to it directly")
    parser.add_argument("--grasp-only", action="store_true",
                        help="Stop after the lift is confirmed and return to the "
                             "captured baseline holding the object; never place it")
    parser.add_argument("--search-angles-deg", type=float, nargs="*",
                        default=[20.0, -20.0, 30.0, -30.0,
                                 40.0, -40.0, 60.0, -60.0,
                                 80.0, -80.0, 120.0, -120.0],
                        help="Base-joint yaw offsets tried in order only when "
                             "the target cannot be found in the current view; "
                             "pass no values or use --no-search to disable")
    parser.add_argument("--no-search", action="store_true",
                        help="Do not move the arm to retry from another J1 view")
    parser.add_argument("--center-target", action="store_true",
                        help="Experimental: rotate J1 to put the detected target "
                             "in the image centre before generating the grasp "
                             "(real --execute only; off by default)")
    parser.add_argument("--no-center-target", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--side-grasps", action="store_true",
                        help="Compatibility flag; diagonal/side GraspNet "
                             "grasps are now tried automatically only after "
                             "all top-down candidates fail")
    parser.add_argument("--obj-x", type=float, help="Initial target x (set once before perception)")
    parser.add_argument("--obj-y", type=float, help="Initial target y (set once before perception)")
    parser.add_argument("--obj-yaw", type=float, default=0.0, help="Initial target yaw in radians")
    parser.add_argument("--drop-x", type=float)
    parser.add_argument("--drop-y", type=float)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-candidates", type=int, default=12,
                        help="Maximum grasp candidates to preflight through the full trajectory")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Minimum fresh perception/execution cycles; "
                             "search adds one cycle per search angle")
    parser.add_argument("--test-release-first", action="store_true",
                        help="Fault test: open the first grasp to verify recovery and fresh perception")
    parser.add_argument("--grasp-roi", type=float, default=0.10,
                        help="Half-size of the target-mask crop around the detected object (m)")
    parser.add_argument("--grasp-insertion-depth", type=float, default=GRASP_INSERTION_DEPTH,
                        help="Inward offset along the GraspNet approach axis (m); clipped above the table")
    parser.add_argument("--diagnostics-dir", type=Path,
                        default=Path("docs/vision_runs") / time.strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict"):
        return jsonable(value.as_dict())
    return value


def write_json(path, payload):
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")


def save_bgr(path, rgb):
    import cv2
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_depth(path, depth):
    import cv2
    valid = depth[depth > 0]
    lo, hi = (float(np.percentile(valid, 1)), float(np.percentile(valid, 99))) if valid.size else (0.0, 1.0)
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    colored = cv2.applyColorMap(((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), colored)


def world_point(u, v, depth, camera):
    z = float(depth[v, u])
    rotation = camera["T_world_camera"][:3, :3]
    translation = camera["T_world_camera"][:3, 3]
    point = np.array([(u - camera["ppx"]) * z / camera["fx"],
                      (v - camera["ppy"]) * z / camera["fy"], z])
    return rotation @ point + translation


def workspace_limits(env=None):
    reach_center = getattr(env, "reach_center", None)
    return {
        "table_x": tuple(getattr(env, "table_x", TABLE_X)),
        "table_y": tuple(getattr(env, "table_y", TABLE_Y)),
        "table_z": tuple(getattr(env, "table_z", TABLE_Z)),
        "object_z": tuple(getattr(env, "object_z", OBJECT_Z)),
        "grasp_z": tuple(getattr(env, "grasp_z", GRASP_Z)),
        "table_top_z": float(getattr(env, "table_top_z", TABLE_TOP_Z)),
        # None for the simulator: its tabletop is far above base_link, so the
        # real-robot URDF sphere would not apply without rebasing it.  The real
        # adapter supplies these values from configs/real_piperx.yaml.
        "reach_center": None if reach_center is None else tuple(reach_center),
        "reach_radius_m": getattr(env, "reach_radius_m", None),
        "reach_min_radius_m": getattr(env, "reach_min_radius_m", 0.0),
    }


def in_table_xy(point, workspace=None):
    workspace = workspace or workspace_limits()
    x_range = workspace["table_x"]
    y_range = workspace["table_y"]
    return x_range[0] <= point[0] <= x_range[1] and y_range[0] <= point[1] <= y_range[1]


def in_workspace_xy(point, workspace=None):
    """Check the table bounds and the URDF-derived spherical reach envelope."""
    workspace = workspace or workspace_limits()
    if not in_table_xy(point, workspace):
        return False
    center = workspace.get("reach_center")
    radius = workspace.get("reach_radius_m")
    if center is None or radius is None:
        return True
    return inside_reach_sphere(
        point, center, float(radius),
        float(workspace.get("reach_min_radius_m", 0.0)))


def select_detection(detections, depth, camera, log, *, labels=None,
                     target_phrase=None, spatial_reference="none",
                     workspace=None):
    """Select the semantically matching on-table box, then apply spatial reference."""
    workspace = workspace or workspace_limits()
    if len(detections) == 0:
        return None, None
    height, width = depth.shape
    candidates = []
    for index, box in enumerate(detections.xyxy):
        label = labels[index] if labels is not None and index < len(labels) else ""
        u = int(np.clip(round((box[0] + box[2]) / 2), 0, width - 1))
        v = int(np.clip(round((box[1] + box[3]) / 2), 0, height - 1))
        if depth[v, u] <= 0:
            continue
        point = world_point(u, v, depth, camera)
        if not (in_workspace_xy(point, workspace) and
                workspace["table_z"][0] <= point[2] <= workspace["table_z"][1]):
            log(f"  skip box {index} {label!r} conf={float(detections.confidence[index]):.3f}: "
                f"centre world={point.round(4).tolist()} outside workspace/reach")
            continue
        candidates.append({
            "index": index,
            "point": point,
            "confidence": float(detections.confidence[index]),
            "match": semantic_match_score(label, target_phrase),
            "label": label,
        })
    if candidates:
        best_match = max(candidate["match"] for candidate in candidates)
        eligible = ([candidate for candidate in candidates if candidate["match"] == best_match]
                    if best_match > 0 else candidates)
        if spatial_reference == "leftmost":
            chosen = max(eligible, key=lambda candidate: (candidate["point"][1], candidate["confidence"]))
        elif spatial_reference == "rightmost":
            chosen = min(eligible, key=lambda candidate: (candidate["point"][1], -candidate["confidence"]))
        elif spatial_reference == "closest":
            camera_position = np.asarray(camera["T_world_camera"])[:3, 3]
            chosen = min(eligible, key=lambda candidate: np.linalg.norm(
                candidate["point"] - camera_position))
        elif spatial_reference == "farthest":
            camera_position = np.asarray(camera["T_world_camera"])[:3, 3]
            chosen = max(eligible, key=lambda candidate: np.linalg.norm(
                candidate["point"] - camera_position))
        else:
            chosen = max(eligible, key=lambda candidate: (candidate["match"], candidate["confidence"]))
        log(f"  selected box {chosen['index']} {chosen['label']!r} "
            f"conf={chosen['confidence']:.3f} match={chosen['match']} "
            f"centre world={chosen['point'].round(4).tolist()}")
        return int(chosen["index"]), chosen["point"]

    ordered = np.argsort(-detections.confidence)
    if target_phrase:
        ordered = sorted(ordered, key=lambda index: (
            -semantic_match_score(labels[index] if labels is not None and index < len(labels) else "",
                                  target_phrase),
            -float(detections.confidence[index])))
    index = int(ordered[0])
    log(f"  no box centre on the table; falling back to highest-ranked box {index}")
    return index, None


def detect_with_dino(image_bgr, caption, weights_dir, diagnostics, log):
    from groundingdino.util.inference import Model
    config_path = weights_dir / "groundingdino_swint_ogc.py"
    checkpoint = weights_dir / "groundingdino_swint_ogc.pth"
    started = time.time()
    model = Model(model_config_path=str(config_path), model_checkpoint_path=str(checkpoint),
                  device="cuda" if _cuda_available() else "cpu")
    loaded = time.time()
    attempts = []
    selected = None
    for box_threshold, text_threshold in ((0.25, 0.25), (0.18, 0.18), (0.12, 0.12)):
        detections, labels = model.predict_with_caption(
            image=image_bgr, caption=caption,
            box_threshold=box_threshold, text_threshold=text_threshold)
        attempts.append({
            "box_threshold": box_threshold, "text_threshold": text_threshold,
            "detections": [{"xyxy": box.tolist(), "confidence": float(conf), "label": label}
                           for box, conf, label in zip(detections.xyxy, detections.confidence, labels)],
        })
        log(f"DINO t={box_threshold:.2f}/{text_threshold:.2f}: {len(detections)} detection(s)")
        if len(detections):
            selected = (detections, labels)
            break
    infer_time = time.time() - loaded
    result = {"caption": caption, "attempts": attempts, "load_s": loaded - started,
              "infer_s": infer_time}
    if selected is None:
        write_json(diagnostics / "detections.json", result)
        return None, result, model
    return selected, result, model


def segment_with_sam(image_bgr, box, sam_path, log):
    import cv2
    import torch
    from ultralytics import SAM
    started = time.time()
    model = SAM(str(sam_path))
    loaded = time.time()
    predictions = model.predict(source=image_bgr, bboxes=[box.tolist()], device=0,
                                verbose=False, save=False)
    infer_time = time.time() - loaded
    masks = predictions[0].masks
    if masks is None or len(masks.data) == 0:
        raise RuntimeError("SAM returned no mask for the detected box")
    data = masks.data.cpu().numpy()
    areas = data.reshape(len(data), -1).sum(axis=1)
    mask = data[int(np.argmax(areas))] > 0.5
    height, width = image_bgr.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == best
    stats_out = {"load_s": loaded - started, "infer_s": infer_time,
                 "mask_shape": list(data.shape), "area_px": int(mask.sum())}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"SAM mask: {int(mask.sum())} px")
    return mask, stats_out


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def perceive(env, args, weights_dir, diagnostics, log, *, predict_grasps=True,
             verify=None):
    """Fresh RGB-D, text detection and segmentation; no simulator object pose."""
    import torch
    # The editable graspnetAPI install is shadowed once graspnet's model
    # directory puts graspnet-baseline on sys.path, so import the API first.
    from graspnetAPI import GraspGroup
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
    from collision_detector import ModelFreeCollisionDetector
    diagnostics.mkdir(parents=True, exist_ok=True)
    timings = {}
    # 1. Smoke render, also the RGB-D frame used for perception.
    started = time.time()
    frame = env.render(camera=VISION_CAMERA, width=RENDER_WIDTH, height=RENDER_HEIGHT)
    timings["render_s"] = time.time() - started
    rgb = frame["img"]
    depth = frame["depth"].astype(np.float32)
    height, width = rgb.shape[:2]
    camera = env.camera_parameters(VISION_CAMERA, width, height)
    if getattr(env, "is_real", False):
        try:
            args.last_view_j1_rad = float(env.get_joint()[0])
        except Exception:
            pass
    limits = workspace_limits(env)
    write_json(diagnostics / "camera.json", camera)
    save_bgr(diagnostics / "rgb.png", rgb)
    save_depth(diagnostics / "depth.png", depth)
    np.save(diagnostics / "depth.npy", depth)
    log(f"RGB-D {rgb.shape} depth range {float(depth.min()):.3f}..{float(depth.max()):.3f}")

    import cv2
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 2. GroundingDINO.
    started = time.time()
    detection, detections_report, dino_model = detect_with_dino(
        image_bgr, args.text, weights_dir, diagnostics, log)
    timings["dino_s"] = time.time() - started
    if detection is None:
        if verify is not None:
            started = time.time()
            verify(image_bgr, None)
            timings["vlm_verify_s"] = time.time() - started
        raise TargetNotFoundError(
            f"GroundingDINO found nothing for caption {args.text!r}")
    detections, labels = detection
    semantic_plan = getattr(args, "semantic_plan", None)
    target_phrase = (semantic_plan.target_phrase if semantic_plan is not None else args.text)
    target_key = env.target_key or None
    spatial_reference = (semantic_plan.spatial_reference if semantic_plan is not None else "none")
    box_index, box_world = select_detection(
        detections, depth, camera, log, labels=labels, target_phrase=target_phrase,
        spatial_reference=spatial_reference, workspace=limits)
    detections_report["selected_index"] = box_index
    detections_report["selected_box"] = detections.xyxy[box_index].tolist()
    detections_report["selected_confidence"] = float(detections.confidence[box_index])
    detections_report["selected_label"] = labels[box_index]
    write_json(diagnostics / "detections.json", detections_report)
    detection_info = {
        "box": detections.xyxy[box_index].tolist(),
        "label": labels[box_index],
        "confidence": float(detections.confidence[box_index]),
    }
    vlm_verification = None
    if verify is not None:
        started = time.time()
        vlm_verification = verify(image_bgr, detection_info)
        timings["vlm_verify_s"] = time.time() - started
        if vlm_verification is not None:
            write_json(diagnostics / "vlm_verification.json",
                       vlm_verification.as_dict())
    visualization = image_bgr.copy()
    for index, (box, confidence) in enumerate(zip(detections.xyxy, detections.confidence)):
        color = (0, 255, 0) if index == box_index else (0, 165, 255)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
        cv2.putText(visualization, f"{labels[index]} {float(confidence):.2f}",
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(diagnostics / "bboxes.png"), visualization)
    del dino_model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. SAM.
    started = time.time()
    mask, sam_report = segment_with_sam(
        image_bgr, detections.xyxy[box_index], weights_dir / "sam_b.pt", log)
    timings["sam_s"] = time.time() - started
    cv2.imwrite(str(diagnostics / "mask.png"), mask.astype(np.uint8) * 255)

    # 4. Point cloud, workspace mask and object cluster.
    camera_info = CameraInfo(width, height, camera["fx"], camera["fy"],
                             camera["ppx"], camera["ppy"], 1.0)
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
    rotation = camera["T_world_camera"][:3, :3]
    translation = camera["T_world_camera"][:3, 3]
    world = cloud @ rotation.T + translation
    valid = depth > 0
    table = ((world[..., 0] >= limits["table_x"][0]) & (world[..., 0] <= limits["table_x"][1]) &
             (world[..., 1] >= limits["table_y"][0]) & (world[..., 1] <= limits["table_y"][1]))
    reach_center = limits.get("reach_center")
    if reach_center is not None and limits.get("reach_radius_m") is not None:
        center = np.asarray(reach_center, dtype=float)
        distance_sq = np.sum((world - center) ** 2, axis=-1)
        radius = float(limits["reach_radius_m"])
        reach = distance_sq <= radius * radius
        min_radius = float(limits.get("reach_min_radius_m", 0.0))
        if min_radius > 0.0:
            reach &= distance_sq >= min_radius * min_radius
        table &= reach
    workspace = (valid & (world[..., 2] >= limits["table_z"][0]) &
                 (world[..., 2] <= limits["table_z"][1]) & table)
    object_mask = (mask & valid & (world[..., 2] >= limits["object_z"][0]) &
                   (world[..., 2] <= limits["object_z"][1]) & table)
    object_points = world[object_mask]
    if len(object_points) < 20:
        raise TargetNotFoundError(
            f"SAM/workspace produced too few object points ({len(object_points)})")
    centroid = object_points.mean(axis=0)
    log(f"Workspace points: {int(workspace.sum())}, object cluster: {len(object_points)}, "
        f"centroid={centroid.round(4).tolist()}")

    cloud_flat = cloud.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    workspace_flat = np.flatnonzero(workspace)
    roi = ((np.abs(world[..., 0] - centroid[0]) <= args.grasp_roi) &
           (np.abs(world[..., 1] - centroid[1]) <= args.grasp_roi))
    # Grasp proposals are target-conditioned: the SAM mask decides which
    # points GraspNet may use.  The full workspace cloud is still passed to the
    # collision detector below, so table and neighbouring objects remain
    # obstacles without becoming grasp candidates.
    from utils.piperx_vision_control import sample_target_grasp_points
    sampled_index, point_sample_report = sample_target_grasp_points(
        object_mask, roi, workspace, NUM_POINT)
    pointcloud_report = {
        "valid_depth_pixels": int(valid.sum()),
        "workspace_points": int(workspace.sum()),
        "object_cluster_points": int(len(object_points)),
        "object_centroid_world": centroid.tolist(),
        "object_bbox_world": [object_points.min(axis=0).tolist(), object_points.max(axis=0).tolist()],
        "grasp_roi_half_size_m": args.grasp_roi,
        "grasp_roi_points": point_sample_report["target_points_in_roi"],
        "workspace_roi_points": point_sample_report["workspace_points_in_roi"],
        "sampled_points": int(len(sampled_index)),
        "sampled_unique_points": point_sample_report["sampled_unique_points"],
        "sampled_with_replacement": point_sample_report["sampled_with_replacement"],
        "workspace_world_z_range": [float(world[workspace][:, 2].min()),
                                    float(world[workspace][:, 2].max())],
        "reach_center": None if limits.get("reach_center") is None else list(limits["reach_center"]),
        "reach_radius_m": limits.get("reach_radius_m"),
        "reach_min_radius_m": limits.get("reach_min_radius_m"),
        "target_key": target_key,
        "target_phrase": target_phrase,
        "spatial_reference": spatial_reference,
        "sam_report": sam_report,
    }
    write_json(diagnostics / "pointcloud_stats.json", pointcloud_report)

    observation = {"id": diagnostics.parent.name + "/" + diagnostics.name,
                   "diagnostics_dir": str(diagnostics),
                   "camera": camera, "rgb": rgb,
                   "centroid": centroid.copy(), "object_points": object_points.copy(),
                   "timings_s": timings,
                   "vlm_verification": (vlm_verification.as_dict()
                                        if vlm_verification is not None else None)}
    if not predict_grasps:
        return observation

    def no_usable_grasp(reason, *, raw, nms, collision):
        write_json(diagnostics / "grasp_counts.json", {
            "raw_count": raw, "nms_count": nms,
            "collision_filtered": collision, "remaining": 0,
            "reason": reason})
        log(reason)
        raise NoUsableGraspError(reason)

    # 5. GraspNet.
    from graspnet import pred_decode
    from utils.graspnet_runner import load_graspnet

    started = time.time()
    net = load_graspnet(str(weights_dir / "logs/log_rs/checkpoint-rs.tar"))
    timings["graspnet_load_s"] = time.time() - started
    points = cloud_flat[sampled_index].astype(np.float32)
    colors = rgb_flat[sampled_index].astype(np.float32) / 255.0
    end_points = {"point_clouds": torch.from_numpy(points[None]).cuda(),
                  "cloud_colors": colors}
    started = time.time()
    with torch.inference_mode():
        end_points = net(end_points)
        predictions = pred_decode(end_points)
    timings["graspnet_infer_s"] = time.time() - started
    raw_array = predictions[0].detach().cpu().numpy()
    del net, end_points, predictions
    gc.collect()
    torch.cuda.empty_cache()
    grasps = GraspGroup(raw_array)
    raw_count = len(grasps)
    log(f"GraspNet raw grasps: {raw_count}")
    if raw_count == 0:
        no_usable_grasp("GraspNet produced no grasps",
                        raw=0, nms=0, collision=0)
    np.save(diagnostics / "grasps_raw.npy", raw_array)
    grasps = grasps.nms()
    grasps.sort_by_score()
    nms_count = len(grasps)
    if nms_count == 0:
        no_usable_grasp("GraspNet NMS produced no grasps",
                        raw=raw_count, nms=0, collision=0)

    # 6. Collision filter in the camera optical frame.
    started = time.time()
    scene_points = cloud_flat[workspace_flat].astype(np.float32)
    detector = ModelFreeCollisionDetector(scene_points, voxel_size=VOXEL_SIZE)
    collision_mask = detector.detect(grasps, approach_dist=APPROACH_DIST,
                                     collision_thresh=COLLISION_THRESH)
    grasps = grasps[~collision_mask]
    grasps.sort_by_score()
    timings["collision_s"] = time.time() - started
    log(f"After NMS: {nms_count}, after collision: {len(grasps)}")
    if len(grasps) == 0:
        no_usable_grasp(
            "Every GraspNet candidate was rejected by collision detection",
            raw=raw_count, nms=nms_count,
            collision=int(collision_mask.sum()))

    observation["grasps"] = grasps
    write_json(diagnostics / "grasp_counts.json", {
        "raw_count": raw_count, "nms_count": nms_count,
        "collision_filtered": int(collision_mask.sum()), "remaining": len(grasps)})
    return observation


def choose_candidate(env, observation, args, diagnostics, log, home_pose, cycle=0, *,
                     exclude=None, stage="final"):
    from utils.piperx_vision_control import (
        APPROACH_HEIGHT, TOP_DOWN, observation_pregrasp_pose,
        parallel_jaw_rotation_variants, plan_grasp_only, plan_pick_place,
        rotation_angle_radians, two_finger_support)
    diagnostics.mkdir(parents=True, exist_ok=True)
    grasps, object_points = observation["grasps"], observation["object_points"]
    limits = workspace_limits(env)
    candidates = []
    checked = 0
    excluded = set() if exclude is None else set(exclude)
    reference_rotation = np.asarray(env.tcp_pose[:3, :3], dtype=float)
    grasp_only = bool(getattr(args, "grasp_only", False))
    # Top-down is preferred, not mandatory.  Try every top-down proposal
    # first; only if none survives the support/path/IK preflight, retry the
    # oblique/side proposals.  This applies to both grasp-only and full
    # pick-and-place: a side grasp is a fallback, never a candidate that can
    # outrank a valid top-down grasp.  ``--side-grasps`` is kept only as a
    # backwards-compatible flag.
    approach_passes = [
        ("top_down", APPROACH_COS, False),
        ("fallback", GRASP_ONLY_APPROACH_COS, True),
    ]

    def preflight(entry):
        nonlocal checked
        if entry["rank"] in excluded or not all(entry["checks"].values()):
            return None
        checked += 1
        proposal_tcp = np.asarray(entry["world_tcp"], dtype=float)
        insertion = None
        if entry.get("source") == "graspnet" and args.grasp_insertion_depth > 0:
            insertion = bounded_grasp_insertion(
                proposal_tcp, args.grasp_insertion_depth,
                table_top_z=limits["table_top_z"])
            base_tcp = insertion["tcp"]
        else:
            base_tcp = proposal_tcp
        if base_tcp.shape != (4, 4) or not np.isfinite(base_tcp).all():
            raise RuntimeError(f"Invalid TCP for {entry.get('source', 'graspnet')} "
                               f"rank {entry['rank']}")
        entry["jaw_attempts"] = []
        for variant in parallel_jaw_rotation_variants(
                base_tcp[:3, :3], reference=reference_rotation):
            tcp = base_tcp.copy()
            tcp[:3, :3] = variant["rotation"]
            trial = dict(entry)
            trial.update({
                "stage": stage,
                "proposal_tcp": proposal_tcp.tolist(),
                "world_tcp": tcp.tolist(),
                "world_position": tcp[:3, 3].tolist(),
                "jaw_variant": variant["name"],
                "jaw_branch": variant["branch"],
                "jaw_flipped": bool(variant["flipped"]),
                "jaw_reference_angle_rad": float(variant["angle_rad"]),
            })
            if insertion is not None:
                trial["grasp_insertion"] = {
                    key: value for key, value in insertion.items() if key != "tcp"}
            # Reject a pose whose point-cloud support is marginal before
            # planning motion.  The later run-time check uses the 15 mm
            # default; this pre-selection stays a little stricter.  The old
            # 10 mm limit missed a valid bottle candidate by 0.3 mm (the
            # candidate had points on both jaws and a 55 mm span), so use
            # 12 mm while still rejecting the 19 mm endpoint-like proposals.
            # A diagonal/side fallback proposal keeps the 15 mm allowance
            # because the thin-object support slice is less complete.
            diagonal = entry.get("approach_mode") == "fallback"
            if diagonal:
                support = two_finger_support(
                    observation["object_points"], tcp,
                    width=entry.get("width"),
                    min_side_points=30, max_center_offset=0.015)
            else:
                support = two_finger_support(
                    observation["object_points"], tcp,
                    width=entry.get("width"),
                    min_side_points=60, max_center_offset=0.012)
            trial["selection_support"] = support
            if not support["supported"]:
                entry["jaw_attempts"].append({
                    "jaw_variant": variant["name"],
                    "jaw_branch": variant["branch"],
                    "jaw_reference_angle_rad": float(variant["angle_rad"]),
                    "status": "rejected_support",
                    "support": support,
                })
                log(f"Selection support rejected "
                    f"{entry.get('source', 'graspnet')} rank {entry['rank']} "
                    f"jaw={variant['name']} ({variant['branch']}): "
                    f"{support['reason']}, center={support['center_m']:.4f} m")
                continue
            reachability = getattr(env, "reachability", None)
            if reachability is not None:
                pregrasp = tcp.copy()
                pregrasp[:3, 3] = (
                    tcp[:3, 3] - tcp[:3, :3][:, 2] * APPROACH_HEIGHT)
                try:
                    feasibility = reachability.feasible(
                        tcp, pregrasp, seed_joints=env.get_joint())
                except Exception as error:
                    feasibility = {"feasible": False, "failed_pose": "ik_check",
                                   "error": str(error), "seeds_tried": 0}
                trial["ik_feasibility"] = feasibility
                if not feasibility["feasible"]:
                    entry["jaw_attempts"].append({
                        "jaw_variant": variant["name"],
                        "jaw_branch": variant["branch"],
                        "jaw_reference_angle_rad": float(variant["angle_rad"]),
                        "status": "rejected_ik",
                        "ik_feasibility": feasibility,
                    })
                    log(f"IK rejected {entry.get('source', 'graspnet')} "
                        f"rank {entry['rank']} jaw={variant['name']} "
                        f"({variant['branch']}): {feasibility.get('error')}")
                    continue
            try:
                if getattr(args, "grasp_only", False):
                    plans = plan_grasp_only(env, tcp)
                else:
                    plans = plan_pick_place(env, tcp, (args.drop_x, args.drop_y), home_pose)
            except RuntimeError as error:
                entry["jaw_attempts"].append({
                    "jaw_variant": variant["name"],
                    "jaw_branch": variant["branch"],
                    "jaw_reference_angle_rad": float(variant["angle_rad"]),
                    "status": "rejected",
                    "error": str(error),
                })
                log(f"Preflight rejected {entry.get('source', 'graspnet')} "
                    f"rank {entry['rank']} jaw={variant['name']} "
                    f"({variant['branch']}): {error}")
                continue
            trial["path_preflight"] = {
                name: len(plan["goals"]) for name, plan in plans.items()}
            trial["status"] = "planned"
            trial["reorient_required"] = "reorient" in plans
            if "reorient" in plans:
                reorient_rotation = plans["reorient"].get("rotation")
                if reorient_rotation is not None:
                    trial["reorient_jaw_variant"] = (
                        "flipped" if rotation_angle_radians(
                            np.asarray(reorient_rotation, dtype=float), TOP_DOWN) > np.pi / 2
                        else "original")
            entry["jaw_attempts"].append({
                "jaw_variant": variant["name"],
                "jaw_branch": variant["branch"],
                "jaw_reference_angle_rad": float(variant["angle_rad"]),
                "status": "planned",
                "segments": sorted(plans),
            })
            entry["status"] = "planned"
            entry["selected_jaw_variant"] = variant["name"]
            return {**trial, "plans": plans}
        return None

    checked_total = 0
    for pass_name, pass_approach_min, _pass_diagonal in approach_passes:
        checked = 0
        for rank in range(len(grasps)):
            if checked >= args.max_candidates:
                break
            if rank in excluded:
                continue
            grasp = grasps[rank]
            tcp = env.camera_grasp_to_world(
                grasp.translation, grasp.rotation_matrix,
                camera=VISION_CAMERA,
                T_world_camera=observation["camera"]["T_world_camera"])
            position, rotation = tcp[:3, 3], tcp[:3, :3]
            approach_cos = float(rotation[:, 2] @ np.array([0., 0., -1.]))
            # Top-down pass handles only the preferred proposals.  The
            # fallback pass must not re-try those with looser support rules.
            if pass_name == "fallback" and approach_cos > APPROACH_COS:
                continue
            approach_axis = rotation[:, 2]
            pregrasp_position = position - approach_axis * APPROACH_HEIGHT
            approach_ok = bool(
                approach_cos > pass_approach_min and
                pregrasp_position[2] >=
                limits["table_top_z"] + PREGRASP_TABLE_CLEARANCE)
            if grasp_only:
                approach_ok = bool(
                    approach_ok and in_workspace_xy(pregrasp_position, limits))
            distance = float(np.linalg.norm(
                object_points[:, :2] - position[None, :2], axis=1).min())
            checks = {"width": bool(MIN_WIDTH <= grasp.width <= MAX_WIDTH),
                      "approach": approach_ok,
                      "center": bool(
                          limits["grasp_z"][0] <= position[2] <= limits["grasp_z"][1]
                          and in_workspace_xy(position, limits)),
                      "object": bool(distance < OBJECT_XY_DIST)}
            entry = {"source": "graspnet", "rank": rank, "stage": stage,
                     "approach_mode": pass_name,
                     "score": float(grasp.score), "width": float(grasp.width),
                     "world_tcp": tcp.tolist(), "world_position": position.tolist(),
                     "pregrasp_position": pregrasp_position.tolist(),
                     "camera_translation": grasp.translation.tolist(),
                     "camera_rotation": grasp.rotation_matrix.tolist(),
                     "checks": checks, "approach_cos": approach_cos,
                     "object_xy_distance": distance}
            candidates.append(entry)
            chosen = preflight(entry)
            if chosen is not None:
                write_json(diagnostics / "grasps.json", {"candidates": candidates})
                write_json(diagnostics / "selected.json",
                           {key: value for key, value in chosen.items()
                            if key != "plans"})
                log(f"Selected {stage} graspnet rank {rank} "
                    f"[{pass_name}] jaw={chosen['jaw_variant']} "
                    f"({chosen['jaw_branch']}, "
                    f"{np.rad2deg(chosen['jaw_reference_angle_rad']):.1f} deg): "
                    f"all {len(chosen['plans'])} trajectory segments preflighted")
                return chosen
            if checked >= args.max_candidates:
                break
        checked_total += checked

    check_failures = {}
    for entry in candidates:
        for name, passed in entry.get("checks", {}).items():
            if not passed:
                check_failures[name] = check_failures.get(name, 0) + 1
    if check_failures:
        log("Geometry checks rejected candidates: " + ", ".join(
            f"{name}={count}" for name, count in sorted(check_failures.items())))
    write_json(diagnostics / "grasps.json", {"candidates": candidates})
    # This is a searchable failure: the target is visible, but this view has
    # no candidate that survives support, IK and path preflight.  The caller
    # should try the next J1 view even if the VLM said the current view looks
    # well framed (visibility is not graspability).
    raise NoUsableGraspError(
        f"No fully reachable path among {checked_total} preflighted candidates "
        f"({len(candidates)} collision-free GraspNet candidates)")


def apply_instruction(env, args, diagnostics, log):
    """Turn an instruction or --target into a DINO prompt and drop point."""
    if args.instruction:
        planner = build_semantic_planner(
            args.semantic_backend, base_url=args.vlm_base_url, model=args.vlm_model,
            timeout=args.vlm_timeout, max_tokens=args.vlm_max_tokens, log=log)
        frame = env.render(camera=VISION_CAMERA, width=RENDER_WIDTH, height=RENDER_HEIGHT)
        save_bgr(diagnostics / "semantic_rgb.png", frame["img"])
        bodies = ([] if getattr(env, "is_real", False)
                  else sorted(scene_bodies(env)))
        plan = planner.plan(frame["img"], args.instruction,
                            scene_context=", ".join(bodies) if bodies else None)
        args.semantic_planner = planner
        if args.text is None:
            args.text = plan.dino_prompt
        args.semantic_plan = plan
        args.drop_x, args.drop_y = drop_xy_for_plan(
            plan, drop_x=args.drop_x, drop_y=args.drop_y)
        write_json(diagnostics / "semantic_plan.json", {
            "planner": planner.name,
            "plan": plan.as_dict(),
            "scene_bodies": bodies,
            "target_body": None,
        })
        log(f"Semantic plan ({planner.name}): {plan.as_dict()}")
        log(f"DINO prompt: {args.text!r}; drop_xy={(args.drop_x, args.drop_y)}; "
            f"vlm_visible={plan.target_visible}, "
            f"vlm_yaw={plan.search_yaw_deg:+.1f} deg")
        return plan

    if args.target and args.text is None:
        args.text = args.target
    if args.drop_x is None:
        args.drop_x = DEFAULT_DROP_XY[0]
    if args.drop_y is None:
        args.drop_y = DEFAULT_DROP_XY[1]
    log(f"DINO prompt: {args.text!r}; drop_xy={(args.drop_x, args.drop_y)}")
    return None


def resolve_target_body(env, observation, diagnostics, log,
                        *, max_distance_m=BODY_MATCH_MAX_DISTANCE):
    """Match the segmented cloud to a free-joint scene body and select it."""
    previous = env.target_body or None
    body, distance, report = resolve_scene_body(
        env, observation["object_points"], max_distance_m=max_distance_m)
    report["selected_body"] = body
    report["selected_distance_m"] = None if body is None else distance
    write_json(diagnostics / "target_body.json", report)
    if body is None:
        candidates = ", ".join(entry["body"] for entry in report["candidates"])
        nearest = report["candidates"][0] if report["candidates"] else None
        nearest_text = (f"nearest {nearest['body']} at {distance:.4f} m"
                        if nearest is not None else "no scene bodies")
        raise RuntimeError(
            f"Detected object does not match any scene body ({nearest_text}, "
            f"limit {max_distance_m:.3f} m); scene bodies: {candidates or 'none'}")
    if previous is not None and previous != body:
        log(f"WARNING: segmentation matched {body!r} but {previous!r} was selected before perception")
    if env.target_body != body:
        env.set_target(body)
    log(f"Matched detected object to scene body {body!r} at {distance:.4f} m")
    return body, report


def check_target_consistency(env, observation, log,
                             *, max_distance_m=BODY_MATCH_MAX_DISTANCE):
    if env.target_body_id < 0:
        return
    position = env.data.xipos[env.target_body_id]
    distance = float(np.linalg.norm(position[:2] - observation["centroid"][:2]))
    if distance > max_distance_m:
        log(f"WARNING: observed object centroid is {distance:.4f} m from selected body "
            f"{env.target_body!r}; the object may have been displaced")


def run(args):
    import mujoco
    import torch
    from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv
    from utils.piperx_vision_control import run_closed_loop
    from utils.piperx_grasp_display import GraspPoseDisplay

    is_real = bool(args.real)
    if not (args.instruction or args.text or args.target):
        raise ValueError("Provide --instruction, --text, or --target")
    if (args.execute or args.open_gate) and not is_real:
        raise ValueError("--execute and --open-gate require --real")
    if is_real and (args.obj_x is not None or args.obj_y is not None or args.obj_yaw):
        raise ValueError("--obj-x/--obj-y/--obj-yaw are simulation-only")
    if not 1 <= args.max_attempts <= 5 or args.max_candidates < 1:
        raise ValueError("max-attempts must be 1..5 and max-candidates must be positive")
    if not np.isfinite(args.grasp_preview_seconds) or args.grasp_preview_seconds < 0:
        raise ValueError("grasp-preview-seconds must be finite and nonnegative")
    if not np.isfinite(args.grasp_insertion_depth) or args.grasp_insertion_depth < 0:
        raise ValueError("grasp-insertion-depth must be finite and nonnegative")
    search_angles = [] if args.no_search else list(args.search_angles_deg)
    if len(search_angles) > 12:
        raise ValueError("search-angles-deg accepts at most 12 offsets")
    if any(not np.isfinite(value) or abs(value) > 120.0 for value in search_angles):
        raise ValueError("search-angles-deg offsets must be finite and within +/-120 deg")
    if not np.isfinite(args.vlm_timeout) or args.vlm_timeout <= 0:
        raise ValueError("vlm-timeout must be positive and finite")
    if args.vlm_max_tokens < 1:
        raise ValueError("vlm-max-tokens must be positive")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        raise RuntimeError("The installed GraspNet CUDA extensions require a working NVIDIA GPU")
    diagnostics = args.diagnostics_dir
    if not diagnostics.is_absolute():
        diagnostics = ROOT / diagnostics
    diagnostics.mkdir(parents=True, exist_ok=True)
    args.text_override = args.text is not None
    args.center_target = bool(
        is_real and args.execute and not args.perception_only and
        args.center_target and not args.no_center_target)
    args.last_view_j1_rad = None
    log = Logger(diagnostics / "run.log")
    log(f"Diagnostics: {diagnostics}")
    log(f"Args: {vars(args)}")
    weights_dir = Path(load_config("cv.yaml").get("weights_dir", "weights"))
    if not weights_dir.is_absolute():
        weights_dir = ROOT / weights_dir

    with ExitStack() as resources:
        if is_real:
            from utils.piperx_real_env import RealPiperXEnv
            from utils.piperx_reachability import Reachability
            env = resources.enter_context(RealPiperXEnv(
                config_path=args.real_config,
                execute=args.execute,
                open_gate=args.open_gate,
                log=log,
            ))
            env.reachability = resources.enter_context(Reachability())
            log("IK reachability checker ready")
            grasp_display = None
            preview = None
        else:
            env = resources.enter_context(PiperXGraspEnv(
                viewer=not args.headless,
                realtime=not args.fast,
                render_size=(RENDER_WIDTH, RENDER_HEIGHT),
            ))
            grasp_display = resources.enter_context(GraspPoseDisplay(env))
            preview = None
            if not args.headless and not args.no_camera_preview:
                from utils.piperx_camera_preview import WristCameraPreview
                preview = resources.enter_context(
                    WristCameraPreview(env, grasp_display=grasp_display))
                physics_step = env.step
                def preview_step():
                    physics_step()
                    preview.update()
                env.step = preview_step
        env.wait(0.4)
        baseline_pose = env.tcp_pose.copy() if is_real else None
        baseline_joints = env.get_joint().copy() if is_real else None
        if is_real and args.execute:
            violations = env.joint_limit_violations()
            if violations:
                details = ", ".join(
                    f"joint{index}={value:.2f} deg outside "
                    f"[{low:.1f}, {high:.1f}]"
                    for index, value, low, high in violations)
                log(f"Current joints outside configured limits: {details}")
                limits = env.config["arm"]["joint_limits_deg"]
                if len(limits) != 6:
                    raise RuntimeError("arm.joint_limits_deg must contain six pairs")
                lows = np.radians([float(pair[0]) for pair in limits])
                highs = np.radians([float(pair[1]) for pair in limits])
                # The firmware limits are real; do not widen them in the
                # config.  Nudge the current pose just inside them so the
                # first Cartesian approach is not rejected by arm_status=4.
                margin = np.radians(2.0)
                safe_joints = np.clip(
                    env.get_joint(), lows + margin, highs - margin)
                log("Auto joint-limit recovery to " +
                    np.array2string(np.degrees(safe_joints), precision=2) +
                    " deg (firmware limits unchanged)")
                env.move_to_joint(safe_joints, duration=3.0)
                env.wait(0.4)
                remaining = env.joint_limit_violations()
                if remaining:
                    remaining_details = ", ".join(
                        f"joint{index}={value:.2f} deg outside "
                        f"[{low:.1f}, {high:.1f}]"
                        for index, value, low, high in remaining)
                    raise RuntimeError(
                        "Auto joint-limit recovery did not reach a safe pose: "
                        f"{remaining_details}")
                baseline_joints = env.get_joint().copy()
                baseline_pose = env.tcp_pose.copy()
                log("Auto joint-limit recovery complete; baseline updated")
        apply_instruction(env, args, diagnostics, log)
        if args.perception_only:
            folder = diagnostics / "perception"
            result = perceive(env, args, weights_dir, folder, log,
                              predict_grasps=True)
            payload = {
                key: value for key, value in result.items()
                if key not in ("rgb", "grasps", "object_points")
            }
            payload["object_point_count"] = int(len(result["object_points"]))
            grasps = result.get("grasps")
            if grasps is not None:
                payload["grasp_count"] = int(len(grasps))
                payload["top_grasps"] = [
                    {
                        "translation": grasp.translation.tolist(),
                        "rotation": grasp.rotation_matrix.tolist(),
                        "score": float(grasp.score),
                        "width": float(grasp.width),
                    }
                    for grasp in list(grasps)[:10]
                ]
            result["success"] = True
            result["perception_only"] = True
            payload["success"] = True
            payload["perception_only"] = True
            write_json(diagnostics / "result.json", payload)
            log("PERCEPTION " + json.dumps(jsonable(payload), indent=2))
            return result
        # Initial conditions are allowed before the first perception frame.
        if not is_real and (args.obj_x is not None or args.obj_y is not None or args.obj_yaw):
            bodies = scene_bodies(env)
            if args.target not in bodies:
                raise ValueError(
                    f"--obj-x/--obj-y/--obj-yaw require --target to be a scene body name; "
                    f"choose one of: {', '.join(sorted(bodies))}")
            env.set_target(args.target)
            address = env.model.joint(env.target_free_joint_name).qposadr[0]
            if args.obj_x is not None:
                env.data.qpos[address] = args.obj_x
            if args.obj_y is not None:
                env.data.qpos[address + 1] = args.obj_y
            env.data.qpos[address + 2] = float(env.model.body(env.target_body).pos[2])
            env.data.qpos[address + 3:address + 7] = [np.cos(args.obj_yaw / 2), 0, 0, np.sin(args.obj_yaw / 2)]
            mujoco.mj_forward(env.model, env.data)
            env.wait(0.3)

        # Independent simulation audit. None of these values enter the policy.
        initial_truth = None
        target_resolved = False
        unexpected_contacts = set()
        if not is_real:
            original_step = env.step
            def audit_step():
                original_step()
                robot = env.model.body_rootid[env.model.body("base_link").id]
                for contact in env.data.contact:
                    roots = env.model.body_rootid[env.model.geom_bodyid[[contact.geom1, contact.geom2]]]
                    if (roots == robot).all() or (robot in roots and 0 in roots):
                        unexpected_contacts.add(tuple(sorted(env.model.geom(i).name for i in (contact.geom1, contact.geom2))))
            env.step = audit_step
        observations = []
        displayed_grasps = []
        semantic_planner = getattr(args, "semantic_planner", None)
        semantic_context = None
        if semantic_planner is not None and not is_real:
            semantic_bodies = sorted(scene_bodies(env))
            semantic_context = (", ".join(semantic_bodies)
                                if semantic_bodies else None)

        def verify_vlm(image_bgr, detection):
            """Ask the VLM whether the target is actually in this view."""
            import cv2
            if semantic_planner is None or not args.instruction:
                return None
            annotated = image_bgr.copy()
            if detection:
                x1, y1, x2, y2 = [int(round(v)) for v in detection["box"]]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    annotated,
                    f"{detection.get('label', '')} "
                    f"{float(detection.get('confidence', 0.0)):.2f}",
                    (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            plan = semantic_planner.plan(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                args.instruction,
                scene_context=semantic_context,
                detection=detection)
            args.semantic_plan = plan
            args.last_vlm_plan = plan
            if not getattr(args, "text_override", False):
                args.text = plan.dino_prompt
            args.drop_x, args.drop_y = drop_xy_for_plan(
                plan, drop_x=args.drop_x, drop_y=args.drop_y)
            log(f"VLM verify: visible={plan.target_visible}, "
                f"selected_matches={plan.selected_matches_target}, "
                f"yaw={plan.search_yaw_deg:+.1f} deg, "
                f"reason={plan.search_reason!r}")
            if not plan.target_visible:
                raise TargetNotFoundError(
                    f"VLM: target not visible "
                    f"({plan.search_reason or 'no reason'}); "
                    f"search_yaw={plan.search_yaw_deg:+.1f} deg",
                    search_yaw_deg=plan.search_yaw_deg)
            if detection is not None and plan.selected_matches_target is False:
                raise TargetNotFoundError(
                    f"VLM: detector box label {detection.get('label')!r} "
                    f"is not the target; "
                    f"search_yaw={plan.search_yaw_deg:+.1f} deg",
                    search_yaw_deg=plan.search_yaw_deg)
            if detection is None:
                raise TargetNotFoundError(
                    f"VLM: target visible but GroundingDINO did not detect it; "
                    f"search_yaw={plan.search_yaw_deg:+.1f} deg",
                    search_yaw_deg=plan.search_yaw_deg)
            return plan

        def observe(cycle, phase):
            nonlocal initial_truth, target_resolved
            grasp_phase = phase in ("pickup", "pregrasp") or phase.startswith("pregrasp_retry")
            if grasp_phase:
                if grasp_display is not None:
                    grasp_display.clear()
                if preview is not None:
                    preview.update(force=True)
            folder = diagnostics / f"cycle_{cycle:02d}" / phase
            verify = (verify_vlm
                      if phase in ("pickup", "pregrasp")
                      or phase.startswith("pregrasp_retry") else None)
            centering_record = None
            if getattr(args, "center_target", False) and grasp_phase:
                # Locate the target once, rotate J1 until its 3D centroid is
                # horizontally centred, then run the full perception again.
                # This prevents GraspNet from proposing a grasp from the edge
                # of the wrist image, where the visible half of a bottle often
                # yields only oblique/unreachable candidates.
                probe = perceive(env, args, weights_dir, folder / "precenter",
                                 log, predict_grasps=False, verify=verify)
                centroid = probe.get("centroid")
                camera = probe.get("camera")
                if centroid is not None and camera is not None:
                    info = yaw_delta_to_center(centroid, camera)
                    centering_record = {
                        "target_centroid": np.asarray(centroid).tolist(),
                        "before_u_px": info.get("current_u"),
                        "target_u_px": camera["ppx"],
                        "tolerance_px": info.get("tolerance_px"),
                        "delta_deg": info.get("delta_deg"),
                        "solution_found": bool(info.get("solution_found")),
                        "centered_before_move": bool(info.get("centered")),
                        "applied_deg": 0.0,
                    }
                    log("Centering target: u="
                        f"{info.get('current_u')} px, centre={camera['ppx']} px, "
                        f"tol={info.get('tolerance_px')} px, "
                        f"J1 delta={info.get('delta_deg'):+.1f} deg")
                    delta_deg = float(info.get("delta_deg") or 0.0)
                    if not info.get("centered") and abs(delta_deg) >= 1.0:
                        current_joint = env.get_joint()
                        target_joint = current_joint.copy()
                        target_joint[0] += np.radians(delta_deg)
                        limits = env.config.get("arm", {}).get(
                            "joint_limits_deg")
                        if limits:
                            low = np.radians(float(limits[0][0]))
                            high = np.radians(float(limits[0][1]))
                            target_joint[0] = float(np.clip(
                                target_joint[0], low + np.radians(1.0),
                                high - np.radians(1.0)))
                        applied = float(np.degrees(
                            target_joint[0] - current_joint[0]))
                        if abs(applied) >= 0.5:
                            log(f"Centering J1 {np.degrees(current_joint[0]):+.1f}"
                                f" -> {np.degrees(target_joint[0]):+.1f} deg "
                                f"({applied:+.1f} deg)")
                            try:
                                env.move_to_joint(target_joint, duration=2.0)
                                env.wait(0.4)
                                centering_record["applied_deg"] = applied
                                centering_record["j1_before_deg"] = float(
                                    np.degrees(current_joint[0]))
                                centering_record["j1_after_deg"] = float(
                                    env.get_joint()[0] * 180.0 / np.pi)
                            except Exception as error:
                                log(f"Centering move failed: {error}")
                                centering_record["error"] = str(error)
                    if centering_record is not None:
                        write_json(folder / "centering.json", centering_record)
                del probe
            result = perceive(env, args, weights_dir, folder,
                              log, predict_grasps=grasp_phase, verify=verify)
            if centering_record is not None:
                after = yaw_delta_to_center(result["centroid"], result["camera"])
                centering_record["after_u_px"] = after.get("current_u")
                centering_record["centered_after"] = bool(after.get("centered"))
                result["centering"] = centering_record
                write_json(folder / "centering.json", centering_record)
            result["id"] = str(folder.relative_to(diagnostics))
            result["diagnostics_dir"] = str(folder)
            if is_real:
                target_resolved = True
            elif not target_resolved:
                resolve_target_body(env, result, folder, log)
                target_resolved = True
                initial_truth = env.object_position.copy()
            else:
                check_target_consistency(env, result, log)
            observations.append({"id": result["id"], "phase": phase,
                                 "centroid": result["centroid"],
                                 "camera": result["camera"],
                                 "timings_s": result["timings_s"],
                                 "centering": result.get("centering"),
                                 "vlm_verification": result.get("vlm_verification")})
            return result
        def propose(observation, cycle, home_pose, *, stage="final", exclude=None):
            folder = diagnostics / f"cycle_{cycle:02d}" / stage
            selected = choose_candidate(env, observation, args, folder, log, home_pose,
                                        cycle=cycle, exclude=exclude, stage=stage)
            if grasp_display is None:
                info = {
                    "frame": "base",
                    "rank": selected.get("rank"),
                    "score": selected.get("score"),
                    "width": selected.get("width"),
                    "world_tcp": selected.get("world_tcp"),
                    "world_position": selected.get("world_position"),
                    "stage": stage,
                }
                displayed_grasps.append(info)
                write_json(folder / "grasp_pose.json", info)
                save_bgr(folder / "selected_pose_rgb.png", observation["rgb"])
            else:
                grasp_display.set_grasp(selected, cycle)
                info = grasp_display.info.copy()
                info["stage"] = stage
                displayed_grasps.append(info)
                write_json(folder / "grasp_pose.json", info)
                save_bgr(folder / "selected_pose_rgb.png",
                         grasp_display.annotate_rgb(observation["rgb"], observation["camera"]))
                save_bgr(folder / "selected_pose_3d.png", grasp_display.render_overview())
                save_bgr(folder / "selected_pose_closeup.png", grasp_display.render_overview(closeup=True))
            log(f"Selected {stage} TCP: " + json.dumps(info))
            if preview is not None:
                preview.update(force=True)
            if not args.headless and grasp_display is not None:
                deadline = time.monotonic() + args.grasp_preview_seconds
                while time.monotonic() < deadline:
                    if not env.viewer.is_running():
                        raise KeyboardInterrupt
                    env.viewer.sync()
                    time.sleep(min(0.03, max(0., deadline - time.monotonic())))
            return selected

        def propose_pregrasp(observation, cycle, home_pose):
            errors = []
            plan = None
            camera_heights = getattr(
                env, "pregrasp_camera_heights", (0.955, 0.935, 0.915))
            for camera_height in camera_heights:
                try:
                    position, rotation = observation_pregrasp_pose(
                        env, observation["centroid"], camera_height=camera_height)
                    plan = env.plan_to(position, rotation, 2.0)
                    break
                except RuntimeError as error:
                    errors.append(f"camera_height={camera_height}: {error}")
            if plan is None:
                raise RuntimeError("No reachable observation pre-grasp pose: " +
                                   " | ".join(errors))
            tcp = np.eye(4)
            tcp[:3, :3] = rotation
            tcp[:3, 3] = position
            log(f"Pre-grasp observation pose {np.round(position, 4).tolist()} "
                f"(camera height {camera_height:.3f} m)")
            return {"source": "observation", "rank": None, "score": None,
                    "width": None, "world_tcp": tcp.tolist(),
                    "world_position": position.tolist(), "stage": "pregrasp",
                    "plans": {"approach": plan}}

        def propose_final(observation, cycle, home_pose, *, exclude=None):
            return propose(observation, cycle, home_pose, stage="final", exclude=exclude)

        two_stage = not args.single_stage
        log(f"Grasp mode: {'two-stage' if two_stage else 'single-stage'}"
            f"{', grasp-only (no placement)' if args.grasp_only else ''}")
        if search_angles:
            log("Search when the target is missing or no usable grasp exists: "
                "base yaw offsets (deg) " +
                ", ".join(f"{value:+.0f}" for value in search_angles))
        else:
            log("Target search: disabled")
        search_state = {
            "index": 0,
            "base_joint": (baseline_joints.copy() if is_real
                           else env.get_joint().copy()),
            "seen_deg": [],
        }
        move_joint = (getattr(env, "move_to_joint", None) or
                      getattr(env, "move_joints", None))

        def recover_to_baseline(env, home_pose, reference_z):
            try:
                recover(env, home_pose, reference_z)
            except Exception as error:
                # A failed side approach can leave the TCP just outside the
                # configured workspace, which makes Cartesian recovery reject
                # its own current pose.  Joint-space return does not have that
                # failure mode.
                log(f"Cartesian recovery failed ({error}); "
                    f"returning to baseline in joint space")
                try:
                    env.gripper_control(0.1)
                except Exception as gripper_error:
                    log(f"Gripper open during fallback recovery failed: {gripper_error}")
                move_joint(search_state["base_joint"], duration=3.0)

        def retry_search(env, home_pose, reference_z, record):
            phase = record.get("phase")
            error_type = record.get("error_type")
            last_vlm = getattr(args, "last_vlm_plan", None)
            last_vlm_yaw = 0.0
            if last_vlm is not None and getattr(last_vlm, "target_visible", False):
                try:
                    last_vlm_yaw = float(last_vlm.search_yaw_deg)
                except (TypeError, ValueError):
                    last_vlm_yaw = 0.0
                if not np.isfinite(last_vlm_yaw):
                    last_vlm_yaw = 0.0
            searchable_failure = error_type in (
                "TargetNotFoundError", "NoUsableGraspError")
            if not searchable_failure and abs(last_vlm_yaw) < 2.0:
                log(f"{error_type}: target is present and VLM did not suggest "
                    f"another view; not rotating J1")
                if phase not in ("perception", "planning"):
                    recover_to_baseline(env, home_pose, reference_z)
                return False
            if error_type == "NoUsableGraspError" and abs(last_vlm_yaw) < 2.0:
                log("No usable grasp in this view (support/IK/path); "
                    "searching the next J1 view even though the target is visible")
            elif error_type != "TargetNotFoundError":
                log(f"{error_type}: VLM suggested a different view; using its yaw")
            index = search_state["index"]
            if index >= len(search_angles):
                return False
            base_deg = float(np.degrees(search_state["base_joint"][0]))
            # VLM yaw is relative to the image/view it just saw.  Fixed
            # fallback angles are absolute offsets from the startup baseline.
            # Keeping these separate avoids the zig-zag where a "+20 to
            # centre" suggestion from a +30 view moved J1 back to +20.
            current_offset = 0.0
            last_view_j1 = getattr(args, "last_view_j1_rad", None)
            if last_view_j1 is not None:
                current_offset = float(np.degrees(last_view_j1) - base_deg)
            elif search_state["seen_deg"]:
                current_offset = float(search_state["seen_deg"][-1] - base_deg)
            seen_offsets = list(search_state["seen_deg"])
            seen_offsets.append(base_deg + current_offset)
            angle = None
            source = "fallback"
            suggested = record.get("search_yaw_deg")
            if suggested is None and abs(last_vlm_yaw) >= 2.0:
                suggested = last_vlm_yaw
            if suggested is not None:
                try:
                    relative = float(np.clip(float(suggested), -120.0, 120.0))
                except (TypeError, ValueError):
                    relative = 0.0
                proposed = float(np.clip(current_offset + relative, -120.0, 120.0))
                if (np.isfinite(proposed) and abs(relative) >= 2.0 and
                        all(abs(base_deg + proposed - seen) >= 3.0
                            for seen in seen_offsets)):
                    angle = proposed
                    source = f"VLM {relative:+.1f} from {current_offset:+.1f}"
                elif np.isfinite(proposed) and abs(relative) >= 2.0:
                    log("VLM suggested an already-tried view; using fallback angle")
            if angle is None:
                angle = search_angles[index]
            search_state["index"] += 1
            search_state["seen_deg"].append(base_deg + angle)
            if phase in ("perception", "planning"):
                log(f"Search retry {index + 1}/{len(search_angles)} ({source}): "
                    f"J1 {angle:+.1f} deg (arm still at the observation pose)")
            else:
                log(f"Search retry {index + 1}/{len(search_angles)} ({source}): "
                    f"recover, then J1 {angle:+.1f} deg")
                recover_to_baseline(env, home_pose, reference_z)
            target_joint = search_state["base_joint"].copy()
            target_joint[0] += np.radians(angle)
            move_joint(target_joint, duration=2.0)
            env.wait(0.4)
            log(f"Search view {index + 1}/{len(search_angles)} ready: "
                f"J1 {angle:+.1f} deg")
            return True

        cycle_limit = (max(args.max_attempts, 1 + len(search_angles))
                       if search_angles else args.max_attempts)
        try:
            result = run_closed_loop(
                env, observe, propose, drop_xy=(args.drop_x, args.drop_y),
                max_attempts=cycle_limit, log=log,
                release_first=args.test_release_first,
                two_stage=two_stage,
                propose_pregrasp=propose_pregrasp if two_stage else None,
                propose_final=propose_final if two_stage else None,
                support_check=True,
                max_support_trials=args.max_candidates,
                reobserve_final=((lambda retry_cycle:
                                  observe(retry_cycle, "pregrasp_retry"))
                                 if two_stage else None),
                grasp_only=args.grasp_only,
                retry=retry_search if search_angles else None)
        except KeyboardInterrupt:
            if is_real and args.execute:
                try:
                    env.emergency_stop()
                    log("Interrupted; emergency hold sent (arm stays enabled)")
                except Exception as error:
                    log(f"Emergency hold failed: {error}")
            raise
        except Exception:
            if is_real and args.execute:
                try:
                    env.emergency_stop()
                    log("Pipeline aborted; emergency hold sent (arm stays enabled)")
                except Exception as error:
                    log(f"Emergency hold failed: {error}")
            raise
        if is_real and args.execute:
            try:
                current = env.tcp_pose
                current_joints = env.get_joint()
                position_gap = np.linalg.norm(
                    current[:3, 3] - baseline_pose[:3, 3])
                angle_gap = np.arccos(np.clip(
                    (np.trace(current[:3, :3].T @ baseline_pose[:3, :3]) - 1) / 2,
                    -1.0, 1.0))
                joint_gap = float(np.max(np.abs(
                    current_joints - baseline_joints)))
                if (position_gap < 0.01 and angle_gap < np.deg2rad(2.0) and
                        joint_gap < 0.02):
                    log("Already at baseline pose")
                else:
                    # A Cartesian straight line can push joint4 into its
                    # +/-89 deg stop (observed on the real arm).  Return in
                    # joint space to the exact captured baseline instead.
                    env.move_to_joint(baseline_joints, duration=3.0)
                    log("Returned to baseline joints")
            except Exception as error:
                log(f"Return to baseline failed: {error}")
                try:
                    env.emergency_stop()
                    log("Emergency hold sent after failed return (arm stays enabled)")
                except Exception as hold_error:
                    log(f"Emergency hold failed: {hold_error}")
        semantic_plan = getattr(args, "semantic_plan", None)
        target_label = (semantic_plan.target_phrase if semantic_plan is not None
                        else (env.target_label or args.text))
        if is_real:
            last_attempt = result["attempts"][-1] if result["attempts"] else {}
            lift = max(
                (float(attempt.get("tcp_lift_m", 0.0))
                 for attempt in result["attempts"]),
                default=0.0,
            )
            xy_error = last_attempt.get("observed_placement_error_m")
            z_error = last_attempt.get("observed_height_error_m")
            truth_success = bool(result["policy_success"])
            initial_position = None
            final_position = None
            simulation_time = None
            truth_acceptance = {
                "mode": "real_vision_only",
                "used_for_control": False,
            }
        else:
            # Independent acceptance is evaluated only after the control policy ends.
            if initial_truth is None:
                raise RuntimeError("Closed-loop policy returned without an observation")
            final_truth = env.object_position.copy()
            lift = float(env.max_object_height - initial_truth[2])
            xy_error = float(np.linalg.norm(final_truth[:2] - [args.drop_x, args.drop_y]))
            z_error = float(abs(final_truth[2] - initial_truth[2]))
            truth_success = bool(lift >= 0.04 and xy_error < 0.04 and z_error < 0.015
                                 and not unexpected_contacts and not env.data.warning.number.any())
            initial_position = initial_truth
            final_position = final_truth
            simulation_time = float(env.data.time)
            truth_acceptance = {
                "success": truth_success,
                "unexpected_contacts": sorted(unexpected_contacts),
                "mujoco_warnings": env.data.warning.number.tolist(),
                "used_for_control": False,
            }
        result.update(success=bool(result["policy_success"] and truth_success),
                      target=(env.target_key or args.text), target_label=target_label,
                      target_body=env.target_body, instruction=args.instruction,
                      semantic_plan=(semantic_plan.as_dict()
                                     if semantic_plan is not None else None),
                      caption=args.text, drop_xy=[args.drop_x, args.drop_y],
                      grasp_visualization={"frame": "world", "selected_pickups": displayed_grasps,
                                           "viewer_enabled": bool(not args.headless and not is_real),
                                           "rgb_overlay": preview is not None},
                      initial_position=initial_position, final_position=final_position,
                      lift_height_m=lift, placement_error_m=xy_error, placement_z_error_m=z_error,
                      bilateral_contact=any(a.get("lift_sensor", {}).get("held", False) for a in result["attempts"]),
                      observations=observations, simulation_time_s=simulation_time,
                      truth_acceptance=truth_acceptance)
        if preview is not None:
            preview.update(force=True)
            result["camera_preview"] = {"enabled": True, "camera": VISION_CAMERA,
                                        "frame_count": preview.frame_count, "fps": 10,
                                        "display": "viewer RGB/depth insets"}
            if preview.last_panel is not None:
                save_bgr(diagnostics / "camera_preview.png", preview.last_panel)
        else:
            result["camera_preview"] = {"enabled": False}
        write_json(diagnostics / "result.json", result)
        write_json(diagnostics / "selected.json", result["selected"])
        log("RESULT " + json.dumps(jsonable(result), indent=2))
        if args.hold and not args.headless and not is_real:
            while env.viewer.is_running():
                env.step()
        return result


def main():
    args = parse_args()
    try:
        result = run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
        raise
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
