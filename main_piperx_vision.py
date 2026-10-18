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

from utils.piperx_semantic import (  # noqa: E402
    DEFAULT_OBJECT,
    TABLE_TOP_Z,
    resolve_target,
    semantic_match_score,
    target_choices,
)
from utils.piperx_vlm import (  # noqa: E402
    DEFAULT_DROP_XY,
    build_semantic_planner,
    drop_xy_for_plan,
)
from utils.piperx_vision_control import (  # noqa: E402
    GRASP_INSERTION_DEPTH,
    bounded_grasp_insertion,
    observation_pregrasp_pose,
)

setup_graspnet_path()

# The editable graspnetAPI install is shadowed once graspnet's model directory
# puts graspnet-baseline on sys.path, so import the API before the model.
from graspnetAPI import GraspGroup  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # noqa: E402

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
APPROACH_COS = 0.8
OBJECT_XY_DIST = 0.04
MIN_WIDTH = 0.01
MAX_WIDTH = 0.095


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
    parser.add_argument("--target", choices=target_choices(), default=DEFAULT_OBJECT,
                        help="Registered target used when no instruction is supplied")
    parser.add_argument("--instruction", help="Natural-language pick/place instruction")
    parser.add_argument("--text", help="Override the GroundingDINO target phrase")
    parser.add_argument("--semantic-backend", choices=("auto", "rules", "openai"), default="auto",
                        help="Instruction parser; auto uses VLM when configured, else rules")
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
    parser.add_argument("--obj-x", type=float, help="Initial target x (set once before perception)")
    parser.add_argument("--obj-y", type=float, help="Initial target y (set once before perception)")
    parser.add_argument("--obj-yaw", type=float, default=0.0, help="Initial target yaw in radians")
    parser.add_argument("--drop-x", type=float)
    parser.add_argument("--drop-y", type=float)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-candidates", type=int, default=12,
                        help="Maximum grasp candidates to preflight through the full trajectory")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Maximum fresh perception/execution cycles")
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


def in_table_xy(point):
    return TABLE_X[0] <= point[0] <= TABLE_X[1] and TABLE_Y[0] <= point[1] <= TABLE_Y[1]


def select_detection(detections, depth, camera, log, *, labels=None,
                     target_phrase=None, target_key=None, spatial_reference="none"):
    """Select the semantically matching on-table box, then apply spatial reference."""
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
        if not (in_table_xy(point) and TABLE_Z[0] <= point[2] <= TABLE_Z[1]):
            log(f"  skip box {index} {label!r} conf={float(detections.confidence[index]):.3f}: "
                f"centre world={point.round(4).tolist()} outside workspace")
            continue
        candidates.append({
            "index": index,
            "point": point,
            "confidence": float(detections.confidence[index]),
            "match": semantic_match_score(label, target_phrase, target_key),
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
    if target_phrase or target_key:
        ordered = sorted(ordered, key=lambda index: (
            -semantic_match_score(labels[index] if labels is not None and index < len(labels) else "",
                                  target_phrase, target_key),
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


def perceive(env, args, weights_dir, diagnostics, log, *, predict_grasps=True):
    """Fresh RGB-D, text detection and segmentation; no simulator object pose."""
    import torch
    diagnostics.mkdir(parents=True, exist_ok=True)
    timings = {}
    # 1. Smoke render, also the RGB-D frame used for perception.
    started = time.time()
    frame = env.render(camera=VISION_CAMERA, width=RENDER_WIDTH, height=RENDER_HEIGHT)
    timings["render_s"] = time.time() - started
    rgb = frame["img"]
    depth = frame["depth"].astype(np.float32)
    camera = env.camera_parameters(VISION_CAMERA, RENDER_WIDTH, RENDER_HEIGHT)
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
        raise RuntimeError(f"GroundingDINO found nothing for caption {args.text!r}")
    detections, labels = detection
    semantic_plan = getattr(args, "semantic_plan", None)
    target_phrase = (semantic_plan.target_phrase if semantic_plan is not None else args.text)
    target_key = (semantic_plan.target_key if semantic_plan is not None else args.target)
    spatial_reference = (semantic_plan.spatial_reference if semantic_plan is not None else "none")
    box_index, box_world = select_detection(
        detections, depth, camera, log, labels=labels, target_phrase=target_phrase,
        target_key=target_key, spatial_reference=spatial_reference)
    detections_report["selected_index"] = box_index
    detections_report["selected_box"] = detections.xyxy[box_index].tolist()
    detections_report["selected_confidence"] = float(detections.confidence[box_index])
    detections_report["selected_label"] = labels[box_index]
    write_json(diagnostics / "detections.json", detections_report)
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
    camera_info = CameraInfo(RENDER_WIDTH, RENDER_HEIGHT, camera["fx"], camera["fy"],
                             camera["ppx"], camera["ppy"], 1.0)
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
    rotation = camera["T_world_camera"][:3, :3]
    translation = camera["T_world_camera"][:3, 3]
    world = cloud @ rotation.T + translation
    valid = depth > 0
    table = ((world[..., 0] >= TABLE_X[0]) & (world[..., 0] <= TABLE_X[1]) &
             (world[..., 1] >= TABLE_Y[0]) & (world[..., 1] <= TABLE_Y[1]))
    workspace = valid & (world[..., 2] >= TABLE_Z[0]) & (world[..., 2] <= TABLE_Z[1]) & table
    object_mask = mask & valid & (world[..., 2] >= OBJECT_Z[0]) & (world[..., 2] <= OBJECT_Z[1]) & table
    object_points = world[object_mask]
    if len(object_points) < 20:
        raise RuntimeError(f"SAM/workspace produced too few object points ({len(object_points)})")
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
        "target_key": target_key,
        "target_phrase": target_phrase,
        "spatial_reference": spatial_reference,
        "sam_report": sam_report,
    }
    write_json(diagnostics / "pointcloud_stats.json", pointcloud_report)

    observation = {"id": diagnostics.parent.name + "/" + diagnostics.name, "camera": camera, "rgb": rgb,
                   "centroid": centroid.copy(), "object_points": object_points.copy(),
                   "timings_s": timings}
    if not predict_grasps:
        return observation

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
        raise RuntimeError("GraspNet produced no grasps")
    np.save(diagnostics / "grasps_raw.npy", raw_array)
    grasps = grasps.nms()
    grasps.sort_by_score()
    nms_count = len(grasps)

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
        raise RuntimeError("Every GraspNet candidate was rejected by collision detection")

    observation["grasps"] = grasps
    write_json(diagnostics / "grasp_counts.json", {
        "raw_count": raw_count, "nms_count": nms_count,
        "collision_filtered": int(collision_mask.sum()), "remaining": len(grasps)})
    return observation


def choose_candidate(env, observation, args, diagnostics, log, home_pose, cycle=0, *,
                     exclude=None, stage="final"):
    from utils.piperx_vision_control import (
        TOP_DOWN, observation_pregrasp_pose, parallel_jaw_rotation_variants, plan_pick_place,
        rotation_angle_radians, top_down_grasp_candidates)
    diagnostics.mkdir(parents=True, exist_ok=True)
    grasps, object_points = observation["grasps"], observation["object_points"]
    candidates = []
    checked = 0
    excluded = set() if exclude is None else set(exclude)
    reference_rotation = np.asarray(env.tcp_pose[:3, :3], dtype=float)

    def preflight(entry):
        nonlocal checked
        if entry["rank"] in excluded or not all(entry["checks"].values()):
            return None
        checked += 1
        proposal_tcp = np.asarray(entry["world_tcp"], dtype=float)
        insertion = None
        if entry.get("source") == "graspnet" and args.grasp_insertion_depth > 0:
            insertion = bounded_grasp_insertion(
                proposal_tcp, args.grasp_insertion_depth, table_top_z=TABLE_TOP_Z)
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
            try:
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

    for rank in range(len(grasps)):
        if checked >= args.max_candidates:
            break
        if rank in excluded:
            continue
        grasp = grasps[rank]
        tcp = env.camera_grasp_to_world(grasp.translation, grasp.rotation_matrix,
                                       camera=VISION_CAMERA,
                                       T_world_camera=observation["camera"]["T_world_camera"])
        position, rotation = tcp[:3, 3], tcp[:3, :3]
        approach_cos = float(rotation[:, 2] @ np.array([0., 0., -1.]))
        distance = float(np.linalg.norm(object_points[:, :2] - position[None, :2], axis=1).min())
        checks = {"width": bool(MIN_WIDTH <= grasp.width <= MAX_WIDTH),
                  "approach": bool(approach_cos > APPROACH_COS),
                  "center": bool(GRASP_Z[0] <= position[2] <= GRASP_Z[1] and in_table_xy(position)),
                  "object": bool(distance < OBJECT_XY_DIST)}
        entry = {"source": "graspnet", "rank": rank, "stage": stage,
                 "score": float(grasp.score), "width": float(grasp.width),
                 "world_tcp": tcp.tolist(), "world_position": position.tolist(),
                 "camera_translation": grasp.translation.tolist(),
                 "camera_rotation": grasp.rotation_matrix.tolist(),
                 "checks": checks, "approach_cos": approach_cos, "object_xy_distance": distance}
        candidates.append(entry)
        chosen = preflight(entry)
        if chosen is not None:
            write_json(diagnostics / "grasps.json", {"candidates": candidates})
            write_json(diagnostics / "selected.json",
                       {key: value for key, value in chosen.items() if key != "plans"})
            log(f"Selected {stage} graspnet rank {rank} "
                f"jaw={chosen['jaw_variant']} ({chosen['jaw_branch']}, "
                f"{np.rad2deg(chosen['jaw_reference_angle_rad']):.1f} deg): "
                f"all {len(chosen['plans'])} trajectory segments preflighted")
            return chosen
        if checked >= args.max_candidates:
            break

    # Keep the simple vertical grasp only as a fallback.  GraspNet remains the
    # primary proposal source; this path is used when the six-DoF candidates
    # cannot produce a fully reachable path.
    analytic = top_down_grasp_candidates(object_points, table_top_z=TABLE_TOP_Z)
    if analytic:
        offset = cycle % len(analytic)
        analytic = analytic[offset:] + analytic[:offset]
    for entry in analytic:
        if checked >= args.max_candidates:
            break
        if entry["rank"] in excluded:
            continue
        position = entry["world_position"]
        entry["checks"]["center"] = bool(
            GRASP_Z[0] <= position[2] <= GRASP_Z[1] and in_table_xy(position))
        candidates.append(entry)
        chosen = preflight(entry)
        if chosen is not None:
            write_json(diagnostics / "grasps.json", {"candidates": candidates})
            write_json(diagnostics / "selected.json",
                       {key: value for key, value in chosen.items() if key != "plans"})
            log(f"Selected {stage} {entry['source']} rank {entry['rank']} "
                f"jaw={chosen['jaw_variant']} ({chosen['jaw_branch']}): "
                f"all {len(chosen['plans'])} trajectory segments preflighted")
            return chosen
    write_json(diagnostics / "grasps.json", {"candidates": candidates})
    raise RuntimeError(f"No fully reachable path among {checked} filtered candidates")


def apply_semantic_instruction(env, args, diagnostics, log):
    """Resolve the operator instruction before the first perception cycle."""
    if args.instruction:
        planner = build_semantic_planner(
            args.semantic_backend, base_url=args.vlm_base_url, model=args.vlm_model,
            timeout=args.vlm_timeout, max_tokens=args.vlm_max_tokens, log=log)
        frame = env.render(camera=VISION_CAMERA, width=RENDER_WIDTH, height=RENDER_HEIGHT)
        save_bgr(diagnostics / "semantic_rgb.png", frame["img"])
        plan = planner.plan(frame["img"], args.instruction)
        spec = resolve_target(plan.target_key or plan.target_phrase or args.instruction)
        if spec is None:
            raise RuntimeError(
                f"VLM phrase {plan.target_phrase!r} does not match a scene object; "
                f"use one of {', '.join(target_choices())} or pass --target explicitly")
        env.set_target(spec)
        args.target = spec.key
        if args.text is None:
            args.text = plan.dino_prompt
        args.semantic_plan = plan
        args.drop_x, args.drop_y = drop_xy_for_plan(
            plan, drop_x=args.drop_x, drop_y=args.drop_y)
        write_json(diagnostics / "semantic_plan.json", {
            "planner": planner.name,
            "plan": plan.as_dict(),
            "resolved_target": spec.key,
        })
        log(f"Semantic plan ({planner.name}): {plan.as_dict()}")
        log(f"DINO prompt: {args.text!r}; drop_xy={(args.drop_x, args.drop_y)}")
        return plan

    spec = resolve_target(args.target)
    if spec is None:
        raise ValueError(f"Unknown target {args.target!r}")
    env.set_target(spec)
    if args.text is None:
        args.text = spec.prompt
    if args.drop_x is None:
        args.drop_x = DEFAULT_DROP_XY[0]
    if args.drop_y is None:
        args.drop_y = DEFAULT_DROP_XY[1]
    log(f"Target: {spec.key}; DINO prompt: {args.text!r}; "
        f"drop_xy={(args.drop_x, args.drop_y)}")
    return None


def run(args):
    import mujoco
    import torch
    from manipulator_grasp.env.piperx_grasp_env import PiperXGraspEnv
    from utils.piperx_vision_control import run_closed_loop
    from utils.piperx_grasp_display import GraspPoseDisplay

    if not 1 <= args.max_attempts <= 5 or args.max_candidates < 1:
        raise ValueError("max-attempts must be 1..5 and max-candidates must be positive")
    if not np.isfinite(args.grasp_preview_seconds) or args.grasp_preview_seconds < 0:
        raise ValueError("grasp-preview-seconds must be finite and nonnegative")
    if not np.isfinite(args.grasp_insertion_depth) or args.grasp_insertion_depth < 0:
        raise ValueError("grasp-insertion-depth must be finite and nonnegative")
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
    log = Logger(diagnostics / "run.log")
    log(f"Diagnostics: {diagnostics}")
    log(f"Args: {vars(args)}")
    weights_dir = Path(load_config("cv.yaml").get("weights_dir", "weights"))
    if not weights_dir.is_absolute():
        weights_dir = ROOT / weights_dir

    with PiperXGraspEnv(viewer=not args.headless, realtime=not args.fast,
                       render_size=(RENDER_WIDTH, RENDER_HEIGHT),
                       target=args.target) as env, ExitStack() as resources:
        grasp_display = resources.enter_context(GraspPoseDisplay(env))
        preview = None
        if not args.headless and not args.no_camera_preview:
            from utils.piperx_camera_preview import WristCameraPreview
            preview = resources.enter_context(WristCameraPreview(env, grasp_display=grasp_display))
            physics_step = env.step
            def preview_step():
                physics_step()
                preview.update()
            env.step = preview_step
        env.wait(0.4)
        apply_semantic_instruction(env, args, diagnostics, log)
        # Initial conditions are allowed before the first perception frame.
        if args.obj_x is not None or args.obj_y is not None or args.obj_yaw:
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
        initial_truth = env.object_position.copy()
        unexpected_contacts = set()
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
        def observe(cycle, phase):
            grasp_phase = phase in ("pickup", "pregrasp") or phase.startswith("pregrasp_retry")
            if grasp_phase:
                grasp_display.clear()
                if preview is not None:
                    preview.update(force=True)
            result = perceive(env, args, weights_dir, diagnostics / f"cycle_{cycle:02d}" / phase,
                              log, predict_grasps=grasp_phase)
            observations.append({"id": result["id"], "phase": phase,
                                 "centroid": result["centroid"],
                                 "camera": result["camera"],
                                 "timings_s": result["timings_s"]})
            return result
        def propose(observation, cycle, home_pose, *, stage="final", exclude=None):
            folder = diagnostics / f"cycle_{cycle:02d}" / stage
            selected = choose_candidate(env, observation, args, folder, log, home_pose,
                                        cycle=cycle, exclude=exclude, stage=stage)
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
            if not args.headless:
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
            for camera_height in (0.955, 0.935, 0.915):
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

        result = run_closed_loop(
            env, observe, propose, drop_xy=(args.drop_x, args.drop_y),
            max_attempts=args.max_attempts, log=log,
            release_first=args.test_release_first,
            two_stage=True, propose_pregrasp=propose_pregrasp,
            propose_final=propose_final, support_check=True,
            max_support_trials=args.max_candidates,
            reobserve_final=lambda retry_cycle: observe(retry_cycle, "pregrasp_retry"))
        # Independent acceptance is evaluated only after the control policy ends.
        final_truth = env.object_position.copy()
        lift = float(env.max_object_height - initial_truth[2])
        xy_error = float(np.linalg.norm(final_truth[:2] - [args.drop_x, args.drop_y]))
        z_error = float(abs(final_truth[2] - initial_truth[2]))
        truth_success = bool(lift >= 0.04 and xy_error < 0.04 and z_error < 0.015
                             and not unexpected_contacts and not env.data.warning.number.any())
        result.update(success=bool(result["policy_success"] and truth_success),
                      target=env.target_key, target_label=env.target_spec.label,
                      target_body=env.target_body, instruction=args.instruction,
                      semantic_plan=(args.semantic_plan.as_dict()
                                     if getattr(args, "semantic_plan", None) is not None else None),
                      caption=args.text, drop_xy=[args.drop_x, args.drop_y],
                      grasp_visualization={"frame": "world", "selected_pickups": displayed_grasps,
                                           "viewer_enabled": not args.headless,
                                           "rgb_overlay": preview is not None},
                      initial_position=initial_truth, final_position=final_truth,
                      lift_height_m=lift, placement_error_m=xy_error, placement_z_error_m=z_error,
                      bilateral_contact=any(a.get("lift_sensor", {}).get("held", False) for a in result["attempts"]),
                      observations=observations, simulation_time_s=float(env.data.time),
                      truth_acceptance={"success": truth_success, "unexpected_contacts": sorted(unexpected_contacts),
                                        "mujoco_warnings": env.data.warning.number.tolist(),
                                        "used_for_control": False})
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
        if args.hold and not args.headless:
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
