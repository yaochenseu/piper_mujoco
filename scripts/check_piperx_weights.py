"""Load downloaded pretrained models and run GPU inference on local sample data."""
import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Verification must not silently fetch missing Hugging Face files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=["graspnet", "groundingdino", "sam", "yolo"])
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()
    os.chdir(ROOT)
    import cv2
    import numpy as np
    import torch

    assert torch.cuda.is_available(), "CUDA GPU is required for this verification"
    image = cv2.imread(str(ROOT / "external/GroundingDINO/.asset/cat_dog.jpeg"))
    assert image is not None, "GroundingDINO's local sample image is missing"
    weights = ROOT / "weights"
    results = {"python": sys.executable, "gpu": torch.cuda.get_device_name(0), "checks": {}}
    context = {}

    def file_info(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": digest.hexdigest()}

    def graspnet():
        from utils.graspnet_runner import load_graspnet
        from graspnet import pred_decode
        path = weights / "logs/log_rs/checkpoint-rs.tar"
        model = load_graspnet(str(path))  # strict state-dict loading
        torch.manual_seed(7)
        with torch.inference_mode():
            outputs = model({"point_clouds": torch.rand(1, 3000, 3, device="cuda") * 0.1})
            grasps = pred_decode(outputs)[0]
        assert grasps.ndim == 2 and grasps.shape[1] == 17
        assert torch.isfinite(grasps).all()
        return {"checkpoint": file_info(path), "synthetic_cloud_grasp_shape": list(grasps.shape)}

    def groundingdino():
        from groundingdino.util.inference import Model
        path = weights / "groundingdino_swint_ogc.pth"
        model = Model(model_config_path=str(weights / "groundingdino_swint_ogc.py"),
                      model_checkpoint_path=str(path), device="cuda")
        detections, labels = model.predict_with_caption(
            image=image, caption="cat . dog .", box_threshold=0.25, text_threshold=0.25)
        assert len(detections) > 0, "No objects detected on the official cat/dog sample"
        context["bbox"] = detections.xyxy[int(np.argmax(detections.confidence))].tolist()
        return {"checkpoint": file_info(path), "detections": len(detections), "labels": labels,
                "bert": "weights/bert-base-uncased (offline)"}

    def sam():
        from ultralytics import SAM
        path = weights / "sam_b.pt"
        h, w = image.shape[:2]
        box = context.get("bbox", [0.05 * w, 0.05 * h, 0.95 * w, 0.95 * h])
        model = SAM(str(path))
        predictions = model.predict(source=image, bboxes=[box], device=0, verbose=False, save=False)
        masks = predictions[0].masks
        assert masks is not None and masks.data.any(), "SAM returned an empty mask"
        return {"checkpoint": file_info(path), "mask_shape": list(masks.data.shape),
                "foreground_pixels": int(masks.data.sum())}

    def yolo():
        import clip
        from ultralytics import YOLO
        path = weights / "yolov8s-world.pt"
        clip_path = weights / "clip/ViT-B-32.pt"
        assert clip_path.is_file(), "Local CLIP text-encoder checkpoint is missing"
        model = YOLO(str(path))
        model.model.clip_model = clip.load(str(clip_path), device="cuda")[0]
        model.set_classes(["cat", "dog"])
        predictions = model.predict(source=image, device=0, verbose=False, save=False)
        assert len(predictions[0].boxes) > 0, "YOLO-World returned no cats/dogs"
        return {"checkpoint": file_info(path), "clip_checkpoint": file_info(clip_path),
                "detections": len(predictions[0].boxes), "classes": predictions[0].names}

    for name, function in [("graspnet", graspnet), ("groundingdino", groundingdino),
                           ("sam", sam), ("yolo", yolo)]:
        if args.only and name not in args.only:
            continue
        print(f"Checking pretrained {name}...", file=sys.stderr, flush=True)
        try:
            with redirect_stdout(sys.stderr):
                details = function()
                torch.cuda.synchronize()
            results["checks"][name] = {"ok": True, "details": details}
        except Exception:
            results["checks"][name] = {"ok": False, "error": traceback.format_exc()}
        gc.collect()
        torch.cuda.empty_cache()
    payload = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if all(item["ok"] for item in results["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
