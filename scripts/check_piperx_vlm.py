"""Send one real image-to-JSON planning request to the configured VLM."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from dotenv import load_dotenv
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from utils.piperx_vlm import OpenAICompatiblePlanner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="docs/screenshots/detection_visualization.jpg",
                        help="Image containing the tabletop objects")
    parser.add_argument("--instruction", default="拿香蕉放到右边",
                        help="Natural-language instruction sent with the image")
    parser.add_argument("--base-url", default=os.environ.get("PIPERX_VLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("PIPERX_VLM_MODEL", ""))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path,
                        help="Optional JSON result path; the API key is never written")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image if Path(args.image).is_absolute() else ROOT / args.image
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    planner = OpenAICompatiblePlanner(
        args.base_url, args.model, timeout=args.timeout, max_tokens=args.max_tokens)
    print(f"Endpoint: {planner.endpoint}")
    print(f"Model: {planner.model}")
    print(f"API key configured: {bool(planner.api_key)}")
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    plan = planner.plan(rgb, args.instruction)
    result = {
        "ok": True,
        "image": str(image_path),
        "instruction": args.instruction,
        "planner": plan.backend,
        "target_phrase": plan.target_phrase,
        "target_label": plan.target_label,
        "spatial_reference": plan.spatial_reference,
        "action": plan.action,
        "destination": plan.destination,
        "dino_prompt": plan.dino_prompt,
        "raw": plan.raw,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
