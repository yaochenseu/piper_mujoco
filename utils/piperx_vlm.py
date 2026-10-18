"""Semantic instruction planning for open-vocabulary visual grounding."""
from dataclasses import asdict, dataclass
import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from utils.piperx_semantic import OBJECTS, resolve_target


SYSTEM_PROMPT = """You are the semantic planner for a tabletop robot.
GroundingDINO accepts arbitrary short English visual noun phrases; it does not
use a fixed class list. Convert the user's Chinese or English instruction into
one concise English visual target phrase. Include colour, shape, or material
only when useful for disambiguation. Do not put actions, pronouns, or spatial
relations in target_phrase.

Return one JSON object and nothing else:
{
  "target_phrase": "short English noun phrase",
  "spatial_reference": "leftmost|rightmost|closest|farthest|none",
  "action": "pick_place|pick",
  "destination": "left|right|front|back|center"
}
"""

DEFAULT_DROP_XY = (0.42, 0.16)
DESTINATION_XY = {
    "left": (0.36, 0.28),
    "right": (0.22, -0.34),
    "front": (0.40, 0.24),
    "back": (0.30, 0.24),
    "center": DEFAULT_DROP_XY,
}


class VLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticPlan:
    instruction: str
    target_phrase: str
    target_key: str | None
    target_label: str
    action: str
    destination: str
    spatial_reference: str
    dino_prompt: str
    backend: str
    raw: dict | None = None

    def as_dict(self):
        return asdict(self)


def normalize_phrase(value):
    if not isinstance(value, str):
        raise VLMError("VLM target phrase must be a string")
    phrase = value.strip().strip("`\"'").rstrip(". ").casefold()
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    if not phrase:
        raise VLMError("VLM returned an empty target phrase")
    return phrase


def dino_prompt(phrase):
    return f"{normalize_phrase(phrase)} ."


_DIRECTION_WORDS = (
    ("中间", "center"), ("中央", "center"), ("center", "center"), ("centre", "center"),
    ("左", "left"), ("left", "left"),
    ("右", "right"), ("right", "right"),
    ("前", "front"), ("front", "front"),
    ("后", "back"), ("back", "back"),
)


def _direction_in(text):
    """Return the last directional word and its canonical name in text."""
    text = str(text or "").casefold()
    matches = [(text.rfind(word), word, canonical)
               for word, canonical in _DIRECTION_WORDS if word in text]
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], len(item[1])))[2]


def normalize_destination(value, instruction=""):
    explicit = _direction_in(value)
    if explicit is not None:
        return explicit
    # Chinese instructions commonly mention the target side first and the
    # destination last ("把左边的香蕉放到右边").  The destination is therefore
    # the final directional word, not the first one.
    inferred = _direction_in(instruction)
    return inferred if inferred is not None else "center"


def infer_spatial_reference(instruction):
    """Infer a left/right target selector without confusing the drop side."""
    text = str(instruction or "").casefold()
    for pattern, canonical in (
            (r"最\s*左", "leftmost"), (r"最\s*右", "rightmost"),
            (r"最\s*近", "closest"), (r"最\s*远", "farthest"),
            (r"leftmost", "leftmost"), (r"rightmost", "rightmost")):
        if re.search(pattern, text):
            return canonical
    # "左边的香蕉" and "left banana" select a target; "放到右边" does not
    # contain 的/那个 and is handled as a destination instead.
    if re.search(r"(?:左|右)(?:侧|边)?(?:的|那个|那只|那件|那)", text):
        return "leftmost" if re.search(r"左(?:侧|边)?(?:的|那个|那只|那件|那)", text) else "rightmost"
    object_words = r"(?:banana|strawberry|pear|tennis|ball|can|object|item|one)"
    match = re.search(rf"\b(left|right)\s+{object_words}\b", text)
    if match:
        return "leftmost" if match.group(1) == "left" else "rightmost"
    return "none"


def normalize_spatial_reference(value, instruction=""):
    text = str(value or "").strip().casefold()
    for word, canonical in (
            ("leftmost", "leftmost"), ("最左", "leftmost"),
            ("rightmost", "rightmost"), ("最右", "rightmost"),
            ("closest", "closest"), ("最近", "closest"),
            ("farthest", "farthest"), ("最远", "farthest"),
            ("left", "leftmost"), ("左", "leftmost"),
            ("right", "rightmost"), ("右", "rightmost")):
        if word in text:
            return canonical
    return infer_spatial_reference(instruction)


def extract_json_object(text):
    if not isinstance(text, str):
        raise VLMError("VLM response is not text")
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise VLMError("VLM response contains no JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as error:
            raise VLMError(f"Invalid VLM JSON: {error}") from error
    if not isinstance(value, dict):
        raise VLMError("VLM response JSON must be an object")
    return value


def plan_from_payload(payload, instruction, backend):
    phrase = payload.get("target_phrase") or payload.get("target") or payload.get("object")
    phrase = normalize_phrase(phrase)
    requested_key = payload.get("target_key")
    spec = resolve_target(requested_key or phrase or instruction)
    if spec is None:
        spec = resolve_target(instruction)
    target_key = spec.key if spec is not None else None
    target_label = spec.label if spec is not None else phrase
    action = str(payload.get("action") or "pick_place").strip().casefold()
    if action not in ("pick_place", "pick"):
        action = "pick_place"
    destination = normalize_destination(payload.get("destination"), instruction)
    spatial_reference = normalize_spatial_reference(
        payload.get("spatial_reference"), instruction)
    return SemanticPlan(
        instruction=instruction,
        target_phrase=phrase,
        target_key=target_key,
        target_label=target_label,
        action=action,
        destination=destination,
        spatial_reference=spatial_reference,
        dino_prompt=dino_prompt(phrase),
        backend=backend,
        raw=payload,
    )


class RuleSemanticPlanner:
    name = "rules"

    def plan(self, rgb, instruction, *, scene_context=None):
        spec = resolve_target(instruction)
        if spec is None:
            choices = ", ".join(OBJECTS)
            raise VLMError(f"No registered object in instruction {instruction!r}; choose one of: {choices}")
        payload = {
            "target_phrase": spec.label,
            "target_key": spec.key,
            "spatial_reference": infer_spatial_reference(instruction),
            "action": "pick_place",
            "destination": normalize_destination(None, instruction),
        }
        return plan_from_payload(payload, instruction, self.name)


def _image_data_url(rgb):
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB/RGBA image")
    buffer = io.BytesIO()
    Image.fromarray(image[..., :3]).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatiblePlanner:
    """Call any OpenAI-compatible multimodal chat-completions endpoint."""

    name = "openai-compatible"

    def __init__(self, base_url, model, *, api_key=None, timeout=60.0, max_tokens=2048):
        if not base_url or not model:
            raise ValueError("VLM base URL and model are required")
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            self.endpoint = base
        elif base.endswith("/v1") or base.endswith("/beta"):
            self.endpoint = base + "/chat/completions"
        else:
            # OpenAI-compatible SDKs treat base_url as the API root.  DeepSeek
            # documents https://api.deepseek.com (without /v1) and serves the
            # route at /chat/completions, while local servers commonly expose
            # the same route under /v1/chat/completions.
            self.endpoint = base + "/chat/completions"
        self.model = model
        self.api_key = api_key or os.environ.get("PIPERX_VLM_API_KEY", "")
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)

    def plan(self, rgb, instruction, *, scene_context=None):
        context = ""
        if scene_context:
            context = f"\nActive simulation objects (reference only): {scene_context}\n"
        body = {
            "model": self.model,
            "temperature": 0,
            # Reasoning models spend tokens on reasoning_content before they
            # emit the JSON answer.  A small budget (for example 8 tokens) can
            # therefore produce an empty content field even on a valid request.
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(rgb)}},
                    {"type": "text", "text": f"{instruction}{context}"},
                ]},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise VLMError(f"VLM request failed: {error}") from error
        try:
            choice = payload["choices"][0]
            message = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VLMError("VLM response is missing choices[0].message.content") from error
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        if not isinstance(message, str) or not message.strip():
            finish_reason = choice.get("finish_reason")
            raise VLMError(
                "VLM returned empty content "
                f"(finish_reason={finish_reason!r}); increase max_tokens")
        semantic = extract_json_object(message)
        return plan_from_payload(semantic, instruction, self.name)


class FallbackSemanticPlanner:
    def __init__(self, primary, fallback, *, log=print):
        self.primary = primary
        self.fallback = fallback
        self.log = log
        self.name = f"auto({primary.name}->{fallback.name})"

    def plan(self, rgb, instruction, *, scene_context=None):
        try:
            return self.primary.plan(rgb, instruction, scene_context=scene_context)
        except Exception as error:
            self.log(f"VLM planner failed ({error}); using {self.fallback.name} fallback")
            return self.fallback.plan(rgb, instruction, scene_context=scene_context)


def build_semantic_planner(backend="auto", *, base_url=None, model=None, timeout=60.0,
                           max_tokens=2048, log=print):
    rules = RuleSemanticPlanner()
    base_url = base_url or os.environ.get("PIPERX_VLM_BASE_URL", "")
    model = model or os.environ.get("PIPERX_VLM_MODEL", "")
    if backend == "rules":
        return rules
    if backend not in ("auto", "openai"):
        raise ValueError("semantic backend must be auto, rules, or openai")
    if not base_url or not model:
        if backend == "openai":
            raise ValueError("--vlm-base-url and --vlm-model are required for the openai backend")
        return rules
    remote = OpenAICompatiblePlanner(base_url, model, timeout=timeout, max_tokens=max_tokens)
    return FallbackSemanticPlanner(remote, rules, log=log) if backend == "auto" else remote


def drop_xy_for_plan(plan, *, drop_x=None, drop_y=None):
    x, y = DESTINATION_XY.get(plan.destination, DEFAULT_DROP_XY)
    return (float(x if drop_x is None else drop_x),
            float(y if drop_y is None else drop_y))
