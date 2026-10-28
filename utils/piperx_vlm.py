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


SYSTEM_PROMPT = """You are the semantic planner and visual verifier for a
tabletop robot.

GroundingDINO accepts arbitrary short English visual noun phrases; it does not
use a fixed class list. Convert the user's Chinese or English instruction into
one concise English visual target phrase. Include colour, shape, or material
only when useful for disambiguation. Do not put actions, pronouns, or spatial
relations in target_phrase.

Look at the supplied wrist-camera image. If a detector box is supplied, it is
outlined in green and its label is shown. Verify the target against the image:
- target_visible: true only if the instructed object is actually visible;
- selected_matches_target: true/false if a detector box is supplied, or null
  if there is no box. Do not call a different object the target just because
  it is green or prominent;
- search_yaw_deg: if the target is not visible, or the visible view is not
  suitable, give a signed J1 yaw in degrees, **relative to this current view**,
  that would bring the target into a better view. Positive means rotate left /
  counter-clockwise viewed from above; negative means rotate right. Use 10..60
  degrees; use 0 when the current view is already good or no direction is known.
- search_reason: a few words explaining visibility/rotation.

Return one JSON object and nothing else:
{
  "target_phrase": "short English noun phrase",
  "target_visible": true,
  "selected_matches_target": null,
  "search_yaw_deg": 0,
  "search_reason": "visible and centred",
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
    target_label: str
    action: str
    destination: str
    spatial_reference: str
    dino_prompt: str
    backend: str
    raw: dict | None = None
    target_visible: bool = True
    selected_matches_target: bool | None = None
    search_yaw_deg: float = 0.0
    search_reason: str = ""

    def as_dict(self):
        return asdict(self)


def _parse_bool(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in ("true", "yes", "1", "visible", "match", "matches"):
        return True
    if text in ("false", "no", "0", "not_visible", "missing", "mismatch"):
        return False
    return default


def _parse_search_yaw(value):
    try:
        yaw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(yaw):
        return 0.0
    return float(np.clip(yaw, -120.0, 120.0))


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
    match = re.search(r"\b(left|right)\s+([a-z0-9]+)\b", text)
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
    action = str(payload.get("action") or "pick_place").strip().casefold()
    if action not in ("pick_place", "pick"):
        action = "pick_place"
    destination = normalize_destination(payload.get("destination"), instruction)
    spatial_reference = normalize_spatial_reference(
        payload.get("spatial_reference"), instruction)
    target_visible = _parse_bool(payload.get("target_visible"), True)
    selected_matches = _parse_bool(
        payload.get("selected_matches_target"), None)
    search_yaw = _parse_search_yaw(payload.get("search_yaw_deg"))
    search_reason = str(payload.get("search_reason") or "").strip()
    return SemanticPlan(
        instruction=instruction,
        target_phrase=phrase,
        target_label=phrase,
        action=action,
        destination=destination,
        spatial_reference=spatial_reference,
        dino_prompt=dino_prompt(phrase),
        backend=backend,
        raw=payload,
        target_visible=bool(target_visible),
        selected_matches_target=selected_matches,
        search_yaw_deg=search_yaw,
        search_reason=search_reason,
    )


_PLACEMENT_TAIL = re.compile(
    r"(?:放到|放在|搬到|移到|摆到|送到|丢到|置于|放入).*$",
    re.IGNORECASE)
_ENGLISH_PLACEMENT = re.compile(
    r"\b(?:put|place|move|bring)\s+(?:the\s+)?(.+?)\s+"
    r"(?:on|to|at|into|in)\s+(?:the\s+)?(?:left|right|front|back|center)\b",
    re.IGNORECASE)
_ACTION_PREFIX = re.compile(
    r"^(?:请|帮我|帮忙|给我|我想要|我要|想要|想|要|把|将|"
    r"pick\s+up|pick|grab|grasp|take|lift|get|fetch|"
    r"抓取|抓住|抓起|拿起|拿取|拿到|拿|取|拾取|捡起|夹起|夹住|夹|捏起|挑选|找出)\s*",
    re.IGNORECASE)
_TARGET_SIDE = re.compile(r"(?:最)?(?:左|右)(?:侧|边)?(?:的|那个|那只|那件|那)?")
_ENGLISH_SIDE = re.compile(r"\b(?:leftmost|rightmost|left|right)\b", re.IGNORECASE)
_DEMONSTRATIVE = re.compile(r"(?:那个|那只|那件|这个|这只|这件|一个|一只|一台|一把)")
_ENGLISH_ARTICLE = re.compile(r"\b(?:the|a|an)\b", re.IGNORECASE)


def _rule_target_phrase(instruction):
    text = str(instruction or "").strip()
    match = _ENGLISH_PLACEMENT.search(text)
    if match:
        text = match.group(1)
    else:
        text = _PLACEMENT_TAIL.sub("", text)
    while text:
        stripped = _ACTION_PREFIX.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    text = _TARGET_SIDE.sub(" ", text)
    text = _ENGLISH_SIDE.sub(" ", text)
    text = _DEMONSTRATIVE.sub(" ", text)
    text = _ENGLISH_ARTICLE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,，。!！?？")
    text = text.strip("的").strip()
    return text


class RuleSemanticPlanner:
    name = "rules"

    def plan(self, rgb, instruction, *, scene_context=None, detection=None):
        phrase = _rule_target_phrase(instruction)
        if not phrase:
            raise VLMError(
                f"No target phrase found in instruction {instruction!r}; "
                "pass --text or use the openai semantic backend")
        payload = {
            "target_phrase": phrase,
            "spatial_reference": infer_spatial_reference(instruction),
            "action": "pick_place",
            "destination": normalize_destination(None, instruction),
            "target_visible": True,
            "selected_matches_target": None,
            "search_yaw_deg": 0,
            "search_reason": "rules backend cannot verify the image",
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

    def plan(self, rgb, instruction, *, scene_context=None, detection=None):
        context = ""
        if scene_context:
            context = f"\nActive simulation objects (reference only): {scene_context}\n"
        if detection:
            box = detection.get("box")
            label = detection.get("label")
            confidence = detection.get("confidence")
            detection_text = (
                f"\nDetector box (green): {box}; label={label!r}; "
                f"confidence={confidence}. Is this the instructed target?")
        else:
            detection_text = "\nNo detector box was produced for this image."
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
                    {"type": "text",
                     "text": f"{instruction}{context}{detection_text}"},
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

    def plan(self, rgb, instruction, *, scene_context=None, detection=None):
        try:
            return self.primary.plan(
                rgb, instruction, scene_context=scene_context, detection=detection)
        except Exception as error:
            self.log(f"VLM planner failed ({error}); using {self.fallback.name} fallback")
            return self.fallback.plan(
                rgb, instruction, scene_context=scene_context, detection=detection)


def build_semantic_planner(backend="openai", *, base_url=None, model=None, timeout=60.0,
                           max_tokens=2048, log=print):
    rules = RuleSemanticPlanner()
    base_url = base_url or os.environ.get("PIPERX_VLM_BASE_URL", "")
    model = model or os.environ.get("PIPERX_VLM_MODEL", "")
    if backend == "rules":
        return rules
    if backend not in ("auto", "openai"):
        raise ValueError("semantic backend must be auto, rules, or openai")
    if not base_url or not model:
        raise ValueError(
            "The openai semantic backend requires PIPERX_VLM_BASE_URL and "
            "PIPERX_VLM_MODEL (or --vlm-base-url/--vlm-model). Set both "
            "environment variables, pass --text to skip instruction parsing, "
            "or use --semantic-backend rules")
    remote = OpenAICompatiblePlanner(base_url, model, timeout=timeout, max_tokens=max_tokens)
    return FallbackSemanticPlanner(remote, rules, log=log) if backend == "auto" else remote


def semantic_match_score(label, target_phrase=None):
    """Score a detector label against an open-vocabulary target phrase."""
    label_text = str(label or "").strip().casefold().rstrip(". ")
    request = str(target_phrase or "").strip().casefold().rstrip(". ")
    if not label_text:
        return 0
    if request and (label_text == request or label_text in request or request in label_text):
        return 4 + len(request)
    label_words = set(re.findall(r"[a-z0-9]+", label_text))
    request_words = set(re.findall(r"[a-z0-9]+", request))
    return 2 * len(label_words & request_words)


def drop_xy_for_plan(plan, *, drop_x=None, drop_y=None):
    x, y = DESTINATION_XY.get(plan.destination, DEFAULT_DROP_XY)
    return (float(x if drop_x is None else drop_x),
            float(y if drop_y is None else drop_y))
