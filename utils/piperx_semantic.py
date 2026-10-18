"""Semantic object registry for the PiperX tabletop scene."""
from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCENE_ASSET_DIR = ROOT / "manipulator_grasp/assets/agilex_piper_x"
TABLE_TOP_Z = 0.740
OBJECT_CLEARANCE_Z = 0.002


@dataclass(frozen=True)
class SemanticObject:
    key: str
    body: str
    label: str
    prompt: str
    aliases: tuple[str, ...]
    source: str
    visual_file: str
    texture_file: str
    mass_kg: float
    position_xy: tuple[float, float]
    yaw_deg: float
    description_zh: str
    refine: str | None = None
    grasp_yaw_deg: float = 0.0


OBJECTS: dict[str, SemanticObject] = {
    "banana": SemanticObject(
        key="banana",
        body="ycb_banana",
        label="banana",
        prompt="banana .",
        aliases=("banana", "香蕉"),
        source="YcbBanana",
        visual_file="textured_simple_reoriented.obj",
        texture_file="texture_map.png",
        mass_kg=0.066,
        position_xy=(0.36, -0.12),
        yaw_deg=0.0,
        description_zh="香蕉",
        grasp_yaw_deg=45.0,
    ),
    "strawberry": SemanticObject(
        key="strawberry",
        body="ycb_strawberry",
        label="strawberry",
        prompt="strawberry .",
        aliases=("strawberry", "草莓"),
        source="YcbStrawberry",
        visual_file="textured_reoriented.obj",
        texture_file="texture_map.png",
        mass_kg=0.018,
        position_xy=(0.28, -0.27),
        yaw_deg=0.0,
        description_zh="草莓",
    ),
    "pear": SemanticObject(
        key="pear",
        body="ycb_pear",
        label="pear",
        prompt="pear .",
        aliases=("pear", "梨", "雪梨"),
        source="YcbPear",
        visual_file="textured_reoriented.obj",
        texture_file="texture_map.png",
        mass_kg=0.049,
        position_xy=(0.38, -0.23),
        yaw_deg=0.0,
        description_zh="梨",
    ),
    "tennis_ball": SemanticObject(
        key="tennis_ball",
        body="ycb_tennis_ball",
        label="tennis ball",
        prompt="tennis ball .",
        aliases=("tennis ball", "tennis", "网球"),
        source="YcbTennisBall",
        visual_file="textured_reoriented.obj",
        texture_file="texture_map.png",
        mass_kg=0.058,
        position_xy=(0.30, 0.13),
        yaw_deg=0.0,
        description_zh="网球",
        grasp_yaw_deg=90.0,
    ),
    "tomato_soup_can": SemanticObject(
        key="tomato_soup_can",
        body="ycb_tomato_soup_can",
        label="tomato soup can",
        prompt="tomato soup can .",
        aliases=("tomato soup can", "tomato soup", "soup can",
                 "番茄汤罐头", "番茄罐头", "汤罐头", "罐头"),
        source="YcbTomatoSoupCan",
        visual_file="textured_simple_reoriented.obj",
        texture_file="texture_map.png",
        mass_kg=0.349,
        position_xy=(0.34, 0.02),
        yaw_deg=0.0,
        description_zh="番茄汤罐头",
    ),
}

DEFAULT_OBJECT = "banana"


def target_choices():
    return tuple(OBJECTS)


def object_body_names():
    return tuple(spec.body for spec in OBJECTS.values())


def resolve_target(text):
    """Resolve a key, body name, alias or natural-language phrase."""
    if not isinstance(text, str) or not text.strip():
        return None
    normalized = text.strip().casefold()
    for spec in OBJECTS.values():
        if normalized in (spec.key.casefold(), spec.body.casefold(), spec.label.casefold()):
            return spec
    matches = []
    for spec in OBJECTS.values():
        for alias in spec.aliases:
            candidate = alias.casefold()
            if candidate and candidate in normalized:
                matches.append((len(candidate), spec))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def semantic_match_score(label, target_phrase=None, target_key=None):
    """Score a detector label against the requested semantic target."""
    label_text = str(label or "").strip().casefold().rstrip(". ")
    if not label_text:
        return 0
    spec = resolve_target(target_key or target_phrase or "")
    aliases = ((spec.key, spec.body, spec.label, *spec.aliases) if spec is not None else ())
    request = str(target_phrase or (spec.label if spec is not None else "")).casefold().rstrip(". ")
    score = 0
    for alias in aliases:
        alias = str(alias).casefold().strip()
        if alias and (alias == label_text or alias in label_text or label_text in request):
            score = max(score, 4 + len(alias))
    if not score and request:
        label_words = set(re.findall(r"[a-z0-9]+", label_text))
        request_words = set(re.findall(r"[a-z0-9]+", request))
        score = 2 * len(label_words & request_words)
    return score
