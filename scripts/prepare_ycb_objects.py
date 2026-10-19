"""Generate MuJoCo assets for the five semantic tabletop objects."""
from pathlib import Path
import math
import shutil
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import load_config  # noqa: E402


SCENE_CONFIG = load_config("piperx_scene.yaml")
OBJECTS = SCENE_CONFIG["objects"]
TABLE_TOP_Z = float(SCENE_CONFIG["table_top_z"])
OBJECT_CLEARANCE_Z = float(SCENE_CONFIG["object_clearance_z"])
SCENE_ASSET_DIR = ROOT / "manipulator_grasp/assets/agilex_piper_x"
SOURCE_ROOT = ROOT / "pybullet-object-models/pybullet_object_models/ycb_objects"
OUTPUT_ROOT = SCENE_ASSET_DIR / "ycb"
MAX_TEXTURE_SIZE = 1024


def obj_groups(path):
    """Return shared vertices and the faces belonging to each OBJ group.

    V-HACD files use one global vertex buffer while each ``o`` group contains
    faces whose indices point into that shared buffer. MuJoCo only consumes the
    first group, so each group must be extracted into its own complete OBJ.
    """
    vertices = []
    groups = []
    current = None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("o "):
            current = {"name": line[2:].strip(), "face_lines": []}
            groups.append(current)
        elif line.startswith("v "):
            vertices.append(line.split()[1:4])
        elif current is not None and line.startswith("f "):
            current["face_lines"].append(line)
    if not groups or not vertices:
        raise ValueError(f"No OBJ object groups in {path}")
    return vertices, groups


def parse_index(token, vertex_count):
    index = int(token.split("/", 1)[0])
    if index < 0:
        index = vertex_count + index + 1
    if not 1 <= index <= vertex_count:
        raise ValueError(f"OBJ index {index} outside 1..{vertex_count}")
    return index - 1


def split_hull(path, destination, prefix):
    hull_files = []
    vertices, groups = obj_groups(path)
    for hull_index, group in enumerate(groups):
        faces = []
        for line in group["face_lines"]:
            indices = [parse_index(token, len(vertices)) for token in line.split()[1:]]
            if len(indices) < 3:
                continue
            for k in range(1, len(indices) - 1):
                faces.append((indices[0], indices[k], indices[k + 1]))
        if not vertices or not faces:
            continue
        used = sorted({index for face in faces for index in face})
        remap = {old: new for new, old in enumerate(used, 1)}
        hull_path = destination / f"{prefix}_{hull_index:02d}.obj"
        with hull_path.open("w") as stream:
            for old in used:
                x, y, z = vertices[old]
                stream.write(f"v {x} {y} {z}\n")
            for face in faces:
                stream.write("f " + " ".join(str(remap[index]) for index in face) + "\n")
        volume = 0.0
        for a, b, c in faces:
            va, vb, vc = (tuple(float(value) for value in vertices[index]) for index in (a, b, c))
            volume += (va[0] * (vb[1] * vc[2] - vb[2] * vc[1])
                       - va[1] * (vb[0] * vc[2] - vb[2] * vc[0])
                       + va[2] * (vb[0] * vc[1] - vb[1] * vc[0])) / 6.0
        hull_files.append((hull_path, abs(volume)))
    if not hull_files:
        raise ValueError(f"No collision hulls found in {path}")
    return hull_files


def visual_min_z(path):
    values = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            values.append(float(line.split()[3]))
    if not values:
        raise ValueError(f"No vertices in {path}")
    return min(values)


def copy_texture(source, destination):
    with Image.open(source) as image:
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image)
        image.thumbnail((MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE), resampling.LANCZOS)
        image.save(destination, format="PNG", optimize=True)


def yaw_quaternion(degrees):
    half = math.radians(degrees) / 2.0
    return math.cos(half), 0.0, 0.0, math.sin(half)


def write_assets(path):
    lines = ["<mujocoinclude>", "  <asset>"]
    for spec in OBJECTS:
        key = spec["key"]
        lines.append(f'    <mesh name="{key}_visual" file="ycb/{key}/visual.obj"/>')
        for hull in sorted((OUTPUT_ROOT / key).glob("collision_*.obj")):
            lines.append(f'    <mesh name="{key}_{hull.stem}" file="ycb/{key}/{hull.name}"/>')
        lines.append(f'    <texture name="{key}_tex" type="2d" file="ycb/{key}/texture.png"/>')
        lines.append(f'    <material name="{key}_mat" texture="{key}_tex" '
                     'specular="0.2" shininess="0.3"/>')
    lines.extend(("  </asset>", "</mujocoinclude>", ""))
    path.write_text("\n".join(lines))


def write_bodies(path, records):
    lines = ["<mujocoinclude>"]
    for spec, record in records:
        key, body = spec["key"], spec["body"]
        position_xy = spec["position_xy"]
        z = TABLE_TOP_Z + OBJECT_CLEARANCE_Z - record["min_z"]
        w, x, y, qz = yaw_quaternion(spec["yaw_deg"])
        lines.append(f'  <body name="{body}" pos="{position_xy[0]:.4f} '
                     f'{position_xy[1]:.4f} {z:.5f}" quat="{w:.8f} {x:.1f} {y:.1f} {qz:.8f}">')
        lines.append(f'    <freejoint name="{body}_free"/>')
        lines.append(f'    <geom name="{key}_visual" type="mesh" mesh="{key}_visual" '
                     f'material="{key}_mat" group="2" contype="0" conaffinity="0" density="0"/>')
        total_volume = sum(volume for _, volume in record["hulls"])
        for hull, volume in record["hulls"]:
            if total_volume > 0:
                mass = spec["mass_kg"] * volume / total_volume
            else:
                mass = spec["mass_kg"] / len(record["hulls"])
            lines.append(f'    <geom name="{key}_{hull.stem}" type="mesh" '
                         f'mesh="{key}_{hull.stem}" group="3" mass="{mass:.8f}" '
                         'condim="6" friction="1 0.02 0.002" solref="0.005 1"/>')
        lines.append("  </body>")
    lines.extend(("</mujocoinclude>", ""))
    path.write_text("\n".join(lines))


def write_readme(path):
    path.write_text(
        "# PiperX YCB objects\n\n"
        "The five tabletop objects are derived from the YCB models in\n"
        "`pybullet-object-models` (source commit 57cc400). Visual meshes keep\n"
        "their original OBJ UVs; textures were downscaled to at most 1024 px.\n"
        "Collision meshes were split from `collision_vhacd.obj` into one OBJ\n"
        "per convex hull because MuJoCo only reads the first OBJ object group.\n\n"
        "YCB data is published under CC BY 4.0; see the YCB Object and Model Set\n"
        "for attribution requirements.\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "python scripts/prepare_ycb_objects.py\n"
        "```\n"
    )


def main():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing source models: {SOURCE_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in OBJECTS:
        source = SOURCE_ROOT / spec["source"]
        visual = source / spec["visual_file"]
        texture = source / spec["texture_file"]
        collision = source / "collision_vhacd.obj"
        for required in (visual, texture, collision):
            if not required.is_file():
                raise FileNotFoundError(required)
        destination = OUTPUT_ROOT / spec["key"]
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        shutil.copyfile(visual, destination / "visual.obj")
        copy_texture(texture, destination / "texture.png")
        hulls = split_hull(collision, destination, "collision")
        record = {"min_z": visual_min_z(visual), "hulls": hulls}
        records.append((spec, record))
        print(f"{spec['key']:16s} z={TABLE_TOP_Z + OBJECT_CLEARANCE_Z - record['min_z']:.4f} m "
              f"hulls={len(hulls)} mass={spec['mass_kg']:.3f} kg")
    write_assets(SCENE_ASSET_DIR / "ycb_assets.xml")
    write_bodies(SCENE_ASSET_DIR / "ycb_objects.xml", records)
    write_readme(OUTPUT_ROOT / "README.md")
    print(f"Wrote {SCENE_ASSET_DIR / 'ycb_assets.xml'}")
    print(f"Wrote {SCENE_ASSET_DIR / 'ycb_objects.xml'}")


if __name__ == "__main__":
    main()
