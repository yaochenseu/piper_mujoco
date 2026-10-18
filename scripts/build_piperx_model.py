"""Build a MuJoCo model from the checked-out official Piper X URDFs.

Only the include-only Xacro used by this model is expanded here. The upstream
checkout is never edited. STL visuals use the same link frames as the DAE files.
"""
from pathlib import Path
import argparse
import copy
import math
import os
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/agx_arm_urdf"
OUTPUT = ROOT / "manipulator_grasp/assets/agilex_piper_x"


def quat(rpy):
    r, p, y = (float(x) / 2 for x in rpy.split())
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return " ".join(f"{x:.12g}" for x in (
        cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy,
    ))


def origin(element):
    value = element.find("origin")
    return {
        "pos": value.get("xyz", "0 0 0") if value is not None else "0 0 0",
        "quat": quat(value.get("rpy", "0 0 0")) if value is not None else "1 0 0 0",
    }


def write(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build(source=SOURCE, output=OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    base_path = source / "piper_x/urdf/piper_x_description.urdf"
    gripper_path = source / "piper_x/urdf/piper_x_with_gripper_description.xacro"
    if not base_path.exists() or not gripper_path.exists():
        raise FileNotFoundError(f"Clone https://github.com/agilexrobotics/agx_arm_urdf.git into {source}")
    robot = ET.parse(base_path).getroot()
    for item in ET.parse(gripper_path).getroot():
        if item.tag.startswith("{"):
            if item.tag != "{http://www.ros.org/wiki/xacro}include":
                raise ValueError(f"Unsupported Xacro element: {item.tag}")
            continue
        robot.append(copy.deepcopy(item))

    output.mkdir(parents=True, exist_ok=True)
    # The massless 'gripper' link is an abstract width command, not a real
    # translating rigid body. Physical fingers remain independent slide joints.
    virtual_joint = robot.find("joint[@name='gripper']")
    virtual_joint.set("type", "fixed")
    for tag in ("axis", "limit"):
        virtual_joint.remove(virtual_joint.find(tag))
    tcp = ET.SubElement(robot, "link", name="grasp_link")
    tcp_joint = ET.SubElement(robot, "joint", name="grasp_joint", type="fixed")
    ET.SubElement(tcp_joint, "parent", link="gripper_base")
    ET.SubElement(tcp_joint, "child", link=tcp.get("name"))
    ET.SubElement(tcp_joint, "origin", xyz="0 0 0.125", rpy="0 0 0")
    # Export an expanded URDF for other kinematics tools; preserve the actual
    # finger relationship rather than referring to the removed virtual DOF.
    for joint in robot.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)
    ET.SubElement(robot.find("joint[@name='gripper_joint2']"), "mimic",
                  joint="gripper_joint1", multiplier="-1", offset="0")

    for mesh in robot.findall(".//mesh"):
        name = mesh.get("filename")
        prefix = "package://agx_arm_description/agx_arm_urdf/"
        if not name.startswith(prefix):
            raise ValueError(f"Unexpected mesh URI: {name}")
        path = source / name[len(prefix):]
        # MuJoCo does not load Collada; use the supplied STL in the same frame.
        if path.suffix.lower() == ".dae":
            path = path.parent.parent / (path.stem + ".stl")
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", os.path.relpath(path, output))
    write(robot, output / "piperx.urdf")

    model = ET.Element("mujoco", model="piper_x")
    ET.SubElement(model, "compiler", angle="radian", autolimits="true")
    ET.SubElement(model, "option", timestep="0.002", integrator="implicitfast",
                  cone="elliptic", iterations="100")
    default = ET.SubElement(model, "default")
    ET.SubElement(default, "joint", damping="2", armature="0.02")
    ET.SubElement(default, "geom", friction="0.8 0.01 0.001", condim="4")
    asset = ET.SubElement(model, "asset")
    links = {link.get("name"): link for link in robot.findall("link")}
    children = {}
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    world = ET.SubElement(model, "worldbody")

    def add_link(name, parent, joint=None):
        attrs = origin(joint) if joint is not None else {"pos": "0 0 0.74"}
        body = ET.SubElement(parent, "body", name=name, **attrs)
        link = links[name]
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            # fullinertia is expressed in the body's frame. Official models
            # have zero inertial RPY; reject future changes needing rotation.
            if origin(inertial)["quat"] != "1 0 0 0":
                raise ValueError(f"Nonzero inertial RPY: {name}")
            ET.SubElement(body, "inertial", pos=origin(inertial)["pos"],
                          mass=inertial.find("mass").get("value"),
                          fullinertia=" ".join(inertia.get(k) for k in (
                              "ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        if joint is not None and joint.get("type") != "fixed":
            limits = joint.find("limit")
            ET.SubElement(body, "joint", name=joint.get("name"),
                          type={"revolute": "hinge", "prismatic": "slide"}[joint.get("type")],
                          axis=joint.find("axis").get("xyz"),
                          range=f"{limits.get('lower')} {limits.get('upper')}")
        visual = link.find("visual")
        collision = link.find("collision")
        if visual is not None:
            mesh_name = name + "_mesh"
            ET.SubElement(asset, "mesh", name=mesh_name,
                          file=visual.find("geometry/mesh").get("filename"))
            ET.SubElement(body, "geom", name=name + "_visual", type="mesh", mesh=mesh_name,
                          group="2", contype="0", conaffinity="0", density="0",
                          rgba="0.18 0.2 0.24 1" if name.startswith("gripper") else "0.82 0.84 0.88 1",
                          **origin(visual))
        if collision is not None:
            ET.SubElement(body, "geom", name=name + "_collision", type="mesh",
                          mesh=name + "_mesh", group="3", **origin(collision))
        if name in ("gripper_link1", "gripper_link2"):
            # Thin pads sit on the inner face near the fingertips; they do
            # not attach the object. The object is held by friction/contact.
            ET.SubElement(body, "geom", name=name + "_pad", type="box",
                          pos="0 -0.012 -0.001", size="0.012 0.010 0.002",
                          friction="1.5 0.02 0.002", rgba="0.12 0.12 0.12 1",
                          mass="0.002", solref="0.005 1")
        if name == "grasp_link":
            ET.SubElement(body, "site", name="tcp", size="0.004", rgba="0 1 0 0.5", group="4")
        if name == "gripper_base":
            # A short plate connects the housing directly to the transverse
            # gripper support (mesh front x=-0.042, beam z=0.041..0.065).
            ET.SubElement(body, "geom", name="wrist_camera_bracket", type="box",
                          pos="-0.040 0 0.048", size="0.006 0.030 0.010",
                          rgba="0.18 0.18 0.18 1", group="2", contype="0",
                          conaffinity="0", density="0")
            # Align the housing with the transverse support, without tilt.
            # MuJoCo looks along camera -Z; this half-turn points it towards
            # the fingertips (+Z of gripper_base).
            mount = ET.SubElement(body, "body", name="wrist_camera_mount",
                                  pos="-0.055 0 0.048", quat="0 0 1 0")
            ET.SubElement(mount, "geom", name="wrist_camera_housing", type="box",
                          pos="0 0 0", size="0.012 0.035 0.015",
                          rgba="0.12 0.12 0.12 1", group="2", contype="0",
                          conaffinity="0", density="0")
            for index, pos in enumerate(("0 -0.018 -0.017", "0 0.018 -0.017"), 1):
                ET.SubElement(mount, "geom", name=f"wrist_camera_lens{index}", type="cylinder",
                              pos=pos, size="0.007 0.003", rgba="0.1 0.3 0.9 1",
                              group="2", contype="0", conaffinity="0", density="0")
            ET.SubElement(mount, "camera", name="wrist",
                          pos="0 0 -0.021", fovy="120", mode="fixed")
        for child_joint in children.get(name, []):
            add_link(child_joint.find("child").get("link"), body, child_joint)

    add_link("base_link", world)
    contact = ET.SubElement(model, "contact")
    for joint in robot.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent != "world":
            ET.SubElement(contact, "exclude", body1=parent, body2=child)
    ET.SubElement(contact, "exclude", body1="gripper_link1", body2="gripper_link2")
    equality = ET.SubElement(model, "equality")
    ET.SubElement(equality, "joint", joint1="gripper_joint2", joint2="gripper_joint1",
                  polycoef="0 -1 0 0 0", solref="0.005 1")
    tendon = ET.SubElement(model, "tendon")
    width = ET.SubElement(tendon, "fixed", name="gripper_width")
    ET.SubElement(width, "joint", joint="gripper_joint1", coef="1")
    ET.SubElement(width, "joint", joint="gripper_joint2", coef="-1")
    actuator = ET.SubElement(model, "actuator")
    for i, kp in enumerate((1800, 1800, 1400, 600, 500, 400), 1):
        joint = robot.find(f"joint[@name='joint{i}']")
        limits = joint.find("limit")
        ET.SubElement(actuator, "position", name=f"joint{i}_position", joint=f"joint{i}",
                      kp=str(kp), kv=str((2 * kp) ** 0.5),
                      ctrlrange=f"{limits.get('lower')} {limits.get('upper')}",
                      forcerange="-100 100")
    ET.SubElement(actuator, "position", name="gripper_position", tendon="gripper_width",
                  kp="1200", kv="12", ctrlrange="0 0.1", forcerange="-20 20")
    write(model, output / "piperx.xml")
    print(f"Generated {output / 'piperx.xml'} and piperx.urdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)
