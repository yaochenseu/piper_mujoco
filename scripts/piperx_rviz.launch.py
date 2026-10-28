#!/usr/bin/env python3
"""Launch RViz2 with the real PiperX robot, wrist camera and grasp markers."""
from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


ROOT = Path(__file__).resolve().parents[1]


def generate_launch_description():
    grasp_glob_arg = DeclareLaunchArgument(
        "grasp_glob",
        default_value=str(ROOT / "docs/vision_runs/*/cycle_*/final/grasp_pose.json"),
        description="Glob for the newest grasp_pose.json to render as markers",
    )
    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(get_package_share_path("agx_arm_description") / "launch/display.launch.py")
        ),
        launch_arguments={
            "arm_type": "piper_x",
            "effector_type": "agx_gripper",
            "follow": "true",
            "control": "false",
            "gui": "false",
            "feedback_topic": "piperx/joint_states",
            "rvizconfig": str(ROOT / "configs/piperx_rviz.rviz"),
        }.items(),
    )
    visualizer = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            str(ROOT / "scripts/piperx_rviz_visualizer.py"),
            "--config", str(ROOT / "configs/real_piperx.yaml"),
            "--grasp-glob", LaunchConfiguration("grasp_glob"),
        ],
        output="screen",
    )
    return LaunchDescription([grasp_glob_arg, display_launch, visualizer])
