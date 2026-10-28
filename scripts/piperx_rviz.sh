#!/usr/bin/env bash
# Launch RViz2 for the real PiperX robot, wrist camera and grasp markers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${PIPERX_RVIZ_PID_FILE:-}" ]; then
    printf '%s\n' "$$" > "$PIPERX_RVIZ_PID_FILE"
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_HOME="${ROS_HOME:-/tmp/piperx_ros_home}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/piperx_ros_log}"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"

set +u
source /opt/ros/humble/setup.bash
if [ -f "$HOME/agx_arm_ws/install/setup.bash" ]; then
    source "$HOME/agx_arm_ws/install/setup.bash"
elif [ -f "$ROOT/real_ws/install/setup.bash" ]; then
    source "$ROOT/real_ws/install/setup.bash"
else
    echo "No built agx_arm_ros workspace found." >&2
    echo "Run scripts/setup_real_workspace.sh first, or source your own workspace." >&2
    exit 2
fi
set -u

exec ros2 launch "$ROOT/scripts/piperx_rviz.launch.py" "$@"
