#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
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

exec /usr/bin/python3 "$ROOT/real_bridge/piperx_ros_bridge.py" \
    --config "$ROOT/configs/real_piperx.yaml" "$@"
