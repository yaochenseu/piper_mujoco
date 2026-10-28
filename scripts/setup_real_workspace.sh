#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL="${PIPER_CONTROL_DIR:-$(cd "$ROOT/../piper_control" && pwd)}"
WS="${PIPER_REAL_WS:-$ROOT/real_ws}"

if [ ! -d "$CONTROL/agx_arm_ros" ]; then
    echo "Missing $CONTROL/agx_arm_ros" >&2
    exit 2
fi

set +u
source /opt/ros/humble/setup.bash
set -u
mkdir -p "$WS/src"
ln -sfn "$CONTROL/agx_arm_ros" "$WS/src/agx_arm_ros"
ln -sfn "$CONTROL/tools" "$WS/tools"

cd "$WS"
colcon build --symlink-install

echo
echo "Built ROS workspace: $WS"
echo "Start the bridge with: $ROOT/scripts/piperx_real_bridge.sh"
