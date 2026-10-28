#!/usr/bin/env bash
# One-sentence real PiperX grasp: pass only the instruction, e.g. "抓取香蕉".
#
# Real-robot defaults: single-stage perception, grasp-only (hold the object and
# return to the captured baseline without placing), and automatic base-yaw
# search when the target is not visible.  Opt out with --two-stage / --place.
# Use --rviz to start RViz2 for this run; it is closed when the pipeline exits.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 '抓取香蕉' [--rviz] [--two-stage] [--place] [extra vision args...]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTRUCTION="$1"
shift
STAMP="$(date +%Y%m%d_%H%M%S)"
DIAGNOSTICS="$ROOT/docs/vision_runs/real_grasp_$STAMP"

MODE=()
EXTRA=()
TWO_STAGE=0
PLACE=0
RVIZ=0
for arg in "$@"; do
    case "$arg" in
        --two-stage)
            TWO_STAGE=1
            ;;
        --place)
            PLACE=1
            ;;
        --rviz)
            RVIZ=1
            ;;
        *)
            EXTRA+=("$arg")
            ;;
    esac
done
[ "$TWO_STAGE" -eq 1 ] || MODE+=(--single-stage)
[ "$PLACE" -eq 1 ] || MODE+=(--grasp-only)

CMD=(
    "$ROOT/scripts/run_real_piperx_vision.sh"
    --instruction "$INSTRUCTION"
    --execute
    --open-gate
    --max-attempts 1
    --diagnostics-dir "$DIAGNOSTICS"
    "${MODE[@]}"
    "${EXTRA[@]}"
)

if [ "$RVIZ" -eq 0 ]; then
    exec "${CMD[@]}"
fi

mkdir -p "$DIAGNOSTICS"
RVIZ_PID=""
RVIZ_GROUP=0
signal_rviz() {
    local signal="$1"
    if [ "$RVIZ_GROUP" -eq 1 ]; then
        kill "-$signal" -- "-$RVIZ_PID" 2>/dev/null || \
            kill "-$signal" "$RVIZ_PID" 2>/dev/null || true
    else
        kill "-$signal" "$RVIZ_PID" 2>/dev/null || true
    fi
}
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ -n "$RVIZ_PID" ] && kill -0 "$RVIZ_PID" 2>/dev/null; then
        # The helper writes its PID before exec'ing ros2 launch.  With setsid,
        # that PID is also the process-group leader for RViz and its children.
        signal_rviz INT
        for _ in $(seq 1 20); do
            kill -0 "$RVIZ_PID" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$RVIZ_PID" 2>/dev/null; then
            signal_rviz TERM
        fi
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

RVIZ_PID_FILE="$DIAGNOSTICS/rviz.pid"
rm -f "$RVIZ_PID_FILE"
if command -v setsid >/dev/null 2>&1; then
    RVIZ_GROUP=1
    PIPERX_RVIZ_PID_FILE="$RVIZ_PID_FILE" \
        setsid "$ROOT/scripts/piperx_rviz.sh" >"$DIAGNOSTICS/rviz.log" 2>&1 &
else
    PIPERX_RVIZ_PID_FILE="$RVIZ_PID_FILE" \
        "$ROOT/scripts/piperx_rviz.sh" >"$DIAGNOSTICS/rviz.log" 2>&1 &
fi
RVIZ_JOB=$!
for _ in $(seq 1 20); do
    if [ -s "$RVIZ_PID_FILE" ]; then
        RVIZ_PID="$(cat "$RVIZ_PID_FILE")"
        break
    fi
    sleep 0.1
done
if [ -z "$RVIZ_PID" ] && kill -0 "$RVIZ_JOB" 2>/dev/null; then
    RVIZ_PID="$RVIZ_JOB"
fi
if [ -z "$RVIZ_PID" ] || ! kill -0 "$RVIZ_PID" 2>/dev/null; then
    echo "RViz failed to start; see $DIAGNOSTICS/rviz.log" >&2
    RVIZ_PID=""
fi

"${CMD[@]}"
