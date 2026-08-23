#!/usr/bin/env bash
# Run this script inside the vins-fisheye-gpu container.

# ROS Noetic's setup hooks may inspect optional variables such as
# ROS_MASTER_URI. Enable nounset only after sourcing the ROS environment so
# this script also works from a clean `docker exec ... bash -lc` shell.
set -eo pipefail

BAG_PATH="${1:-/data/20260818-160433.repaired.bag}"
VIZ="${VIZ:-true}"
TRACKING_VIZ="${TRACKING_VIZ:-true}"
LOOP_FUSION="${LOOP_FUSION:-true}"
REBUILD="${REBUILD:-0}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
DRAIN_SECONDS="${DRAIN_SECONDS:-40}"
ROSBAG_QUIET="${ROSBAG_QUIET:-1}"
BAG_RATE="${BAG_RATE:-1.0}"
BAG_NATIVE_TOPICS="${BAG_NATIVE_TOPICS:-0}"
BAG_LEFT_TOPIC="${BAG_LEFT_TOPIC:-/cam1/image/compressed}"
BAG_RIGHT_TOPIC="${BAG_RIGHT_TOPIC:-/cam2/image/compressed}"
VINS_LEFT_TOPIC="${VINS_LEFT_TOPIC:-/cam0/image/compressed}"
VINS_RIGHT_TOPIC="${VINS_RIGHT_TOPIC:-/cam1/image/compressed}"

INPUT_LEFT_TOPIC="${VINS_LEFT_TOPIC}"
INPUT_RIGHT_TOPIC="${VINS_RIGHT_TOPIC}"
if [[ "${BAG_NATIVE_TOPICS}" == "1" ]]; then
    INPUT_LEFT_TOPIC="${BAG_LEFT_TOPIC}"
    INPUT_RIGHT_TOPIC="${BAG_RIGHT_TOPIC}"
fi

source /opt/ros/noetic/setup.bash
set -u

if [[ "${REBUILD}" == "1" ]]; then
    cd /root/catkin_ws
    catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -j"${BUILD_JOBS:-8}"
fi

source /root/catkin_ws/devel/setup.bash

if [[ ! -f "${BAG_PATH}" ]]; then
    echo "Bag not found: ${BAG_PATH}" >&2
    exit 1
fi

ROSCORE_PID=""
LAUNCH_PID=""

cleanup() {
    set +e
    if [[ -n "${LAUNCH_PID}" ]]; then
        kill -INT "${LAUNCH_PID}" 2>/dev/null
        wait "${LAUNCH_PID}" 2>/dev/null
    fi
    if [[ -n "${ROSCORE_PID}" ]]; then
        kill -INT "${ROSCORE_PID}" 2>/dev/null
        wait "${ROSCORE_PID}" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

if ! rostopic list >/dev/null 2>&1; then
    roscore >/tmp/vins_fisheye_roscore.log 2>&1 &
    ROSCORE_PID=$!
    for _ in $(seq 1 50); do
        if rostopic list >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
fi

roslaunch fishloop_vins vins_fisheye_loop.launch \
    viz:="${VIZ}" \
    tracking_viz:="${TRACKING_VIZ}" \
    loop_fusion:="${LOOP_FUSION}" \
    input_left_topic:="${INPUT_LEFT_TOPIC}" \
    input_right_topic:="${INPUT_RIGHT_TOPIC}" \
    >/tmp/vins_fisheye_loop_run.log 2>&1 &
LAUNCH_PID=$!

for _ in $(seq 1 100); do
    if rosnode list 2>/dev/null | grep -q '^/vins_estimator$' && \
       { [[ "${LOOP_FUSION}" != "true" ]] || rosnode list 2>/dev/null | grep -q '^/loop_fusion$'; }; then
        break
    fi
    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        echo "roslaunch exited early; see /tmp/vins_fisheye_loop_run.log" >&2
        exit 1
    fi
    sleep 0.1
done

echo "Waiting ${STARTUP_WAIT_SECONDS}s for camera models and loop vocabulary to initialize"
sleep "${STARTUP_WAIT_SECONDS}"

echo "Running VINS-Fisheye-loop with ${BAG_PATH}"
echo "Rosbag playback rate: ${BAG_RATE}x"
if [[ "${BAG_NATIVE_TOPICS}" == "1" ]]; then
    echo "Bag-native topics: left=${BAG_LEFT_TOPIC}, right=${BAG_RIGHT_TOPIC}"
else
    echo "Topic remap: ${BAG_LEFT_TOPIC} -> ${VINS_LEFT_TOPIC}"
    echo "Topic remap: ${BAG_RIGHT_TOPIC} -> ${VINS_RIGHT_TOPIC}"
fi
echo "VINS log: /tmp/vins_fisheye_loop_run.log"
ROSBAG_ARGS=(play "${BAG_PATH}" --clock)
if [[ -n "${BAG_RATE}" ]]; then
    ROSBAG_ARGS+=(--rate "${BAG_RATE}")
fi
if [[ "${ROSBAG_QUIET}" == "1" ]]; then
    ROSBAG_ARGS+=(--quiet)
fi
if [[ "${BAG_NATIVE_TOPICS}" == "1" ]]; then
    rosbag "${ROSBAG_ARGS[@]}"
else
    rosbag "${ROSBAG_ARGS[@]}" \
        "${BAG_LEFT_TOPIC}:=${VINS_LEFT_TOPIC}" \
        "${BAG_RIGHT_TOPIC}:=${VINS_RIGHT_TOPIC}"
fi

# Let the estimator and pose graph drain their final queued messages.  This is
# deliberately configurable: when rosbag playback outruns the estimator, the
# final loop-closure segment may still be queued after rosbag itself exits.
echo "Rosbag finished; waiting ${DRAIN_SECONDS}s for estimator/pose-graph queues to drain"
sleep "${DRAIN_SECONDS}"
echo "Finished. Trajectories:"
ls -lh /root/catkin_ws/src/fishloop_vins/data/vio.csv \
       /root/catkin_ws/src/fishloop_vins/data/vio_loop.csv
