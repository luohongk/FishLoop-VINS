#!/usr/bin/env bash
# regen_vins_config.sh
# --------------------
# One-shot regenerator: re-reads my_kalibr_result/ and rewrites
# config/my_kalibr_fisheye/ + the launch file.
#
# Edit the variables below (or pass through as CLI args) when your bag
# topic names or stereo pair selection changes.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KALIBR_DIR="${REPO_ROOT}/my_kalibr_result"
OUT_DIR="${REPO_ROOT}/config/my_kalibr_fisheye"
LAUNCH_NAME="my_kalibr_fisheye"

CAM_PAIR="${CAM_PAIR:-0,1}"
TOPIC_MAP="${TOPIC_MAP:-0:/fisheye/left/image_raw,1:/fisheye/right/image_raw,2:/fisheye/bleft/image_raw,3:/fisheye/bright/image_raw}"
IMU_TOPIC="${IMU_TOPIC:-/imu_data_raw}"
IMU_NOISE_SCALE="${IMU_NOISE_SCALE:-5.0}"
USE_GPU="${USE_GPU:-1}"
IS_FISHEYE="${IS_FISHEYE:-1}"
FISHEYE_FOV="${FISHEYE_FOV:-235}"
ENABLE_SIDES="${ENABLE_SIDES:-0}"
SHOW_TRACK="${SHOW_TRACK:--1}"
PAD_TOP="${PAD_TOP:-0}"
PAD_BOTTOM="${PAD_BOTTOM:-0}"
PAD_LEFT="${PAD_LEFT:-0}"
PAD_RIGHT="${PAD_RIGHT:-0}"
FLATTEN_WIDTH="${FLATTEN_WIDTH:-600}"
FLATTEN_HEIGHT="${FLATTEN_HEIGHT:-200}"
SHOW_WIDTH="${SHOW_WIDTH:-1080}"
FISHEYE_MASK="${FISHEYE_MASK:-${REPO_ROOT}/config/my_kalibr_fisheye/fisheye_mask.png}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/lhk/output}"

python3 "${SCRIPT_DIR}/kalibr_to_vins.py" \
    --kalibr-dir       "${KALIBR_DIR}" \
    --out-dir          "${OUT_DIR}" \
    --cam-pair         "${CAM_PAIR}" \
    --topic-map        "${TOPIC_MAP}" \
    --imu-topic        "${IMU_TOPIC}" \
    --imu-noise-scale  "${IMU_NOISE_SCALE}" \
    --use-gpu          "${USE_GPU}" \
    --is-fisheye       "${IS_FISHEYE}" \
    --fisheye-fov      "${FISHEYE_FOV}" \
    --enable-sides     "${ENABLE_SIDES}" \
    --show-track       "${SHOW_TRACK}" \
    --pad-top          "${PAD_TOP}" \
    --pad-bottom       "${PAD_BOTTOM}" \
    --pad-left         "${PAD_LEFT}" \
    --pad-right        "${PAD_RIGHT}" \
    --flatten-width    "${FLATTEN_WIDTH}" \
    --flatten-height   "${FLATTEN_HEIGHT}" \
    --show-width       "${SHOW_WIDTH}" \
    --fisheye-mask     "${FISHEYE_MASK}" \
    --output-path      "${OUTPUT_PATH}" \
    --launch           "${LAUNCH_NAME}" \
    "$@"
