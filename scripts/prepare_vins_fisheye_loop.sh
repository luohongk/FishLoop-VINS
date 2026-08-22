#!/usr/bin/env bash
# Kalibr cam1/cam2 are exposed to VINS as the standardized cam0/cam1 topics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/config/vins_fisheye_loop"

python3 "${SCRIPT_DIR}/kalibr_to_vins.py" \
    --kalibr-dir "${REPO_ROOT}/my_kalibr_result" \
    --out-dir "${CONFIG_DIR}" \
    --cam-pair 1,2 \
    --topic-map 1:/cam0/image,2:/cam1/image \
    --imu-topic /imu/data_raw \
    --compressed \
    --use-gpu "${USE_GPU:-0}" \
    --imu-noise-scale "${IMU_NOISE_SCALE:-5.0}" \
    --output-path /root/catkin_ws/src/VINS-Fisheye/data \
    --estimate-td 0 \
    --fisheye-fov "${FISHEYE_FOV:-250}" \
    --enable-sides 0 \
    --show-track 0 \
    --flatten-width "${FLATTEN_WIDTH:-600}" \
    --flatten-height "${FLATTEN_HEIGHT:-200}" \
    --external-viz 0

echo "Prepared ${CONFIG_DIR}: Kalibr cam1->/cam0, cam2->/cam1."
