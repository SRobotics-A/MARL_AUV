#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/uaea-data}"
DEFAULT_ISAAC_ASSETS_DIR="/media/xtj/1CC8D044C8D01DB8/RL-download/isaac-sim/v5.1.0/Assets/Isaac/5.1/Isaac"
ISAAC_ASSETS_DIR="${ISAAC_ASSETS_DIR:-$DEFAULT_ISAAC_ASSETS_DIR}"
HEADLESS_MODE="${HEADLESS:-1}"
LIVESTREAM_MODE="${LIVESTREAM:-0}"
ENABLE_CAMERAS_MODE="${ENABLE_CAMERAS:-0}"

mkdir -p outputs logs "$DATA_DIR/logs/model_service" "$DATA_DIR/swanlab" "$DATA_DIR/checkpoints"
chmod a+rwX outputs logs || true
mkdir -p "$DATA_DIR/logs/model_service" "$DATA_DIR/swanlab" "$DATA_DIR/checkpoints"
chmod a+rwX "$DATA_DIR" "$DATA_DIR/logs" "$DATA_DIR/logs/model_service" "$DATA_DIR/swanlab" "$DATA_DIR/checkpoints" || true

xhost +local:docker >/dev/null || true
xhost +SI:localuser:"$(id -un)" >/dev/null || true

ASSET_MOUNT_ARGS=()
if [ -n "$ISAAC_ASSETS_DIR" ]; then
    if [ ! -e "$ISAAC_ASSETS_DIR/Environments/Outdoor/Rivermark/rivermark.usd" ]; then
        echo "警告: ISAAC_ASSETS_DIR 下未找到 Environments/Outdoor/Rivermark/rivermark.usd: $ISAAC_ASSETS_DIR" >&2
    fi
    ASSET_MOUNT_ARGS=(-v "$ISAAC_ASSETS_DIR":/opt/isaac-assets/Isaac:ro)
fi

docker run --rm -it \
    --runtime=nvidia \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    -e TERM=xterm \
    -e HEADLESS="$HEADLESS_MODE" \
    -e LIVESTREAM="$LIVESTREAM_MODE" \
    -e ENABLE_CAMERAS="$ENABLE_CAMERAS_MODE" \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e QT_X11_NO_MITSHM=1 \
    -e SDL_VIDEODRIVER=x11 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network host --ipc host --shm-size=16g \
    -v "$PWD":/app/uaea-src:rw \
    -v "$DATA_DIR":/data:rw \
    "${ASSET_MOUNT_ARGS[@]}" \
    uaea-unified:1.0.2 /bin/bash



# docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --network host --ipc host --shm-size=16g -v "$PWD":/app/uaea-src:rw -v "$HOME/uaea-data":/data:rw uaea-unified:1.0.2 /bin/bash

# cd /app/uaea-src
# FLY_FORWARD_DISABLE_ACC_LOAD=1 /opt/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=Isaac-fly-forward-v0 --headless --num_envs=32 --algorithm="PPO"
