#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="$HOME/uaea-data"
DEFAULT_CONFIGS_DIR="$SCRIPT_DIR/exts/MARL_mav_carry_ext/config"
DEFAULT_EXAMPLES_DIR="$SCRIPT_DIR/scripts"
DEFAULT_ASSET_DIR="$SCRIPT_DIR/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets"
DEFAULT_IMAGE_TAR="$SCRIPT_DIR/../docker-images-20260506/uaea-unified-1.0.2.tar.gz"

IMAGE="${IMAGE:-uaea-unified:1.0.2}"
IMAGE_TAR="${IMAGE_TAR:-$DEFAULT_IMAGE_TAR}"
CONTAINER_NAME="${CONTAINER_NAME:-uaea-unified}"
DASHBOARD_PORT="${UAEA_DASHBOARD_PORT:-8000}"
BACKEND_PRESET="${BACKEND_PRESET:-airsim/citypark}"
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
CONFIGS_DIR="${CONFIGS_DIR:-$DEFAULT_CONFIGS_DIR}"
EXAMPLES_DIR="${EXAMPLES_DIR:-$DEFAULT_EXAMPLES_DIR}"
ASSET_DIR="${ASSET_DIR:-$DEFAULT_ASSET_DIR}"
UAEA_SRC_DIR="${UAEA_SRC_DIR:-$SCRIPT_DIR}"
CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173}"
USE_SUDO_DOCKER="${USE_SUDO_DOCKER:-0}"
USE_GPU="${USE_GPU:-auto}"
MIN_NVIDIA_DRIVER_MAJOR="${MIN_NVIDIA_DRIVER_MAJOR:-570}"
MOUNT_HOST_CONFIGS="${MOUNT_HOST_CONFIGS:-auto}"
MOUNT_HOST_EXAMPLES="${MOUNT_HOST_EXAMPLES:-auto}"

MODE="${1:-dashboard}"

usage() {
    cat <<'USAGE'
用法:
  ./uaea-unified.sh [dashboard|shell]

模式:
  dashboard  启动 UAEA Dashboard 后端，并由后端托管前端 dist
  shell      仅进入容器 shell，方便手动排查

常用环境变量:
  IMAGE=uaea-unified:1.0.2
  IMAGE_TAR=<默认: ../docker-images-20260506/uaea-unified-1.0.2.tar.gz>
  CONTAINER_NAME=uaea-unified
  UAEA_DASHBOARD_PORT=8000
  BACKEND_PRESET=airsim/citypark
  DATA_DIR=$HOME/uaea-data
  CONFIGS_DIR=<默认: 当前仓库/exts/MARL_mav_carry_ext/config>
  EXAMPLES_DIR=<默认: 当前仓库/scripts>
  ASSET_DIR=<默认: 当前仓库/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets>
  UAEA_SRC_DIR=<默认: 当前仓库根目录>
  USE_SUDO_DOCKER=0|1|auto
  USE_GPU=auto|0|1
  MOUNT_HOST_CONFIGS=auto|0|1
  MOUNT_HOST_EXAMPLES=auto|0|1
USAGE
}

case "$MODE" in
    dashboard|shell)
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "未知模式: $MODE" >&2
        usage >&2
        exit 1
        ;;
esac

mkdir -p "$DATA_DIR"

for required_dir in "$CONFIGS_DIR" "$EXAMPLES_DIR" "$ASSET_DIR" "$UAEA_SRC_DIR" "$DATA_DIR"; do
    if [ ! -e "$required_dir" ]; then
        echo "宿主机路径不存在: $required_dir" >&2
        exit 1
    fi
done

prepare_data_dir() {
    mkdir -p \
        "$DATA_DIR/logs/model_service" \
        "$DATA_DIR/swanlab" \
        "$DATA_DIR/checkpoints"

    chmod a+rwX \
        "$DATA_DIR" \
        "$DATA_DIR/logs" \
        "$DATA_DIR/logs/model_service" \
        "$DATA_DIR/swanlab" \
        "$DATA_DIR/checkpoints" || true

    if [ -e "$DATA_DIR/logs/dashboard_streaming.log" ] && [ ! -w "$DATA_DIR/logs/dashboard_streaming.log" ]; then
        echo "警告: $DATA_DIR/logs/dashboard_streaming.log 当前用户不可写。"
        echo "如后续仍报日志权限错误，请执行:"
        echo "  sudo chown -R 1000:1000 '$DATA_DIR'"
    fi
}

configure_repo_mounts() {
    if [ "$MOUNT_HOST_CONFIGS" = "1" ] || { [ "$MOUNT_HOST_CONFIGS" = "auto" ] && [ -f "$CONFIGS_DIR/registry.yaml" ]; }; then
        COMMON_ARGS+=(-v "$CONFIGS_DIR":/app/configs:rw)
    elif [ "$MOUNT_HOST_CONFIGS" = "auto" ]; then
        echo "配置目录缺少 registry.yaml，保留镜像内置 /app/configs。"
    elif [ "$MOUNT_HOST_CONFIGS" != "0" ]; then
        echo "未知 MOUNT_HOST_CONFIGS: $MOUNT_HOST_CONFIGS，应为 auto、0 或 1。" >&2
        exit 1
    fi

    if [ "$MOUNT_HOST_EXAMPLES" = "1" ] || { [ "$MOUNT_HOST_EXAMPLES" = "auto" ] && [ -d "$EXAMPLES_DIR/configs" ]; }; then
        COMMON_ARGS+=(-v "$EXAMPLES_DIR":/app/examples:rw)
    elif [ "$MOUNT_HOST_EXAMPLES" = "auto" ]; then
        echo "示例目录不是 UAEA examples 结构，保留镜像内置 /app/examples。"
    elif [ "$MOUNT_HOST_EXAMPLES" != "0" ]; then
        echo "未知 MOUNT_HOST_EXAMPLES: $MOUNT_HOST_EXAMPLES，应为 auto、0 或 1。" >&2
        exit 1
    fi
}

select_docker_cmd() {
    if [ "$USE_SUDO_DOCKER" = "1" ]; then
        DOCKER_CMD=(sudo docker)
        return
    fi

    if [ "$USE_SUDO_DOCKER" = "0" ]; then
        DOCKER_CMD=(docker)
        return
    fi

    if docker info >/dev/null 2>&1; then
        DOCKER_CMD=(docker)
        return
    fi

    if command -v sudo >/dev/null 2>&1; then
        echo "当前用户无法直接访问 Docker daemon，将改用 sudo docker。"
        DOCKER_CMD=(sudo docker)
        return
    fi

    DOCKER_CMD=(docker)
}

configure_gpu_args() {
    case "$USE_GPU" in
        1)
            COMMON_ARGS+=(
                --gpus all
                -e NVIDIA_VISIBLE_DEVICES=all
                -e NVIDIA_DRIVER_CAPABILITIES=all
                -e __GLX_VENDOR_LIBRARY_NAME=nvidia
                -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
            )
            return
            ;;
        0)
            echo "GPU 挂载已关闭: USE_GPU=0"
            return
            ;;
        auto)
            ;;
        *)
            echo "未知 USE_GPU: $USE_GPU，应为 auto、0 或 1。" >&2
            exit 1
            ;;
    esac

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "未找到 nvidia-smi，跳过 GPU 挂载。如需强制启用，设置 USE_GPU=1。"
        return
    fi

    local driver_version
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [ -z "$driver_version" ]; then
        echo "nvidia-smi 无法读取驱动状态，跳过 GPU 挂载。如需强制启用，设置 USE_GPU=1。"
        return
    fi

    local driver_major="${driver_version%%.*}"
    if [ "$driver_major" -lt "$MIN_NVIDIA_DRIVER_MAJOR" ]; then
        echo "NVIDIA 驱动版本 $driver_version 低于镜像 CUDA 12.8 建议的 ${MIN_NVIDIA_DRIVER_MAJOR}.x，跳过 GPU 挂载。"
        echo "如已升级驱动或想强制尝试，可设置 USE_GPU=1。"
        return
    fi

    COMMON_ARGS+=(
        --gpus all
        -e NVIDIA_VISIBLE_DEVICES=all
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e __GLX_VENDOR_LIBRARY_NAME=nvidia
        -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    )
}

ensure_image() {
    if "${DOCKER_CMD[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
        return
    fi

    if [ -f "$IMAGE_TAR" ]; then
        echo "Docker 镜像不存在，正在从本地文件导入: $IMAGE_TAR"
        "${DOCKER_CMD[@]}" load -i "$IMAGE_TAR"
        return
    fi

    echo "提示: 未找到本地 Docker 镜像 $IMAGE，也未找到镜像包 $IMAGE_TAR。" >&2
    echo "Docker 将按默认行为尝试运行/拉取该镜像。" >&2
}

COMMON_ARGS=(
    --rm
    -it
    --name "$CONTAINER_NAME"
    --runtime=nvidia
    --network host
    --ipc host
    --shm-size=16g
    -e TERM=xterm
    -e QT_X11_NO_MITSHM=1
    -e SDL_VIDEODRIVER=x11
    -e UAEA_DASHBOARD_PORT="$DASHBOARD_PORT"
    -e UAEA_DASHBOARD_DIST=/app/dist
    -e DASHBOARD_DATABASE_URL=sqlite:////data/dashboard.db
    -e UAEA_DATA_DIR=/data
    -e UAEA_SWANLAB_ROOT=/data/swanlab
    -e SWANLAB_LOG_DIR=/data/swanlab
    -e UAEA_LOG_DIR=/data/logs
    -e UAEA_MODEL_SERVICE_LOG_DIR=/data/logs/model_service
    -e UAEA_DATASETS_META_FILE=/data/datasets_meta.json
    -e CHECKPOINT_DIR=/data/checkpoints
    -e UAEA_REPO_ROOT=/app
    -e UAEA_ROOT=/app/uaea-src
    -e MARL_UAV_ROOT=/app/uaea-src
    -e CORS_ORIGINS="$CORS_ORIGINS"
    -v "$ASSET_DIR":/opt/isaac-assets/Isaac:ro
    -v "$UAEA_SRC_DIR":/app/uaea-src:ro
    -v "$DATA_DIR":/data:rw
)

if [ -n "${DISPLAY:-}" ]; then
    COMMON_ARGS+=(-e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)

    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null || true
        cleanup() {
            xhost -local:docker >/dev/null 2>&1 || true
        }
        trap cleanup EXIT
    fi
fi

case "$MODE" in
    dashboard)
        echo "启动 Dashboard: http://localhost:${DASHBOARD_PORT}"
        echo "前端说明: 1.0.2 镜像模式下，前端 dist 由后端直接托管，不需要单独再起 Vite。"
        echo "当前挂载:"
        echo "  配置目录: $CONFIGS_DIR -> /app/configs (仅当包含 registry.yaml 时覆盖)"
        echo "  示例目录: $EXAMPLES_DIR -> /app/examples (仅当符合 UAEA examples 结构时覆盖)"
        echo "  资产目录: $ASSET_DIR -> /opt/isaac-assets/Isaac"
        echo "  仓库目录: $UAEA_SRC_DIR -> /app/uaea-src"
        echo "  数据目录: $DATA_DIR -> /data"
        CMD=(
            /bin/bash
            -lc
            "exec python -m UAEA.modules.dashboard.back_end.main --host 0.0.0.0 --port ${DASHBOARD_PORT} --backend-preset '${BACKEND_PRESET}'"
        )
        ;;
    shell)
        echo "进入容器 shell: ${CONTAINER_NAME}"
        CMD=(/bin/bash)
        ;;
esac

select_docker_cmd
prepare_data_dir
configure_repo_mounts
configure_gpu_args
ensure_image

exec "${DOCKER_CMD[@]}" run "${COMMON_ARGS[@]}" "$IMAGE" "${CMD[@]}"
