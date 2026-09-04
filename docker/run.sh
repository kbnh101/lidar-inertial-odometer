#!/usr/bin/env bash
#
# clobot_assignment 개발 컨테이너를 빌드/기동한다.
#
#   ./run.sh              이미지가 없으면 빌드하고, 컨테이너를 띄운 뒤 쉘로 진입
#   ./run.sh -d           컨테이너만 백그라운드로 띄우고 진입은 하지 않음
#   ./run.sh --rebuild    이미지를 새로 빌드 (기존 컨테이너는 제거 후 재생성)
#   ./run.sh --recreate   이미지는 그대로 두고 컨테이너만 재생성
#
# docs/ 를 뺀 최상위 디렉토리 전부를 컨테이너의
#   /home/clobot_assignment/dev_ws/src/<dir>
# 로 bind-mount 하며, rviz 를 위한 X11 / GPU / ROS 네트워크 옵션을 함께 준다.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 이 스크립트는 <repo>/docker/ 에 있고, 마운트 대상은 그 상위인 <repo> 다.
REPO_ROOT="${REPO_ROOT:-$(cd "${HERE}/.." && pwd)}"

IMAGE="${IMAGE:-clobot_assignment:noetic}"
CONTAINER="${CONTAINER:-clobot_assignment}"
CONTAINER_USER="${CONTAINER_USER:-clobot}"
WS_ROOT=/home/clobot_assignment/dev_ws
WS_SRC="${WS_ROOT}/src"

# 마운트에서 제외할 최상위 디렉토리
EXCLUDE_DIRS=("docs" "docker")

DETACH_ONLY=0
REBUILD=0
RECREATE=0
for arg in "$@"; do
    case "${arg}" in
        -d|--detach)   DETACH_ONLY=1 ;;
        --rebuild)     REBUILD=1; RECREATE=1 ;;
        --recreate)    RECREATE=1 ;;
        -h|--help)     sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

log() { printf '\033[1;32m[run.sh]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run.sh]\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker 가 설치되어 있지 않습니다."

# ---------------------------------------------------------------------------
# 1. 이미지 빌드
# ---------------------------------------------------------------------------
if [ "${REBUILD}" -eq 1 ] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    log "이미지 빌드: ${IMAGE} (처음 빌드는 ceres/eigen 소스 빌드 때문에 오래 걸립니다)"
    DOCKER_BUILDKIT=1 docker build \
        -t "${IMAGE}" \
        -f "${HERE}/Dockerfile" \
        --build-arg USER_UID="$(id -u)" \
        --build-arg USER_GID="$(id -g)" \
        --build-arg USERNAME="${CONTAINER_USER}" \
        --build-arg RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3)" \
        "${HERE}"
fi

# ---------------------------------------------------------------------------
# 2. X11 (rviz / rqt) 준비
# ---------------------------------------------------------------------------
XSOCK=/tmp/.X11-unix
# xauth 쿠키는 /tmp 밖에 둔다. /tmp 가 비워진 뒤(WSL 재시작 등) 컨테이너를 start
# 하면 docker 데몬이 사라진 bind 소스를 root 소유 "디렉토리"로 새로 만들어 버리고,
# 파일을 기대하는 마운트가 깨져서 컨테이너가 아예 뜨지 않는다. 게다가 sticky 비트가
# 걸린 /tmp 라 일반 사용자는 그 디렉토리를 지우지도 못한다.
XAUTH="${XAUTH_FILE:-${HOME}/.cache/${CONTAINER}/xauth}"

X11_OPTS=()
if [ -n "${DISPLAY:-}" ]; then
    mkdir -p "$(dirname "${XAUTH}")"
    # 과거 실행에서 디렉토리로 잘못 생겼을 수 있다.
    if [ -d "${XAUTH}" ]; then
        rmdir "${XAUTH}" 2>/dev/null || die "'${XAUTH}' 가 디렉토리입니다. 지운 뒤 다시 실행하세요: sudo rm -rf '${XAUTH}'"
    fi
    : > "${XAUTH}" || die "xauth 쿠키 파일을 만들 수 없습니다: ${XAUTH}"
    # 호스트 쿠키를 컨테이너 hostname 과 무관하게 통하도록 wildcard(ffff) 로 변환
    if command -v xauth >/dev/null; then
        xauth nlist "${DISPLAY}" 2>/dev/null | sed -e 's/^..../ffff/' \
            | xauth -f "${XAUTH}" nmerge - 2>/dev/null || true
    fi
    chmod 0644 "${XAUTH}"
    # 위 방식이 안 먹는 환경(Xwayland 등)을 위한 보조 수단
    command -v xhost >/dev/null && xhost +local: >/dev/null 2>&1 || true

    X11_OPTS+=( -e DISPLAY="${DISPLAY}"
                -e XAUTHORITY="${XAUTH}"
                -e QT_X11_NO_MITSHM=1
                -v "${XSOCK}:${XSOCK}:rw"
                -v "${XAUTH}:${XAUTH}:rw" )
    [ -d "${XSOCK}" ] || log "경고: ${XSOCK} 가 없습니다 (순수 Wayland 세션이면 Xwayland 를 켜세요)."
else
    log "경고: DISPLAY 가 비어 있습니다. rviz 같은 GUI 는 뜨지 않습니다."
fi

# ---------------------------------------------------------------------------
# 3. 마운트 목록 (docs 제외한 최상위 디렉토리 전부)
# ---------------------------------------------------------------------------
MOUNTS=()
MOUNTED_NAMES=()
for path in "${REPO_ROOT}"/*/; do
    name="$(basename "${path}")"
    skip=0
    for ex in "${EXCLUDE_DIRS[@]}"; do
        [ "${name}" = "${ex}" ] && skip=1
    done
    [ "${skip}" -eq 1 ] && continue
    MOUNTS+=( -v "${REPO_ROOT}/${name}:${WS_SRC}/${name}" )
    MOUNTED_NAMES+=( "${name}" )
done
[ "${#MOUNTS[@]}" -eq 0 ] && die "마운트할 디렉토리가 없습니다: ${REPO_ROOT}"

# .clang-format 은 repo 루트에 있는데 src 디렉토리 자체는 마운트하지 않으므로
# (하위 폴더만 마운트한다) 이 파일만 따로 넣어 준다. 컨테이너 안에서
# clang-format 을 돌려도 호스트와 같은 규칙이 적용된다.
if [ -f "${REPO_ROOT}/.clang-format" ]; then
    MOUNTS+=( -v "${REPO_ROOT}/.clang-format:${WS_SRC}/.clang-format:ro" )
fi

# 빌드 산출물(devel/, build_isolated/ 등)은 호스트를 더럽히지 않도록
# named volume 에 담아둔다. 컨테이너를 지워도 재빌드 캐시가 남는다.
MOUNTS+=( -v "${CONTAINER}_ws_build:${WS_ROOT}/build" )
MOUNTS+=( -v "${CONTAINER}_ws_devel:${WS_ROOT}/devel" )

# ---------------------------------------------------------------------------
# 4. GPU / OpenGL (rviz 렌더링)
#
#   focal 의 Mesa(21.2)는 최신 Intel iGPU 를 못 잡아서 llvmpipe(소프트웨어)로
#   떨어진다. NVIDIA runtime 이 있으면 PRIME render offload 로 dGPU 에 붙여
#   하드웨어 가속을 쓴다. 강제로 Mesa 를 쓰고 싶으면 GL=mesa ./run.sh
# ---------------------------------------------------------------------------
GPU_OPTS=()
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    GPU_OPTS+=( --gpus all
                -e NVIDIA_VISIBLE_DEVICES=all
                -e NVIDIA_DRIVER_CAPABILITIES=all )
    if [ "${GL:-nvidia}" = "nvidia" ]; then
        GPU_OPTS+=( -e __NV_PRIME_RENDER_OFFLOAD=1
                    -e __GLX_VENDOR_LIBRARY_NAME=nvidia
                    -e __VK_LAYER_NV_optimus=NVIDIA_only )
        log "NVIDIA runtime 감지 -> --gpus all + PRIME render offload"
    else
        log "NVIDIA runtime 감지 -> --gpus all (GL=${GL}, offload 미사용)"
    fi
fi
if [ -d /dev/dri ]; then
    GPU_OPTS+=( --device /dev/dri:/dev/dri )
    # /dev/dri/renderD* 는 호스트의 render/video 그룹 소유라 GID 를 넘겨줘야
    # 컨테이너의 비-root 사용자가 접근할 수 있다.
    for grp in video render; do
        gid="$(getent group "${grp}" 2>/dev/null | cut -d: -f3 || true)"
        [ -n "${gid}" ] && GPU_OPTS+=( --group-add "${gid}" )
    done
fi

# ---------------------------------------------------------------------------
# 5. 컨테이너 기동
# ---------------------------------------------------------------------------
if [ "${RECREATE}" -eq 1 ] && docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
    log "기존 컨테이너 제거: ${CONTAINER}"
    docker rm -f "${CONTAINER}" >/dev/null
fi

# 기존 컨테이너가 있으면 되살려 보고, 되살릴 수 없으면 지운다.
if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
    if [ "$(docker container inspect -f '{{.State.Running}}' "${CONTAINER}")" = "true" ]; then
        log "이미 실행 중: ${CONTAINER}"
    else
        log "정지된 컨테이너 재시작: ${CONTAINER}"
        # 마운트 설정은 컨테이너를 만들 때 고정된다. 그래서 호스트 쪽 bind 소스가
        # 사라졌거나 타입이 바뀌면(파일 <-> 디렉토리) 위쪽에서 경로를 고쳐 놔도
        # 옛 설정을 물고 있는 기존 컨테이너는 start 할 때마다 같은 자리에서 죽는다.
        # 이럴 때는 다시 만드는 것 외에 방법이 없다. 빌드 산출물은 named volume 에
        # 있으므로 컨테이너를 지워도 재빌드 캐시는 남는다.
        if ! start_error="$(docker start "${CONTAINER}" 2>&1 >/dev/null)"; then
            [ -n "${start_error}" ] && printf '%s\n' "${start_error}" | sed 's/^/    /' >&2
            log "재시작 실패 -> 컨테이너를 새로 만듭니다 (빌드 캐시는 유지됩니다)"
            docker rm -f "${CONTAINER}" >/dev/null
        fi
    fi
fi

if ! docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
    log "컨테이너 생성: ${CONTAINER}"
    docker run -d -it \
        --name "${CONTAINER}" \
        --network host \
        --ipc host \
        --privileged \
        --security-opt seccomp=unconfined \
        --ulimit nofile=65536:65536 \
        -e TERM="${TERM:-xterm-256color}" \
        -e ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}" \
        -e ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}" \
        -v /etc/localtime:/etc/localtime:ro \
        "${X11_OPTS[@]}" \
        "${MOUNTS[@]}" \
        "${GPU_OPTS[@]}" \
        -w "${WS_SRC}" \
        "${IMAGE}" \
        bash >/dev/null
fi

log "마운트됨 -> ${WS_SRC}/{$(IFS=,; echo "${MOUNTED_NAMES[*]}")}  (제외: ${EXCLUDE_DIRS[*]})"

if [ "${DETACH_ONLY}" -eq 1 ]; then
    log "백그라운드 실행 중. 접속하려면: ./exec.sh"
    exit 0
fi

exec "${HERE}/exec.sh"
