#!/usr/bin/env bash
#
# 이미 떠 있는 clobot_assignment 컨테이너에 접속한다. (터미널 여러 개 열 때 사용)
#
#   ./exec.sh                       ROS 가 source 된 bash 쉘로 진입
#   ./exec.sh roscore               컨테이너 안에서 명령 하나만 실행
#   ./exec.sh rviz                  rviz 실행
#   ./exec.sh --root                root 로 진입 (apt install 등)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER="${CONTAINER:-clobot_assignment}"
WS_SRC=/home/clobot_assignment/dev_ws/src

log() { printf '\033[1;32m[exec.sh]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[exec.sh]\033[0m %s\n' "$*" >&2; exit 1; }

USER_OPTS=()
if [ "${1:-}" = "--root" ]; then
    USER_OPTS=( --user root )
    shift
fi

# TTY 가 없는 환경(스크립트/CI)에서도 `./exec.sh <명령>` 이 돌아가게
TTY_OPTS=( -i )
if [ -t 0 ] && [ -t 1 ]; then
    TTY_OPTS+=( -t )
fi

# 컨테이너가 없거나 멈춰 있으면 run.sh 로 넘긴다.
if ! docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
    log "컨테이너 '${CONTAINER}' 가 없습니다. run.sh 로 생성합니다."
    exec "${HERE}/run.sh"
fi
if [ "$(docker container inspect -f '{{.State.Running}}' "${CONTAINER}")" != "true" ]; then
    log "컨테이너 '${CONTAINER}' 가 정지 상태입니다. 시작합니다."
    docker start "${CONTAINER}" >/dev/null
fi

# 접속할 때마다 호스트의 현재 DISPLAY 를 넘겨준다 (재로그인 시 :0 -> :1 등으로 바뀜)
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null; then
    xhost +local: >/dev/null 2>&1 || true
fi

if [ "$#" -eq 0 ]; then
    exec docker exec "${TTY_OPTS[@]}" "${USER_OPTS[@]}" \
        -e DISPLAY="${DISPLAY:-}" \
        -e TERM="${TERM:-xterm-256color}" \
        -w "${WS_SRC}" \
        "${CONTAINER}" bash
else
    exec docker exec "${TTY_OPTS[@]}" "${USER_OPTS[@]}" \
        -e DISPLAY="${DISPLAY:-}" \
        -e TERM="${TERM:-xterm-256color}" \
        -w "${WS_SRC}" \
        "${CONTAINER}" bash -lc "$*"
fi
