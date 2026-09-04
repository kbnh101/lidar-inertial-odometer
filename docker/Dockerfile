#
# Ubuntu 20.04 (focal) + ROS Noetic + Ceres 2.1.0 + Eigen 3.4.0
#
#   Eigen / Ceres 는 focal apt 버전(Eigen 3.3.7)이 과제 요구사항(3.4.0)과 다르므로
#   /usr/local 에 소스 빌드해서 넣는다. CMake 는 /usr 보다 /usr/local 을 먼저
#   찾으므로 find_package(Eigen3 3.4 REQUIRED NO_MODULE) 가 이쪽을 집는다.
#
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Asia/Seoul
ENV TZ=${TZ}
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=noetic \
    WS_ROOT=/home/clobot_assignment/dev_ws

# ---------------------------------------------------------------------------
# 1. 기본 툴체인 / 로케일 / 타임존
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg2 lsb-release tzdata locales sudo \
        build-essential cmake pkg-config git \
        gdb vim nano less tmux htop unzip \
        python3 python3-pip python3-dev \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone \
    && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. ROS Noetic (desktop-full: rviz / rqt / rosbag 포함)
# ---------------------------------------------------------------------------
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
        | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
        > /etc/apt/sources.list.d/ros1-latest.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full \
        ros-noetic-rviz \
        ros-noetic-tf2-tools \
        ros-noetic-rqt-graph \
        python3-rosdep python3-rosinstall python3-rosinstall-generator \
        python3-wstool python3-catkin-tools python3-osrf-pycommon \
    && rosdep init \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Eigen 3.4.0 (header-only, /usr/local)
# ---------------------------------------------------------------------------
ARG EIGEN_VERSION=3.4.0
RUN cd /tmp \
    && wget -q https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz \
    && tar -xf eigen-${EIGEN_VERSION}.tar.gz \
    && cmake -S eigen-${EIGEN_VERSION} -B eigen-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
    && cmake --build eigen-build --target install \
    && rm -rf /tmp/eigen-${EIGEN_VERSION} /tmp/eigen-${EIGEN_VERSION}.tar.gz /tmp/eigen-build

# ---------------------------------------------------------------------------
# 4. Ceres Solver 2.1.0 (위에서 설치한 Eigen 3.4.0 에 링크)
# ---------------------------------------------------------------------------
ARG CERES_VERSION=2.1.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgoogle-glog-dev libgflags-dev \
        libatlas-base-dev liblapack-dev libblas-dev \
        libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && wget -q -O ceres-${CERES_VERSION}.tar.gz \
        https://github.com/ceres-solver/ceres-solver/archive/refs/tags/${CERES_VERSION}.tar.gz \
    && tar -xf ceres-${CERES_VERSION}.tar.gz \
    && cmake -S ceres-solver-${CERES_VERSION} -B ceres-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CXX_STANDARD=17 \
        -DEigen3_DIR=/usr/local/share/eigen3/cmake \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
    && cmake --build ceres-build -j"$(nproc)" --target install \
    && ldconfig \
    && rm -rf /tmp/ceres-solver-${CERES_VERSION} /tmp/ceres-${CERES_VERSION}.tar.gz /tmp/ceres-build

# ---------------------------------------------------------------------------
# 5. 테스트 / 시각화용 부속 (GoogleTest 는 focal 에서 소스만 제공 -> 직접 빌드)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgtest-dev python3-numpy python3-matplotlib mesa-utils \
    && cmake -S /usr/src/googletest -B /tmp/gtest-build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/gtest-build -j"$(nproc)" --target install \
    && rm -rf /tmp/gtest-build /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 5b. 코드 서식 도구 (.clang-format 규칙을 적용할 때 쓴다)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang-format clang-tidy \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 6. 호스트 UID/GID 와 맞춘 사용자 (마운트한 소스의 파일 권한이 틀어지지 않게)
# ---------------------------------------------------------------------------
ARG USERNAME=clobot
ARG USER_UID=1000
ARG USER_GID=1000
# 호스트의 render 그룹 GID. /dev/dri/renderD* 가 이 그룹 소유라 run.sh 가
# --group-add 로 넘기는데, 컨테이너에 같은 GID 의 그룹이 없으면
# /etc/bash.bashrc 의 `groups` 호출이 "cannot find name for group ID" 를 뱉는다.
ARG RENDER_GID=110
RUN if getent group ${USER_GID} >/dev/null; then \
        groupmod -n ${USERNAME} "$(getent group ${USER_GID} | cut -d: -f1)"; \
    else \
        groupadd -g ${USER_GID} ${USERNAME}; \
    fi \
    && if getent passwd ${USER_UID} >/dev/null; then \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m "$(getent passwd ${USER_UID} | cut -d: -f1)"; \
    else \
        useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME}; \
    fi \
    && if [ -n "${RENDER_GID}" ] && ! getent group "${RENDER_GID}" >/dev/null; then \
        groupadd -g "${RENDER_GID}" render; \
    fi \
    && usermod -aG video,audio,dialout,plugdev ${USERNAME} \
    && RENDER_GRP="$(getent group "${RENDER_GID}" 2>/dev/null | cut -d: -f1 || true)" \
    && if [ -n "${RENDER_GRP}" ]; then usermod -aG "${RENDER_GRP}" ${USERNAME}; fi \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p ${WS_ROOT}/src ${WS_ROOT}/build ${WS_ROOT}/devel ${WS_ROOT}/logs \
    && chown -R ${USER_UID}:${USER_GID} /home/clobot_assignment /home/${USERNAME}

# ---------------------------------------------------------------------------
# 7. ROS 환경 source 스크립트
#
#    한 군데(/etc/ros_setup.sh)에만 적어두고 세 경로에서 모두 부른다:
#      - entrypoint            (docker run 으로 들어올 때)
#      - /etc/profile.d        (docker exec ... bash -lc "rviz" 같은 login 셸)
#      - ~/.bashrc             (docker exec -it ... bash 같은 interactive 셸)
#    ubuntu 의 기본 .bashrc 는 비대화형 셸에서 곧바로 return 하므로
#    .bashrc 에만 넣으면 `bash -lc` 로 실행한 명령에서 ROS 가 안 잡힌다.
# ---------------------------------------------------------------------------
RUN cat <<'EOF' > /etc/ros_setup.sh && chmod +x /etc/ros_setup.sh
# shellcheck shell=bash
WS_ROOT="${WS_ROOT:-/home/clobot_assignment/dev_ws}"
source /opt/ros/noetic/setup.bash
for _overlay in "${WS_ROOT}/devel/setup.bash" "${WS_ROOT}/install/setup.bash"; do
    [ -f "${_overlay}" ] && source "${_overlay}"
done
unset _overlay
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
export ROSCONSOLE_FORMAT='[${severity}] [${node}] ${message}'
# rviz / Qt 가 컨테이너 안에서 X11 로 뜨게
export QT_X11_NO_MITSHM=1
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$(id -un)}"
mkdir -p "${XDG_RUNTIME_DIR}" 2>/dev/null && chmod 0700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true
EOF

RUN echo '. /etc/ros_setup.sh' > /etc/profile.d/10-ros-setup.sh \
    && cat <<'EOF' > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh
#!/usr/bin/env bash
set -e
. /etc/ros_setup.sh
exec "$@"
EOF

USER ${USERNAME}

RUN cat <<'EOF' >> /home/${USERNAME}/.bashrc

# ---- ROS Noetic ----
. /etc/ros_setup.sh
cd "${WS_ROOT:-/home/clobot_assignment/dev_ws}/src" 2>/dev/null || true
EOF

RUN rosdep update --rosdistro noetic || true

WORKDIR /home/clobot_assignment/dev_ws/src
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
