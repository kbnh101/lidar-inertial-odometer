# ROS 2 Humble / Ubuntu 22.04, Eigen 3.4 and Ceres 2.1.
FROM ros:humble-ros-base-jammy
ARG DEBIAN_FRONTEND=noninteractive
ENV WS_ROOT=/home/clobot_assignment/dev_ws \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 QT_X11_NO_MITSHM=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates sudo gdb clang-format clang-tidy \
    libeigen3-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev \
    libgtest-dev python3-colcon-common-extensions python3-numpy python3-matplotlib \
    python3-pip ros-humble-rviz2 ros-humble-rosbag2 ros-humble-ament-cmake-gtest \
    && rm -rf /var/lib/apt/lists/*
ARG CERES_VERSION=2.1.0
RUN curl -fsSL "https://github.com/ceres-solver/ceres-solver/archive/refs/tags/${CERES_VERSION}.tar.gz" -o /tmp/ceres.tar.gz \
    && tar -xzf /tmp/ceres.tar.gz -C /tmp \
    && cmake -S /tmp/ceres-solver-${CERES_VERSION} -B /tmp/ceres-build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
    && cmake --build /tmp/ceres-build -j2 --target install \
    && ldconfig \
    && rm -rf /tmp/ceres*
# Offline ROS1 bag -> ROS2 conversion; ROS1 is not required at runtime.
RUN pip3 install --no-cache-dir 'rosbags==0.10.9'
ARG USERNAME=clobot
ARG USER_UID=1000
ARG USER_GID=1000
ARG RENDER_GID=110
RUN groupadd -g ${USER_GID} ${USERNAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} \
    && if [ -n "${RENDER_GID}" ] && ! getent group "${RENDER_GID}" >/dev/null; then groupadd -g "${RENDER_GID}" render; fi \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p ${WS_ROOT}/src ${WS_ROOT}/build ${WS_ROOT}/install ${WS_ROOT}/log \
    && chown -R ${USER_UID}:${USER_GID} /home/clobot_assignment
RUN printf '%s\n' \
    'source /opt/ros/humble/setup.bash' \
    'if [ -f "${WS_ROOT}/install/setup.bash" ]; then source "${WS_ROOT}/install/setup.bash"; fi' \
    > /etc/ros_setup.sh \
    && echo '. /etc/ros_setup.sh' > /etc/profile.d/10-ros-setup.sh \
    && printf '%s\n' '#!/usr/bin/env bash' 'set -e' '. /etc/ros_setup.sh' 'exec "$@"' > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh
USER ${USERNAME}
RUN echo '. /etc/ros_setup.sh' >> /home/${USERNAME}/.bashrc
WORKDIR /home/clobot_assignment/dev_ws
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
