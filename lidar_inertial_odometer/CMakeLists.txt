cmake_minimum_required(VERSION 3.16)
project(lidar_inertial_odometer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
)

# 의존성: Eigen 3.4.0 + Ceres 2.1.0 만 사용한다 (GTSAM / PCL 없음).
# Eigen 은 /usr/local 의 3.4.0 을 잡아야 한다 (focal apt 는 3.3.7).
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres 2.1 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES lio_core
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros
  DEPENDS EIGEN3 CERES
)

# ---------------------------------------------------------------------------
# point-to-plane-icp 라이브러리를 소스째로 재사용한다.
# 코드를 복사하지 않고 sibling 디렉토리를 참조해 단일 소스로 유지한다.
# catkin workspace 에서는 <ws>/src/point-to-plane-icp 가 된다.
# ---------------------------------------------------------------------------
set(P2P_ICP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../point-to-plane-icp"
    CACHE PATH "Checkout of the point-to-plane ICP project")
if(NOT EXISTS "${P2P_ICP_DIR}/include/p2p_icp/icp_point_to_plane.hpp")
  message(FATAL_ERROR
    "point-to-plane-icp not found at '${P2P_ICP_DIR}'. "
    "Set -DP2P_ICP_DIR=<path> to its checkout.")
endif()

add_library(p2p_icp_vendored STATIC
  ${P2P_ICP_DIR}/src/rotation.cpp
  ${P2P_ICP_DIR}/src/point_cloud.cpp
  ${P2P_ICP_DIR}/src/kdtree.cpp
  ${P2P_ICP_DIR}/src/icp_point_to_plane.cpp
)
target_include_directories(p2p_icp_vendored PUBLIC ${P2P_ICP_DIR}/include)
target_link_libraries(p2p_icp_vendored PUBLIC Eigen3::Eigen Ceres::ceres)
set_target_properties(p2p_icp_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# LIO core -- ROS 의존성이 전혀 없는 순수 C++ 클래스들.
# ROS 는 lio_node 에서 입출력 인터페이스로만 쓴다.
# ---------------------------------------------------------------------------
add_library(lio_core
  src/imu_preintegrator.cpp
  src/kdtree_radius.cpp
  src/feature_extractor.cpp
  src/local_map.cpp
  src/lio_odometer.cpp
)
target_include_directories(lio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(lio_core PUBLIC p2p_icp_vendored Eigen3::Eigen)
target_compile_options(lio_core PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# ROS1 node -- 콜백에서 메시지를 Eigen 타입으로 변환해 core 클래스 함수를
# 호출하고, 결과를 Odometry / Path / PointCloud2 / TF 로 내보내는 것만 담당.
# ---------------------------------------------------------------------------
add_executable(lio_node src/lio_node.cpp)
target_include_directories(lio_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(lio_node PRIVATE lio_core ${catkin_LIBRARIES})
target_compile_options(lio_node PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------------
# 자체 검증 테스트 -- rosbag 없이 core class 들을 합성 데이터로 확인한다.
#   catkin_make -DLIO_BUILD_TESTS=ON && ./devel/lib/lidar_inertial_odometer/test_lio_core
# ---------------------------------------------------------------------------
option(LIO_BUILD_TESTS "Build the lio_core self-check" ON)
if(LIO_BUILD_TESTS)
  add_executable(test_lio_core test/test_lio_core.cpp)
  target_link_libraries(test_lio_core PRIVATE lio_core)
  target_compile_options(test_lio_core PRIVATE -Wall -Wextra)
  set_target_properties(test_lio_core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})
endif()

install(TARGETS lio_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS lio_core
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(PROGRAMS scripts/plot_trajectory.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
