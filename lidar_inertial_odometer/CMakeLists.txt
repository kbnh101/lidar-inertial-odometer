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

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres 2.1 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES lio_core
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros
  DEPENDS EIGEN3 CERES
)

# ---------------------------------------------------------------------------
# 공용 ICP 기하(점군 / kd-tree / 회전)와 point-to-plane-icp 를 소스째로 재사용한다.
# 코드를 복사하지 않고 sibling 디렉토리를 참조해 단일 소스로 유지한다.
# catkin workspace 에서는 <ws>/src/{common,point-to-plane-icp} 가 된다.
# ---------------------------------------------------------------------------
set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common"
    CACHE PATH "Checkout of the shared ICP geometry project")
if(NOT EXISTS "${COMMON_DIR}/include/common/point_cloud.hpp")
  message(FATAL_ERROR
    "common not found at '${COMMON_DIR}'. "
    "Set -DCOMMON_DIR=<path> to its checkout.")
endif()

add_library(common_vendored STATIC
  ${COMMON_DIR}/src/rotation.cpp
  ${COMMON_DIR}/src/point_cloud.cpp
  ${COMMON_DIR}/src/kdtree.cpp
  ${COMMON_DIR}/src/kdtree_radius.cpp
)
target_include_directories(common_vendored PUBLIC ${COMMON_DIR}/include)
target_link_libraries(common_vendored PUBLIC Eigen3::Eigen)
set_target_properties(common_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(P2P_ICP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../point-to-plane-icp"
    CACHE PATH "Checkout of the point-to-plane ICP project")
if(NOT EXISTS "${P2P_ICP_DIR}/include/p2p_icp/icp_point_to_plane.hpp")
  message(FATAL_ERROR
    "point-to-plane-icp not found at '${P2P_ICP_DIR}'. "
    "Set -DP2P_ICP_DIR=<path> to its checkout.")
endif()

add_library(p2p_icp_vendored STATIC
  ${P2P_ICP_DIR}/src/icp_point_to_plane.cpp
)
target_include_directories(p2p_icp_vendored PUBLIC ${P2P_ICP_DIR}/include)
target_link_libraries(p2p_icp_vendored PUBLIC common_vendored Eigen3::Eigen Ceres::ceres)
set_target_properties(p2p_icp_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(IMU_PREINT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../imu-preintegration"
    CACHE PATH "Checkout of the IMU preintegration project")
if(NOT EXISTS "${IMU_PREINT_DIR}/include/imu_preint/imu_preintegrator.hpp")
  message(FATAL_ERROR
    "imu-preintegration not found at '${IMU_PREINT_DIR}'. "
    "Set -DIMU_PREINT_DIR=<path> to its checkout.")
endif()

add_library(imu_preint_vendored STATIC
  ${IMU_PREINT_DIR}/src/imu_preintegrator.cpp
)
target_include_directories(imu_preint_vendored PUBLIC ${IMU_PREINT_DIR}/include)
target_link_libraries(imu_preint_vendored PUBLIC Eigen3::Eigen)
set_target_properties(imu_preint_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(lio_core
  src/feature_extractor.cpp
  src/local_map.cpp
  src/lio_odometer.cpp
)
target_include_directories(lio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(lio_core PUBLIC common_vendored p2p_icp_vendored imu_preint_vendored Eigen3::Eigen)
target_compile_options(lio_core PRIVATE -Wall -Wextra)

add_executable(lio_node src/lio_node.cpp)
target_include_directories(lio_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(lio_node PRIVATE lio_core ${catkin_LIBRARIES})
target_compile_options(lio_node PRIVATE -Wall -Wextra)

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
install(TARGETS lio_core common_vendored p2p_icp_vendored imu_preint_vendored
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(PROGRAMS scripts/plot_trajectory.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
