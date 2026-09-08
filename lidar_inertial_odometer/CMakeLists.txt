cmake_minimum_required(VERSION 3.16)
project(lidar_inertial_odometer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres 2.1 REQUIRED)

# ---------------------------------------------------------------------------
# 공용 ICP 기하(점군 / kd-tree / 회전)와 point-to-plane-icp 를 소스째로 재사용한다.
# 코드를 복사하지 않고 sibling 디렉토리를 참조해 단일 소스로 유지한다.
# Dependencies remain ROS-free sibling CMake projects.
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

find_package(point_to_point_plane_icp REQUIRED)

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
target_link_libraries(lio_core PUBLIC common_vendored point_to_point_plane_icp::p2ptpl_icp imu_preint_vendored Eigen3::Eigen)
target_compile_options(lio_core PRIVATE -Wall -Wextra)

add_executable(lio_node src/lio_node.cpp)
target_link_libraries(lio_node lio_core)
ament_target_dependencies(lio_node rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs tf2_ros)
target_compile_options(lio_node PRIVATE -Wall -Wextra)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_lio_core test/test_lio_core.cpp)
  target_link_libraries(test_lio_core PRIVATE lio_core)
  target_compile_options(test_lio_core PRIVATE -Wall -Wextra)
  add_test(NAME test_lio_core COMMAND test_lio_core)
endif()

install(TARGETS lio_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(TARGETS lio_core common_vendored imu_preint_vendored
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/plot_trajectory.py DESTINATION lib/${PROJECT_NAME})
ament_package()
