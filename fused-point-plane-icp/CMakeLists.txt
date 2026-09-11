cmake_minimum_required(VERSION 3.16)
project(fused_point_plane_icp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Assignment constraints: Eigen 3.4.0 + Ceres 2.1.0, C++, no PCL / no ROS.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres 2.1 REQUIRED)

# ---------------------------------------------------------------------------
# The cloud type, its loader and the kd-tree are shared with the other ICP variants, so they live
# in common and are referenced by source, not copied. This package holds only what is specific to
# the fused residual: the SE(3) helpers, the residual algebra, the Ceres cost/manifold and the loop.
# ---------------------------------------------------------------------------
set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common"
    CACHE PATH "Checkout of the shared ICP geometry project")
if(NOT EXISTS "${COMMON_DIR}/include/common/point_cloud.hpp")
  message(FATAL_ERROR
    "common not found at '${COMMON_DIR}'. "
    "Set -DCOMMON_DIR=<path> to its checkout.")
endif()

# The sample clouds live with the rest of the datasets, next to the KITTI bag.
set(FUSED_ICP_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")

add_library(fused_icp
  src/icp_fused_point_plane.cpp
  ${COMMON_DIR}/src/rotation.cpp
  ${COMMON_DIR}/src/point_cloud.cpp
  ${COMMON_DIR}/src/kdtree.cpp
)
target_include_directories(fused_icp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${COMMON_DIR}/include
)
target_link_libraries(fused_icp PUBLIC Eigen3::Eigen Ceres::ceres)
target_compile_options(fused_icp PRIVATE -Wall -Wextra -Wpedantic)

add_executable(icp_demo app/main.cpp)
target_link_libraries(icp_demo PRIVATE fused_icp)
target_compile_definitions(icp_demo PRIVATE FUSED_ICP_DATA_DIR="${FUSED_ICP_DATA_DIR}")

option(FUSED_ICP_BUILD_TESTS "Build the GoogleTest suite" ON)
if(FUSED_ICP_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_icp test/test_icp.cpp)
    target_link_libraries(test_icp PRIVATE fused_icp GTest::gtest GTest::gtest_main)
    target_compile_definitions(test_icp PRIVATE FUSED_ICP_DATA_DIR="${FUSED_ICP_DATA_DIR}")
    add_test(NAME test_icp COMMAND test_icp)
  else()
    message(STATUS "GTest not found -- skipping the test target (build the demo only).")
  endif()
endif()
