cmake_minimum_required(VERSION 3.16)
project(point_to_point_icp LANGUAGES CXX)

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
# The cloud type, its loader, the kd-tree and the Euler angle helpers are shared with the
# point-to-plane project, so they live in common and are referenced by source, not copied.
# This package holds only what is specific to point-to-point: the cost function and the ICP loop.
# ---------------------------------------------------------------------------
set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common"
    CACHE PATH "Checkout of the shared ICP geometry project")
if(NOT EXISTS "${COMMON_DIR}/include/common/point_cloud.hpp")
  message(FATAL_ERROR
    "common not found at '${COMMON_DIR}'. "
    "Set -DCOMMON_DIR=<path> to its checkout.")
endif()

# The sample clouds live with the rest of the datasets, next to the KITTI bag.
set(P2PT_ICP_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")

add_library(p2pt_icp
  src/icp_point_to_point.cpp
  ${COMMON_DIR}/src/rotation.cpp
  ${COMMON_DIR}/src/point_cloud.cpp
  ${COMMON_DIR}/src/kdtree.cpp
)
target_include_directories(p2pt_icp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${COMMON_DIR}/include
)
target_link_libraries(p2pt_icp PUBLIC Eigen3::Eigen Ceres::ceres)
target_compile_options(p2pt_icp PRIVATE -Wall -Wextra -Wpedantic)

add_executable(icp_demo app/main.cpp)
target_link_libraries(icp_demo PRIVATE p2pt_icp)
target_compile_definitions(icp_demo PRIVATE P2PT_ICP_DATA_DIR="${P2PT_ICP_DATA_DIR}")

option(P2PT_ICP_BUILD_TESTS "Build the GoogleTest suite" ON)
if(P2PT_ICP_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_icp test/test_icp.cpp)
    target_link_libraries(test_icp PRIVATE p2pt_icp GTest::gtest GTest::gtest_main)
    target_compile_definitions(test_icp PRIVATE P2PT_ICP_DATA_DIR="${P2PT_ICP_DATA_DIR}")
    add_test(NAME test_icp COMMAND test_icp)
  else()
    message(STATUS "GTest not found -- skipping the test target (build the demo only).")
  endif()
endif()
