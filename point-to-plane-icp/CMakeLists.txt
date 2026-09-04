cmake_minimum_required(VERSION 3.16)
project(point_to_plane_icp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Assignment constraints: Eigen 3.4.0 + Ceres 2.1.0, C++, no PCL / no ROS.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres 2.1 REQUIRED)

set(P2P_ICP_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

add_library(p2p_icp
  src/rotation.cpp
  src/point_cloud.cpp
  src/kdtree.cpp
  src/icp_point_to_plane.cpp
)
target_include_directories(p2p_icp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(p2p_icp PUBLIC Eigen3::Eigen Ceres::ceres)
target_compile_options(p2p_icp PRIVATE -Wall -Wextra -Wpedantic)

add_executable(icp_demo app/main.cpp)
target_link_libraries(icp_demo PRIVATE p2p_icp)
target_compile_definitions(icp_demo PRIVATE P2P_ICP_DATA_DIR="${P2P_ICP_DATA_DIR}")

option(P2P_ICP_BUILD_TESTS "Build the GoogleTest suite" ON)
if(P2P_ICP_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_icp test/test_icp.cpp)
    target_link_libraries(test_icp PRIVATE p2p_icp GTest::gtest GTest::gtest_main)
    target_compile_definitions(test_icp PRIVATE P2P_ICP_DATA_DIR="${P2P_ICP_DATA_DIR}")
    add_test(NAME test_icp COMMAND test_icp)
  else()
    message(STATUS "GTest not found -- skipping the test target (build the demo only).")
  endif()
endif()
