cmake_minimum_required(VERSION 3.16)
project(common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Assignment constraints: Eigen 3.4.0, C++, no PCL / no ROS. Ceres is not needed here -- this
# package holds no optimization, only the geometry the ICP variants and the odometry share.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# The sample clouds live with the rest of the datasets, next to the KITTI bag.
set(COMMON_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")

add_library(common
  src/rotation.cpp
  src/point_cloud.cpp
  src/kdtree.cpp
  src/kdtree_radius.cpp
)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(common PUBLIC Eigen3::Eigen)
target_compile_options(common PRIVATE -Wall -Wextra -Wpedantic)

option(COMMON_BUILD_TESTS "Build the GoogleTest suite" ON)
if(COMMON_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_common test/test_common.cpp)
    target_link_libraries(test_common PRIVATE common GTest::gtest GTest::gtest_main)
    target_compile_definitions(test_common PRIVATE COMMON_DATA_DIR="${COMMON_DATA_DIR}")
    target_compile_options(test_common PRIVATE -Wall -Wextra)
    add_test(NAME test_common COMMAND test_common)
  else()
    message(STATUS "GTest not found -- skipping the test target.")
  endif()
endif()
