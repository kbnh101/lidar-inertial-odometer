cmake_minimum_required(VERSION 3.16)
project(imu_preintegration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Assignment constraints: Eigen 3.4.0, C++, no GTSAM / no ROS. Ceres is not needed here.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(imu_preint
  src/imu_preintegrator.cpp
)
target_include_directories(imu_preint PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(imu_preint PUBLIC Eigen3::Eigen)
target_compile_options(imu_preint PRIVATE -Wall -Wextra -Wpedantic)

option(IMU_PREINT_BUILD_TESTS "Build the GoogleTest suite" ON)
if(IMU_PREINT_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_imu_preintegration test/test_imu_preintegration.cpp)
    target_link_libraries(test_imu_preintegration PRIVATE imu_preint GTest::gtest GTest::gtest_main)
    target_compile_options(test_imu_preintegration PRIVATE -Wall -Wextra)
    add_test(NAME test_imu_preintegration COMMAND test_imu_preintegration)
  else()
    message(STATUS "GTest not found -- skipping the test target.")
  endif()
endif()
