cmake_minimum_required(VERSION 3.16)
project(point_to_point_plane_icp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres 2.1 REQUIRED)
find_package(ament_cmake QUIET)
set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common" CACHE PATH "Shared geometry checkout")
add_library(p2ptpl_icp
  src/icp_point_to_point_plane.cpp
  ${COMMON_DIR}/src/rotation.cpp
  ${COMMON_DIR}/src/point_cloud.cpp
  ${COMMON_DIR}/src/kdtree.cpp)
target_include_directories(p2ptpl_icp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${COMMON_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(p2ptpl_icp PUBLIC Eigen3::Eigen Ceres::ceres)
target_compile_options(p2ptpl_icp PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(p2ptpl_icp PROPERTIES POSITION_INDEPENDENT_CODE ON)
option(BUILD_TESTING "Build hybrid ICP tests" ON)
if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(test_hybrid_icp test/test_icp.cpp)
  target_link_libraries(test_hybrid_icp PRIVATE p2ptpl_icp GTest::gtest_main)
  add_test(NAME test_hybrid_icp COMMAND test_hybrid_icp)
endif()
install(TARGETS p2ptpl_icp EXPORT p2ptpl_icp_targets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ ${COMMON_DIR}/include/ DESTINATION include)
if(ament_cmake_FOUND)
  ament_export_targets(p2ptpl_icp_targets HAS_LIBRARY_TARGET NAMESPACE point_to_point_plane_icp::)
  ament_export_dependencies(Eigen3 Ceres)
  ament_package()
endif()
