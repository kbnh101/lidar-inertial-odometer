cmake_minimum_required(VERSION 3.18)
project(cuda_residual_tutorial LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(ament_cmake QUIET)

set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common" CACHE PATH "Shared geometry checkout")
add_library(tutorial_residual STATIC src/residual.cu)
target_include_directories(tutorial_residual PUBLIC include)
target_link_libraries(tutorial_residual PUBLIC CUDA::cudart)

# Reuse common directly, without pulling in Ceres or the ICP solver.
add_executable(cuda_residual_demo
  app/main.cpp
  ${COMMON_DIR}/src/point_cloud.cpp
  ${COMMON_DIR}/src/kdtree.cpp
  ${COMMON_DIR}/src/rotation.cpp)
target_include_directories(cuda_residual_demo PRIVATE ${COMMON_DIR}/include)
target_link_libraries(cuda_residual_demo PRIVATE tutorial_residual Eigen3::Eigen)
target_compile_options(cuda_residual_demo PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(cuda_residual_demo PRIVATE
  TUTORIAL_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")

option(BUILD_TESTING "Build residual correctness checks (requires a CUDA GPU)" ON)
if(BUILD_TESTING)
  enable_testing()
  add_executable(test_residual test/test_residual.cpp)
  target_link_libraries(test_residual PRIVATE tutorial_residual)
  add_test(NAME tutorial_residual_known_pairs COMMAND test_residual)
  add_test(NAME tutorial_residual_data COMMAND cuda_residual_demo --verify)
  add_test(NAME tutorial_residual_pose COMMAND cuda_residual_demo --pose 0.1 -0.2 0.3 0.02 -0.03 0.04 --verify)
endif()

install(TARGETS cuda_residual_demo RUNTIME DESTINATION lib/${PROJECT_NAME})
install(FILES ../data/source_point.txt ../data/target_point.txt DESTINATION share/${PROJECT_NAME}/data)
install(FILES README.md DESTINATION share/${PROJECT_NAME})
if(ament_cmake_FOUND)
  ament_package()
endif()
