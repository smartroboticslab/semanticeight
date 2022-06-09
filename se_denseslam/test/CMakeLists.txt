cmake_minimum_required(VERSION 3.5...3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

include(GoogleTest)

# Libraries to link all tests with
link_libraries(
  SE::Shared
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)

add_subdirectory(depth_utils)
add_subdirectory(entropy)
add_subdirectory(frustum_intersection)
add_subdirectory(gain_raycasting)
add_subdirectory(lod)
add_subdirectory(planning)
add_subdirectory(pose_grid_history)
add_subdirectory(pose_mask_history)
add_subdirectory(preprocessing)

