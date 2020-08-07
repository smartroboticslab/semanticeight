cmake_minimum_required(VERSION 3.0.0)

add_subdirectory(lodepng)

option(YAML_CPP_BUILD_TESTS "Enable testing" OFF)
add_subdirectory(yaml-cpp)

option(SRL_PROJECTION_BUILD_TESTS "Enable testing" OFF)
add_subdirectory(srl_projection)

