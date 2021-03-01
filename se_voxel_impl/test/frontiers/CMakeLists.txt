cmake_minimum_required(VERSION 3.9...3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(unit_test_name frontiers-unittest)
add_executable(${unit_test_name} "frontiers_unittest.cpp")
target_include_directories(${unit_test_name} BEFORE PRIVATE "../../include" "../../src")
target_compile_definitions(${unit_test_name}
  PUBLIC
    SE_SENSOR_IMPLEMENTATION=PinholeCamera
)
gtest_add_tests(${unit_test_name} "" AUTO)

