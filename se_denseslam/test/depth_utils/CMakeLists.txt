cmake_minimum_required(VERSION 3.9...3.16)

set(unit_test_name depth-utils-unittest)
add_executable(${unit_test_name}
  "depth_utils_unittest.cpp"
    "../../src/preprocessing.cpp"
    "../../src/depth_utils.cpp"
)
target_include_directories(${unit_test_name} BEFORE
  PRIVATE
    "../../include"
    "../../../se_voxel_impl/include"
)
target_compile_definitions(${unit_test_name}
  PUBLIC
    SE_SENSOR_IMPLEMENTATION=PinholeCamera
)
gtest_add_tests(${unit_test_name} "" AUTO)

