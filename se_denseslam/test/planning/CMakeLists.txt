cmake_minimum_required(VERSION 3.9...3.16)

set(unit_test_name planning-unittest)
add_executable(${unit_test_name}
  "planning_unittest.cpp"
)
target_include_directories(${unit_test_name} BEFORE
  PRIVATE
    "../../include"
    "../../../se_voxel_impl/include"
)
target_link_libraries(${unit_test_name}
  PRIVATE
    SE::DenseSLAMPinholeCamera
)
gtest_add_tests(${unit_test_name} "" AUTO)

