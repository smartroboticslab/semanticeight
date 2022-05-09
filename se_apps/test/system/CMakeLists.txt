cmake_minimum_required(VERSION 3.9...3.16)

set(unit_test_name gain-unittest)
add_executable(${unit_test_name}
  "../../src/reader_base.cpp"
  "../../src/reader_raw.cpp"
  "gain_unittest.cpp"
)
target_link_libraries(${unit_test_name}
  PRIVATE
    SE::DenseSLAMPinholeCamera
)
get_filename_component(SEQUENCE_PATH "fr1_desk" ABSOLUTE)
target_compile_definitions(${unit_test_name} PRIVATE SEQUENCE_PATH="${SEQUENCE_PATH}")
gtest_add_tests(${unit_test_name} "" AUTO)

