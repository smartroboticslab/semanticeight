cmake_minimum_required(VERSION 3.9...3.16)

add_executable(segmentation-unittest "segmentation_unittest.cpp")
gtest_add_tests(segmentation-unittest "" AUTO)

add_executable(segmentation-result-unittest "segmentation_result_unittest.cpp")
get_filename_component(SEQUENCE_PATH "fr1_desk" ABSOLUTE)
target_compile_definitions(segmentation-result-unittest PRIVATE SEQUENCE_PATH="${SEQUENCE_PATH}")
gtest_add_tests(segmentation-result-unittest "" AUTO)
