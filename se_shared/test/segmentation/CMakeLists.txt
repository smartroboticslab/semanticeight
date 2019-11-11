cmake_minimum_required(VERSION 3.9...3.16)

add_executable(segmentation-unittest "segmentation_unittest.cpp")
gtest_add_tests(segmentation-unittest "" AUTO)

