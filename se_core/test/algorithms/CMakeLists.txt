cmake_minimum_required(VERSION 3.10)
project(octree_lib)


set(PROJECT_TEST_NAME algorithms_unittest)

set(UNIT_TEST_NAME unique-unittest)
add_executable(${UNIT_TEST_NAME} unique_unittest.cpp)
target_include_directories(${UNIT_TEST_NAME} PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(${UNIT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} pthread)

GTEST_ADD_TESTS(${UNIT_TEST_NAME} "" AUTO)

set(UNIT_TEST_NAME meshing-unittest)
add_executable(${UNIT_TEST_NAME} meshing_unittest.cpp)
target_include_directories(${UNIT_TEST_NAME} PUBLIC ${GTEST_INCLUDE_DIRS} ../../../apps/kfusion/thirdparty)
target_link_libraries(${UNIT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} pthread)

GTEST_ADD_TESTS(${UNIT_TEST_NAME} "" AUTO)

