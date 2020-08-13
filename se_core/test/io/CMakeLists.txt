# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required(VERSION 3.9.0 FATAL_ERROR)
project(octree-io-unittest LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(octomap)

set(PROJECT_TEST_NAME io)
set(UNIT_TEST_NAME octree-${PROJECT_TEST_NAME}-unittest)
add_executable(${UNIT_TEST_NAME} io_unittest.cpp)
target_include_directories(${UNIT_TEST_NAME}
  PUBLIC
  ../../include
  ${EIGEN3_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(${UNIT_TEST_NAME}
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

GTEST_ADD_TESTS(${UNIT_TEST_NAME} "" AUTO)

set(UNIT_TEST_NAME octomap-${PROJECT_TEST_NAME}-unittest)
add_executable(${UNIT_TEST_NAME} octomap_io_unittest.cpp)
target_include_directories(${UNIT_TEST_NAME}
  PUBLIC
  ../../include
  ${EIGEN3_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(${UNIT_TEST_NAME}
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)
if(DEFINED OCTOMAP_INCLUDE_DIRS)
  target_include_directories(${UNIT_TEST_NAME} PUBLIC ${OCTOMAP_INCLUDE_DIRS})
  target_link_libraries(${UNIT_TEST_NAME} ${OCTOMAP_LIBRARIES})
  target_compile_definitions(${UNIT_TEST_NAME} PUBLIC SE_OCTOMAP=1)
else()
  target_compile_definitions(${UNIT_TEST_NAME} PUBLIC SE_OCTOMAP=0)
endif()

GTEST_ADD_TESTS(${UNIT_TEST_NAME} "" AUTO)