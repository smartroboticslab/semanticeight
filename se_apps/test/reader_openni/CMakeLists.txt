cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(reader-no-openni-unittest LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME}
  ../../src/reader_base.cpp
  ../../src/reader_openni.cpp
  reader_no_openni_unittest.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ../../../se_core/include
    ../../../se_shared/include
    ../../../se_denseslam/include
    ../../include
    ${EIGEN3_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  stdc++fs
  SE::Shared
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

