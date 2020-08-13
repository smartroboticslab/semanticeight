cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(se-apps-tests LANGUAGES C CXX)

# Build the dependencies inside the test build directory.
add_subdirectory(../../third_party third_party)
add_subdirectory(../../se_shared se_shared)
add_subdirectory(../../se_core se_core)
add_subdirectory(../../se_voxel_impl se_voxel_impl)
add_subdirectory(../../se_denseslam se_denseslam)

# Make sure that custom modules are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# Testing dependencies
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

enable_testing()

# Build the tests.
add_subdirectory(reader_iclnuim)
add_subdirectory(reader_openni)
add_subdirectory(reader_raw)

