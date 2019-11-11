cmake_minimum_required(VERSION 3.9...3.16)

# Create a preprocessor definition with the absolute path to the test_data
# directory.
get_filename_component(TEST_DATA_DIR "../test_data" REALPATH)

add_executable(bounding-sphere-unittest "bounding_sphere_unittest.cpp")
target_compile_definitions(bounding-sphere-unittest PRIVATE TEST_DATA_DIR="${TEST_DATA_DIR}")
gtest_add_tests(bounding-sphere-unittest "" AUTO)

