cmake_minimum_required(VERSION 3.9...3.16)

set(unit_test_name frustum-intersection-unittest)
add_executable(${unit_test_name}
  "frustum_intersection_unittest.cpp"
)
target_include_directories(${unit_test_name} BEFORE
  PRIVATE
    "../../include"
)
target_link_libraries(${unit_test_name}
  PRIVATE
    FI::FrustumIntersector
    CGAL::CGAL
)
target_compile_definitions(${unit_test_name}
  PUBLIC
    SE_SENSOR_IMPLEMENTATION=PinholeCamera
)
gtest_add_tests(${unit_test_name} "" AUTO)

