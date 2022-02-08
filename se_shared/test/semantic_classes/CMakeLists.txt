cmake_minimum_required(VERSION 3.9...3.16)

add_executable(semantic-classes-unittest "semantic_classes_unittest.cpp")
gtest_add_tests(semantic-classes-unittest "" AUTO)

