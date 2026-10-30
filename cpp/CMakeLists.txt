cmake_minimum_required(VERSION 3.15...3.30)
project(enumerate_plan)

set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -g -Wall -pthread -march=native")

add_subdirectory(pybind11)
pybind11_add_module(enumerate_plan enumerate_plan.cpp bindings.cpp)

add_executable(test.out test.cpp enumerate_plan.cpp)
