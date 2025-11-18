cmake_minimum_required(VERSION 4.0)
project(equation-solver)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)

add_executable(equation-solver main.cpp
        old_helper_functions.h
        old_helper_functions.cpp
        constants.h)
target_link_libraries(equation-solver ${OpenCV_LIBS})