cmake_minimum_required(VERSION 3.26)
project(tensorlib C)

set(CMAKE_C_STANDARD 99)

add_executable(tensorlib src/Tensor.c
        include/tensor.h)
