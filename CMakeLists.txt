cmake_minimum_required(VERSION 3.26)
project(tensorlib C)

set(CMAKE_C_STANDARD 99)

add_library(tensorlib STATIC src/Tensor.c include/tensor.h)
target_include_directories(tensorlib PUBLIC include)

enable_testing()

add_executable(test_tensor tests/test_tensor.c)
target_link_libraries(test_tensor PRIVATE tensorlib)

add_test(NAME test_tensor COMMAND test_tensor)
