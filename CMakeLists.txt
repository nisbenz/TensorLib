cmake_minimum_required(VERSION 3.26)
project(tensorlib C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(TENSORLIB_SOURCES
    src/tensor_core.c
    src/tensor_ops.c
    src/tensor_alloc.c
    src/tensor_view.c
    src/tensor_reduc.c
    src/tensor_matmul.c
)

add_library(tensorlib STATIC ${TENSORLIB_SOURCES} include/tensor.h)
target_include_directories(tensorlib PUBLIC include)

if (WIN32)
    set(TENSORLIB_MATH_LIB)
else()
    set(TENSORLIB_MATH_LIB m)
endif()

enable_testing()

set(TENSORLIB_TESTS
    tensor_core
    tensor_alloc
    tensor_view
    tensor_ops
    tensor_reduc
    tensor_matmul
)

foreach(test_name IN LISTS TENSORLIB_TESTS)
    add_executable(test_${test_name} tests/test_${test_name}.c)
    target_link_libraries(test_${test_name} PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
    add_test(NAME test_${test_name} COMMAND test_${test_name})
endforeach()
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/playground.c")
    add_executable(playground examples/playground.c)
    target_link_libraries(playground PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

