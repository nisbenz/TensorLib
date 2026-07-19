cmake_minimum_required(VERSION 3.26)
project(tensorlib C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native -mtune=native")
endif()

set(TENSORLIB_SOURCES
    src/tensor_core.c
    src/tensor_ops.c
    src/tensor_alloc.c
    src/tensor_view.c
    src/tensor_reduc.c
    src/tensor_matmul.c
    include/tensor_matmul.h
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

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_tensor_matmul.c")
    add_executable(bench_tensor_matmul benchmarks/bench_tensor_matmul.c)
    target_link_libraries(bench_tensor_matmul PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_tensor_matmul_reuse.c")
    add_executable(bench_tensor_matmul_reuse benchmarks/bench_tensor_matmul_reuse.c)
    target_link_libraries(bench_tensor_matmul_reuse PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_openblas_matmul.c")
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h
              HINTS C:/msys64/ucrt64/include
              PATH_SUFFIXES openblas)
    find_library(OPENBLAS_LIBRARY NAMES openblas
                 HINTS C:/msys64/ucrt64/lib)

    if (OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY)
        add_executable(bench_openblas_matmul benchmarks/bench_openblas_matmul.c)
        target_include_directories(bench_openblas_matmul PRIVATE ${OPENBLAS_INCLUDE_DIR})
        target_link_libraries(bench_openblas_matmul PRIVATE
                              tensorlib ${OPENBLAS_LIBRARY} ${TENSORLIB_MATH_LIB})

        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_openblas_matmul_reuse.c")
            add_executable(bench_openblas_matmul_reuse
                           benchmarks/bench_openblas_matmul_reuse.c)
            target_include_directories(bench_openblas_matmul_reuse
                                       PRIVATE ${OPENBLAS_INCLUDE_DIR})
            target_link_libraries(bench_openblas_matmul_reuse PRIVATE
                                  tensorlib ${OPENBLAS_LIBRARY} ${TENSORLIB_MATH_LIB})
        endif()
    else()
        message(WARNING "OpenBLAS not found; bench_openblas_matmul will not be built")
    endif()
endif()
