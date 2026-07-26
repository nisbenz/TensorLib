cmake_minimum_required(VERSION 3.26)
project(tensorlib C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native -mtune=native")
endif()

set(TENSORLIB_SOURCES
    src/tensor/tensor_core.c
    src/tensor/tensor_ops.c
    src/tensor/tensor_alloc.c
    src/tensor/tensor_view.c
    src/tensor/tensor_gather.c
    src/tensor/tensor_reduc.c
    src/tensor/tensor_matmul.c
    src/autograd/autograd_core.c
    src/autograd/autograd_ops.c
    src/autograd/autograd_view.c
    src/autograd/autograd_gather.c
    src/autograd/autograd_reduc.c
    src/autograd/autograd_matmul.c
    src/autograd/autograd_backward.c
    src/init/rng.c
    src/nn/parameter.c
    src/nn/module.c
    src/nn/linear.c
    src/nn/embedding.c
    src/nn/mlp.c
    src/losses/classification.c
    src/optim/sgd.c
    include/tensorlib/tensor_matmul.h
)

add_library(tensorlib STATIC ${TENSORLIB_SOURCES}
            include/tensorlib/tensor.h
            include/tensorlib/autograd.h
            include/tensorlib/nn.h)
target_include_directories(tensorlib PUBLIC include/tensorlib)

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
    autograd_core
    autograd_ops
    autograd_view
    autograd_gather
    autograd_reduc
    autograd_matmul
    autograd_backward
    autograd_integration
    autograd_public_contract
    nn_rng
    nn_parameter
    nn_module
    nn_init
    nn_linear
    nn_embedding
    nn_loss
    nn_sgd
    nn_mlp
)

foreach(test_name IN LISTS TENSORLIB_TESTS)
    if (test_name STREQUAL "nn_sgd")
        set(test_source tests/unit/optim/test_${test_name}.c)
    elseif(test_name MATCHES "^nn_")
        set(test_source tests/unit/nn/test_${test_name}.c)
    elseif(test_name MATCHES "^autograd_")
        set(test_source tests/unit/autograd/test_${test_name}.c)
    else()
        set(test_source tests/unit/tensor/test_${test_name}.c)
    endif()
    add_executable(test_${test_name} ${test_source})
    target_include_directories(test_${test_name} PRIVATE tests/fixtures)
    if (test_name STREQUAL "autograd_public_contract" OR test_name MATCHES "^nn_")
        target_include_directories(test_${test_name} PRIVATE include)
    endif()
    target_link_libraries(test_${test_name} PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
    add_test(NAME test_${test_name} COMMAND test_${test_name})
endforeach()
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/autograd_example.c")
    add_executable(autograd_example examples/autograd_example.c)
    target_link_libraries(autograd_example PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/mnsit/mnist_mlp.c")
    add_executable(mnist_mlp examples/mnsit/mnist_mlp.c)
    target_link_libraries(mnist_mlp PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/matmul/bench_tensor_matmul.c")
    add_executable(bench_tensor_matmul benchmarks/matmul/bench_tensor_matmul.c)
    target_link_libraries(bench_tensor_matmul PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/matmul/bench_tensor_matmul_reuse.c")
    add_executable(bench_tensor_matmul_reuse benchmarks/matmul/bench_tensor_matmul_reuse.c)
    target_link_libraries(bench_tensor_matmul_reuse PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/matmul/bench_openblas_matmul.c")
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h
              HINTS C:/msys64/ucrt64/include
              PATH_SUFFIXES openblas)
    find_library(OPENBLAS_LIBRARY NAMES openblas
                 HINTS C:/msys64/ucrt64/lib)

    if (OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY)
        add_executable(bench_openblas_matmul benchmarks/matmul/bench_openblas_matmul.c)
        target_include_directories(bench_openblas_matmul PRIVATE ${OPENBLAS_INCLUDE_DIR})
        target_link_libraries(bench_openblas_matmul PRIVATE
                              tensorlib ${OPENBLAS_LIBRARY} ${TENSORLIB_MATH_LIB})

        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/matmul/bench_openblas_matmul_reuse.c")
            add_executable(bench_openblas_matmul_reuse
                           benchmarks/matmul/bench_openblas_matmul_reuse.c)
            target_include_directories(bench_openblas_matmul_reuse
                                       PRIVATE ${OPENBLAS_INCLUDE_DIR})
            target_link_libraries(bench_openblas_matmul_reuse PRIVATE
                                  tensorlib ${OPENBLAS_LIBRARY} ${TENSORLIB_MATH_LIB})
        endif()
    else()
        message(WARNING "OpenBLAS not found; bench_openblas_matmul will not be built")
    endif()
endif()
