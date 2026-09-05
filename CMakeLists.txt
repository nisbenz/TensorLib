cmake_minimum_required(VERSION 3.26)
project(tensorlib VERSION 0.1.0 LANGUAGES C)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(TENSORLIB_ENABLE_OPENMP "Enable OpenMP parallelism" ON)
option(TENSORLIB_NATIVE_OPTIMIZATIONS "Tune release builds for this CPU" OFF)

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -fno-math-errno -fno-signed-zeros -fno-trapping-math -freciprocal-math -funroll-loops")
    if (TENSORLIB_NATIVE_OPTIMIZATIONS)
        string(APPEND CMAKE_C_FLAGS_RELEASE " -march=native -mtune=native")
    endif()
endif()

if (TENSORLIB_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
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
    src/nn/positional_embedding.c
    src/nn/layer_norm.c
    src/nn/dropout.c
    src/nn/multihead_attention.c
    src/nn/decoder_block.c
    src/nn/decoder.c
    src/nn/mlp.c
    src/losses/classification.c
    src/nn/causal_mask.c
    src/optim/sgd.c
    src/optim/optim_common.c
    src/optim/adamw.c
    src/serialization/checkpoint.c
    include/tensorlib/tensor_matmul.h
)

add_library(tensorlib STATIC ${TENSORLIB_SOURCES}
            include/tensorlib/tensor.h
            include/tensorlib/autograd.h
            include/tensorlib/nn.h)
target_include_directories(tensorlib PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/tensorlib>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

if (WIN32)
    set(TENSORLIB_MATH_LIB)
else()
    set(TENSORLIB_MATH_LIB m)
endif()

target_link_libraries(tensorlib PUBLIC ${TENSORLIB_MATH_LIB})
if (TENSORLIB_ENABLE_OPENMP)
    target_link_libraries(tensorlib PUBLIC OpenMP::OpenMP_C)
    set(TENSORLIB_FIND_OPENMP "find_dependency(OpenMP)")
else()
    set(TENSORLIB_FIND_OPENMP "")
endif()

install(TARGETS tensorlib EXPORT TensorLibTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/tensorlib DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
install(EXPORT TensorLibTargets
        FILE TensorLibTargets.cmake
        NAMESPACE TensorLib::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorLib)

configure_package_config_file(
    cmake/TensorLibConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TensorLibConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorLib)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TensorLibConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/TensorLibConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/TensorLibConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorLib)

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
    nn_positional_embedding
    nn_layer_norm
    nn_dropout
    nn_multihead_attention
    nn_decoder_block
    nn_decoder
    nn_loss
    nn_causal_mask
    nn_sgd
    nn_adamw
    nn_checkpoint
    nn_mlp
)

foreach(test_name IN LISTS TENSORLIB_TESTS)
    if (test_name MATCHES "^nn_(sgd|adamw)$")
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

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/mnist/mnist_mlp.c")
    add_executable(mnist_mlp examples/mnist/mnist_mlp.c)
    target_link_libraries(mnist_mlp PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/tiny_lm/tiny_lm.c")
    add_executable(tiny_lm examples/tiny_lm/tiny_lm.c)
    target_link_libraries(tiny_lm PRIVATE tensorlib ${TENSORLIB_MATH_LIB})
    add_test(NAME test_tiny_lm_help COMMAND tiny_lm --help)
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
              PATH_SUFFIXES openblas)
    find_library(OPENBLAS_LIBRARY NAMES openblas
                 )

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
