cmake_minimum_required(VERSION 3.26)
project(tensorlib_package_consumer C)

find_package(TensorLib CONFIG REQUIRED)

add_executable(package_consumer main.c)
target_link_libraries(package_consumer PRIVATE TensorLib::tensorlib)

enable_testing()
add_test(NAME package_consumer COMMAND package_consumer)
