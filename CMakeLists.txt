cmake_minimum_required(VERSION 3.25)
project(rcre VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler Warnings
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Core Library
add_library(rcre STATIC
    src/core/arena.c
    src/core/tensor.c
    src/core/adapt.c
)

target_include_directories(rcre PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable Testing
enable_testing()

add_executable(test_arena tests/test_arena.c)
target_link_libraries(test_arena PRIVATE rcre)
add_test(NAME test_arena COMMAND test_arena)

add_executable(test_tensor_strides tests/test_tensor_strides.c)
target_link_libraries(test_tensor_strides PRIVATE rcre)
add_test(NAME test_tensor_strides COMMAND test_tensor_strides)

add_executable(test_adapt_spec tests/test_adapt_spec.c)
target_link_libraries(test_adapt_spec PRIVATE rcre)
add_test(NAME test_adapt_spec COMMAND test_adapt_spec)
