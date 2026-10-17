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
    src/core/threadpool.c
    src/cv/color.c
    src/cv/filter.c
    src/ml/linear.c
)

target_include_directories(rcre PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(Threads REQUIRED)
target_link_libraries(rcre PUBLIC Threads::Threads)
if(UNIX AND NOT APPLE)
    target_link_libraries(rcre PUBLIC m)
endif()

# Enable Testing
enable_testing()

set(TESTS
    test_arena
    test_tensor_strides
    test_adapt_spec
    test_threadpool
    test_cv_filters
    test_ml_linear
)

foreach(test_name ${TESTS})
    add_executable(${test_name} tests/${test_name}.c)
    target_link_libraries(${test_name} PRIVATE rcre)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
