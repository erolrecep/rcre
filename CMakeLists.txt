cmake_minimum_required(VERSION 3.25)
project(rcre VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler Warnings
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Core Library Sources
set(RCRE_SRCS
    src/core/arena.c
    src/core/tensor.c
    src/core/adapt.c
    src/core/threadpool.c
    src/cv/color.c
    src/cv/filter.c
    src/ml/linear.c
    src/ml/cluster.c
    src/ml/gbdt_infer.c
    src/ml/hnsw.c
    src/hal/cpu_probe.c
    src/hal/hal_dispatch.c
    src/hal/kernels_scalar.c
    src/hal/kernels_neon.c
    src/hal/kernels_avx2.c
    src/hal/kernels_wasm.c
    src/extras/cam_core.c
    src/extras/cam_dummy.c
)

if(APPLE)
    enable_language(OBJC)
    list(APPEND RCRE_SRCS src/extras/cam_avf.m)
    set_source_files_properties(src/extras/cam_avf.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")
elseif(UNIX)
    list(APPEND RCRE_SRCS src/extras/cam_v4l2.c)
endif()

add_library(rcre STATIC ${RCRE_SRCS})

target_include_directories(rcre PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hal>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/extras>
    $<INSTALL_INTERFACE:include>
)

find_package(Threads REQUIRED)
target_link_libraries(rcre PUBLIC Threads::Threads)
if(APPLE)
    target_link_libraries(rcre PUBLIC
        "-framework AVFoundation"
        "-framework CoreVideo"
        "-framework CoreMedia"
        "-framework Foundation"
    )
elseif(UNIX)
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
    test_hal
    test_cluster
    test_gbdt
    test_hnsw
    test_cam
)

foreach(test_name ${TESTS})
    add_executable(${test_name} tests/${test_name}.c)
    target_link_libraries(${test_name} PRIVATE rcre)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Amalgamated Header Target
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/dist/rcre.h
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/amalgamate.py
    DEPENDS ${RCRE_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/include/rcre.h
    COMMENT "Amalgamating RCRE single-header release..."
)
add_custom_target(dist ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dist/rcre.h)

add_executable(test_amalgamated tests/test_amalgamated.c)
add_dependencies(test_amalgamated dist)
target_include_directories(test_amalgamated PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dist)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_amalgamated PRIVATE m)
endif()
add_test(NAME test_amalgamated COMMAND test_amalgamated)

# Example Binaries
set(EXAMPLES
    webcam_edge_filter
    tabular_gbdt_predict
    hnsw_vector_search
)

foreach(example_name ${EXAMPLES})
    add_executable(${example_name} examples/${example_name}.c)
    target_link_libraries(${example_name} PRIVATE rcre)
endforeach()
