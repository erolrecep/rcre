cmake_minimum_required(VERSION 3.20)
project(rcre VERSION 2.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(RCRE_ENABLE_METAL "Build macOS Metal GPU offload backend" ON)
option(RCRE_FREESTANDING "Build freestanding no-malloc profile" OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(RCRE_FREESTANDING)
    add_compile_definitions(RCRE_FREESTANDING=1)
endif()

set(RCRE_CORE_SRCS
    src/core/arena.c
    src/core/slice.c
    src/core/str.c
    src/core/tensor.c
    src/core/adapt.c
    src/core/threadpool.c
    src/core/prof.c
    src/core/sparse.c
    src/core/nlp.c
    src/core/io.c
    src/cv/color.c
    src/cv/geom.c
    src/cv/resize_gm.c
    src/cv/filter.c
    src/cv/fft.c
    src/cv/features.c
    src/cv/homography.c
    src/cv/optical_flow.c
    src/cv/nn.c
    src/cv/codec.c
    src/cv/onnx_subset.c
    src/ml/linear.c
    src/ml/cluster.c
    src/ml/e_means.c
    src/ml/gbdt_infer.c
    src/ml/forest.c
    src/ml/hnsw.c
    src/autograd/scalar_autograd.c
    src/autograd/tensor_autograd.c
    src/nn/loss.c
    src/nn/optim.c
    src/nn/data.c
    src/nn/vis.c
    src/hal/cpu_probe.c
    src/hal/hal_dispatch.c
    src/hal/kernels_scalar.c
    src/hal/kernels_neon.c
    src/hal/kernels_avx2.c
    src/hal/kernels_avx512.c
    src/hal/kernels_wasm.c
    src/transformer/nn.c
    src/transformer/model.c
    src/transformer/kv_cache.c
    src/transformer/tokenizer.c
    src/transformer/dataset.c
    src/transformer/optim.c
    src/transformer/generate.c
    src/transformer/gguf.c
    src/transformer/llm.c
    src/gen/ae.c
    src/gen/vae.c
    src/gen/diffusion.c
    src/gen/vit_dit.c
    src/gen/lifecycle.c
)

if(APPLE)
    enable_language(OBJC)
    list(APPEND RCRE_CORE_SRCS src/extras/gpu_metal.m)
else()
    list(APPEND RCRE_CORE_SRCS src/extras/gpu_stub.c)
endif()

add_library(rcre STATIC ${RCRE_CORE_SRCS})
add_library(rcre_shared SHARED ${RCRE_CORE_SRCS})

set_target_properties(rcre_shared PROPERTIES
    OUTPUT_NAME rcre
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
)

foreach(tgt rcre rcre_shared)
    target_include_directories(${tgt} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    if(UNIX AND NOT APPLE)
        target_link_libraries(${tgt} PUBLIC m pthread)
    elseif(APPLE)
        target_link_libraries(${tgt} PUBLIC m "-framework Metal" "-framework Foundation")
    endif()
endforeach()

add_executable(rcre_chat examples/rcre_chat.c)
target_link_libraries(rcre_chat PRIVATE rcre)

include(GNUInstallDirs)
install(TARGETS rcre rcre_shared rcre_chat
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
configure_file(packaging/rcre.pc.in ${CMAKE_CURRENT_BINARY_DIR}/rcre.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rcre.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
