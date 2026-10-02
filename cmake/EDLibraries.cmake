# =============================================================================
# cmake/EDLibraries.cmake -- qed_engine, the one static library: every engine source, and under
# WITH_CUDA the device sources (separable compilation: combinadic.cu's constant-memory Pascal
# table is extern in the others; the device link happens in each executable and in _core).
# Consumers link qed_engine and inherit its headers, WITH_CUDA and the link stack.
# =============================================================================

set(ED_ENGINE_SOURCES
    src/parallel/numa.cpp
    src/parallel/thread_budget.cpp
    src/ops/algebra.cpp
    src/ops/invariance.cpp
    src/ops/program.cpp
    src/basis/group.cpp
    src/basis/irreps.cpp
    src/input/lattice.cpp
    src/engine/oftlm.cpp
    src/engine/context.cpp
    src/engine/block_solve.cpp
    src/engine/stars.cpp
    src/engine/eigs.cpp
    src/engine/thermal.cpp
    src/engine/dynamics.cpp
    src/engine/expect.cpp
    src/engine/validate.cpp
    src/engine/group_sector.cpp
    src/engine/ground_state.cpp
    src/engine/tower.cpp
)
if(WITH_CUDA)
    list(APPEND ED_ENGINE_SOURCES
        src/gpu/combinadic.cu
        src/gpu/little_group.cu
        src/gpu/rep_matvec.cu
        src/gpu/rep_matrix_elements.cu
    )
else()
    list(APPEND ED_ENGINE_SOURCES src/gpu/rep_matvec_stub.cpp)   # throwing stubs
endif()

add_library(qed_engine STATIC ${ED_ENGINE_SOURCES})
target_include_directories(qed_engine PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
find_package(Threads REQUIRED)
if(WITH_CUDA)
    target_link_libraries(qed_engine PUBLIC CUDA::cudart CUDA::cublas CUDA::curand CUDA::cusolver)
endif()
target_link_libraries(qed_engine PUBLIC ${ED_LINALG_LIBRARIES} OpenMP::OpenMP_CXX Threads::Threads)
target_compile_options(qed_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(qed_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WITH_CUDA)
    target_compile_definitions(qed_engine PUBLIC WITH_CUDA)
    # Host TUs include <cuComplex.h> / <cublas_v2.h> through cuda_backend.cuh.
    target_include_directories(qed_engine PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(qed_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
