# =============================================================================
# examples/CMakeLists.txt
#
# Builds the runnable C++ examples in this directory. Off by default
# (cmake -DED_BUILD_EXAMPLES=ON to opt in). The MPI examples are skipped
# when WITH_MPI=OFF; the GPU example is skipped when WITH_CUDA=OFF.
#
# Usage from the top-level build:
#
#     cmake -B build -DED_BUILD_EXAMPLES=ON ...
#     cmake --build build --target ed_examples
# =============================================================================

if(NOT ED_BUILD_EXAMPLES)
    return()
endif()

set(_ed_example_targets "")

function(ed_add_example name source)
    add_executable(${name} ${source})
    if(WITH_CUDA)
        # `ed_solvers_cpu` compiles orchestrator.cpp + the kernel
        # templates with WITH_CUDA defined; the resulting object file
        # references cudaMalloc / cudaFree / etc. Even examples that
        # never run on the GPU need to link CUDA libs so the
        # instantiated symbols resolve. Linking `ed_solvers_gpu`
        # PUBLIC-pulls the CUDA toolchain.
        target_link_libraries(${name} PRIVATE ed_solvers_gpu ed_solvers_cpu)
    else()
        target_link_libraries(${name} PRIVATE ed_solvers_cpu)
    endif()
    # Wave A (Full unified-interface collapse, May 2026):
    # `make_operator.h` references the DistributedOperator /
    # DistributedSymmetryOperator constructors in inline factory bodies
    # under `WITH_MPI`. Link transitively when the distributed library
    # is in scope.
    if(WITH_MPI AND TARGET ed_distributed)
        target_link_libraries(${name} PRIVATE ed_distributed)
    endif()
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    list(APPEND _ed_example_targets ${name})
    set(_ed_example_targets ${_ed_example_targets} PARENT_SCOPE)
endfunction()

ed_add_example(ex00_unified_interface   00_unified_interface.cpp)
ed_add_example(ex01_cpp_ground_state    01_cpp_ground_state.cpp)
ed_add_example(ex02_cpp_full_spectrum   02_cpp_full_spectrum.cpp)
ed_add_example(ex03_cpp_ftlm_thermal    03_cpp_ftlm_thermal.cpp)

if(WITH_CUDA)
    add_executable(ex04_cpp_gpu_lanczos 04_cpp_gpu_lanczos.cpp)
    target_link_libraries(ex04_cpp_gpu_lanczos PRIVATE
        ed_solvers_gpu
        ed_solvers_cpu
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
    )
    set_target_properties(ex04_cpp_gpu_lanczos PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    list(APPEND _ed_example_targets ex04_cpp_gpu_lanczos)
endif()

if(WITH_MPI)
    function(ed_add_mpi_example name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE ed_distributed ed_solvers_cpu MPI::MPI_CXX)
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endfunction()

    ed_add_mpi_example(ex05_mpi_distributed_lanczos      05_mpi_distributed_lanczos.cpp)
    ed_add_mpi_example(ex06_mpi_distributed_eigenvectors 06_mpi_distributed_eigenvectors.cpp)
    ed_add_mpi_example(ex07_mpi_distributed_ftlm         07_mpi_distributed_ftlm.cpp)
    ed_add_mpi_example(ex08_mpi_distributed_tpq          08_mpi_distributed_tpq.cpp)

    list(APPEND _ed_example_targets
        ex05_mpi_distributed_lanczos
        ex06_mpi_distributed_eigenvectors
        ex07_mpi_distributed_ftlm
        ex08_mpi_distributed_tpq
    )
endif()

add_custom_target(ed_examples DEPENDS ${_ed_example_targets})
message(STATUS "Examples enabled: build with `cmake --build build --target ed_examples`.")
