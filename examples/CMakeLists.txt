# =============================================================================
# examples/CMakeLists.txt
#
# Builds the runnable C++ examples. Off by default
# (`cmake -DED_BUILD_EXAMPLES=ON` to opt in). The MPI examples are
# skipped when WITH_MPI=OFF; the GPU examples are skipped when
# WITH_CUDA=OFF.
#
# Two layers coexist during the transition (May 2026):
#
#   * Legacy numbered examples (`00_*.cpp` .. `08_*.cpp`) build under
#     their original names: `ex00_unified_interface`, `ex01_cpp_ground_state`,
#     ...
#   * New per-cell tree under `solve/{lanczos,block_lanczos,krylov_schur,full}/`
#     (PR-2), `thermal/{ftlm,ltlm,mtpq,ctpq,kpm_dos}/` (PR-3), and
#     `spectral/{single_expectation,ground_state_dssf,static_thermal,
#     dynamical_thermal}/` (PR-4). Each cell file is dispatched to the
#     correct link list based on its filename prefix:
#
#         cpu_*       -> ed_solvers_cpu only
#         gpu_*       -> ed_solvers_gpu (CPU path included)
#         mpi_*       -> ed_distributed + MPI
#         mpi_gpu_*   -> ed_distributed + MPI + CUDA
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

# -----------------------------------------------------------------------------
# Helpers: declare each example with the right link list based on its
# filename prefix (cpu_ / gpu_ / mpi_ / mpi_gpu_).
# -----------------------------------------------------------------------------

function(_ed_example_link target lane)
    if(lane STREQUAL "cpu")
        if(WITH_CUDA)
            # When WITH_CUDA is on, every TU compiles with cudaMalloc/Free
            # symbols visible (orchestrator.cpp instantiates kernel templates),
            # so even the cpu_* cells link the CUDA toolchain via ed_solvers_gpu.
            target_link_libraries(${target} PRIVATE ed_solvers_gpu ed_solvers_cpu)
        else()
            target_link_libraries(${target} PRIVATE ed_solvers_cpu)
        endif()
    elseif(lane STREQUAL "gpu")
        target_link_libraries(${target} PRIVATE
            ed_solvers_gpu ed_solvers_cpu
            CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand)
    elseif(lane STREQUAL "mpi")
        target_link_libraries(${target} PRIVATE ed_distributed ed_solvers_cpu MPI::MPI_CXX)
    elseif(lane STREQUAL "mpi_gpu")
        if(TARGET ed_distributed_gpu)
            target_link_libraries(${target} PRIVATE
                ed_distributed_gpu ed_distributed
                ed_solvers_gpu ed_solvers_cpu MPI::MPI_CXX
                CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand)
        else()
            target_link_libraries(${target} PRIVATE
                ed_distributed ed_solvers_cpu MPI::MPI_CXX
                CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand)
        endif()
    endif()

    # Wave A pattern: pull in ed_distributed transitively when MPI is on so
    # the WITH_MPI branches of `ed::make_operator` resolve.
    if(WITH_MPI AND TARGET ed_distributed AND NOT lane MATCHES "mpi")
        target_link_libraries(${target} PRIVATE ed_distributed)
    endif()
endfunction()

# Detect the lane of a per-cell example from its filename prefix.
function(_ed_example_lane outvar filename)
    get_filename_component(_stem "${filename}" NAME_WE)
    if(_stem MATCHES "^mpi_gpu_")
        set(${outvar} "mpi_gpu" PARENT_SCOPE)
    elseif(_stem MATCHES "^mpi_")
        set(${outvar} "mpi" PARENT_SCOPE)
    elseif(_stem MATCHES "^gpu_")
        set(${outvar} "gpu" PARENT_SCOPE)
    elseif(_stem MATCHES "^cpu_")
        set(${outvar} "cpu" PARENT_SCOPE)
    else()
        set(${outvar} "cpu" PARENT_SCOPE)
    endif()
endfunction()

# Map a cell file (e.g. examples/solve/lanczos/cpu_sz.cpp) to a unique
# CMake target name (`ex_solve_lanczos_cpu_sz`).
function(_ed_example_target_name outvar source_path)
    file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${source_path})
    string(REGEX REPLACE "\\.cpp$" "" _stem ${_rel})
    string(REPLACE "/" "_" _flat ${_stem})
    set(${outvar} "ex_${_flat}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# Per-cell example registration (PR-2 onwards).
# -----------------------------------------------------------------------------

set(_ed_cell_dirs solve thermal spectral)
foreach(family IN LISTS _ed_cell_dirs)
    file(GLOB_RECURSE _family_sources
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/${family}/*.cpp)

    foreach(_cell_rel IN LISTS _family_sources)
        set(_cell_abs ${CMAKE_CURRENT_SOURCE_DIR}/${_cell_rel})
        _ed_example_lane(_lane ${_cell_rel})
        _ed_example_target_name(_target ${_cell_abs})

        # Skip MPI / GPU / NCCL cells when the build can't service them.
        if(_lane STREQUAL "gpu" AND NOT WITH_CUDA)
            continue()
        endif()
        if(_lane STREQUAL "mpi" AND NOT WITH_MPI)
            continue()
        endif()
        if(_lane STREQUAL "mpi_gpu" AND NOT (WITH_MPI AND WITH_CUDA))
            continue()
        endif()

        add_executable(${_target} ${_cell_rel})
        target_include_directories(${_target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})
        _ed_example_link(${_target} ${_lane})
        set_target_properties(${_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        list(APPEND _ed_example_targets ${_target})
    endforeach()
endforeach()

# -----------------------------------------------------------------------------
# Legacy numbered examples (frozen as of the May 2026 mirror-examples
# overhaul; see `_legacy/README.md`). Kept building so the CLI / NLCE
# shell scripts in `_legacy/` still have artifacts to drive.
# -----------------------------------------------------------------------------

function(ed_add_example name source)
    add_executable(${name} ${source})
    if(WITH_CUDA)
        target_link_libraries(${name} PRIVATE ed_solvers_gpu ed_solvers_cpu)
    else()
        target_link_libraries(${name} PRIVATE ed_solvers_cpu)
    endif()
    if(WITH_MPI AND TARGET ed_distributed)
        target_link_libraries(${name} PRIVATE ed_distributed)
    endif()
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    list(APPEND _ed_example_targets ${name})
    set(_ed_example_targets ${_ed_example_targets} PARENT_SCOPE)
endfunction()

ed_add_example(ex00_unified_interface   _legacy/00_unified_interface.cpp)
ed_add_example(ex01_cpp_ground_state    _legacy/01_cpp_ground_state.cpp)
ed_add_example(ex02_cpp_full_spectrum   _legacy/02_cpp_full_spectrum.cpp)
ed_add_example(ex03_cpp_ftlm_thermal    _legacy/03_cpp_ftlm_thermal.cpp)

if(WITH_CUDA)
    add_executable(ex04_cpp_gpu_lanczos _legacy/04_cpp_gpu_lanczos.cpp)
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

    ed_add_mpi_example(ex05_mpi_distributed_lanczos      _legacy/05_mpi_distributed_lanczos.cpp)
    ed_add_mpi_example(ex06_mpi_distributed_eigenvectors _legacy/06_mpi_distributed_eigenvectors.cpp)
    ed_add_mpi_example(ex07_mpi_distributed_ftlm         _legacy/07_mpi_distributed_ftlm.cpp)
    ed_add_mpi_example(ex08_mpi_distributed_tpq          _legacy/08_mpi_distributed_tpq.cpp)

    list(APPEND _ed_example_targets
        ex05_mpi_distributed_lanczos
        ex06_mpi_distributed_eigenvectors
        ex07_mpi_distributed_ftlm
        ex08_mpi_distributed_tpq
    )
endif()

add_custom_target(ed_examples DEPENDS ${_ed_example_targets})

# -----------------------------------------------------------------------------
# Smoke target: build only the CPU per-cell binaries (the new tree's
# deterministic / regression-testable subset). The Python twin smoke
# harness is `scripts/check_examples_output.py`, which is wired into
# CI by `.github/workflows/ci.yml::linux-examples-smoke`.
# -----------------------------------------------------------------------------
set(_ed_smoke_targets "")
foreach(_target IN LISTS _ed_example_targets)
    if(_target MATCHES "^ex_(solve|thermal|spectral)_.*_cpu($|_)")
        list(APPEND _ed_smoke_targets ${_target})
    endif()
endforeach()
add_custom_target(ed_examples_smoke DEPENDS ${_ed_smoke_targets})

message(STATUS "Examples enabled: build with `cmake --build build --target ed_examples` "
               "(per-cell targets: ${CMAKE_CURRENT_BINARY_DIR}/ex_solve_*, ex_thermal_*, ex_spectral_*).")
message(STATUS "  Smoke subset: `cmake --build build --target ed_examples_smoke` "
               "(only the CPU-lane cells in the new tree).")
