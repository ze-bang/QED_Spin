# =============================================================================
# benchmarks/CMakeLists.txt
#
# Micro-benchmarks built on Google Benchmark. Opt in via
# `-DED_BUILD_BENCHMARKS=ON`. Targets:
#
#   bench_operator_apply        Hamiltonian H*v matrix-free apply()
#   bench_lanczos_ground_state  ground-state Lanczos on a 1D Heisenberg ring
#   bench_full_diagonalization  dense LAPACK eigensolve up to N=10
#
# Run with:
#   cmake --preset default -DED_BUILD_BENCHMARKS=ON
#   cmake --build build --target bench_operator_apply
#   ./build/benchmarks/bench_operator_apply
#       --benchmark_format=json --benchmark_out=results.json
#
# CI nightly job (.github/workflows/benchmarks.yml) builds them and posts
# JSON results as an artifact so trends can be tracked across commits.
#
# Audit ref: P2.13.
# =============================================================================

if(NOT ED_BUILD_BENCHMARKS)
    return()
endif()

include(EDBenchmark)

# Helper that mirrors `ed_add_test` from the top-level CMakeLists.txt: every
# benchmark gets its own binary linking against ed_solvers_cpu (which
# transitively pulls in ed_core, ed_io, BLAS/LAPACK/HDF5/ARPACK).
function(ed_add_benchmark bench_name bench_source)
    add_executable(${bench_name} ${bench_source})
    target_link_libraries(${bench_name} PRIVATE
        ed_solvers_cpu
        benchmark::benchmark_main
    )
    target_include_directories(${bench_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(${bench_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    )
    set_target_properties(${bench_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endfunction()

ed_add_benchmark(bench_operator_apply       bench_operator_apply.cpp)
ed_add_benchmark(bench_lanczos_ground_state bench_lanczos_ground_state.cpp)
ed_add_benchmark(bench_full_diagonalization bench_full_diagonalization.cpp)

# GPU benchmarks: only built when WITH_CUDA=ON. The C++ files compile
# unconditionally (they short-circuit to a SkipWithError when CUDA is not
# available at runtime), but they need ed_solvers_gpu and the cuSPARSE /
# cuBLAS link line, which only exist in the CUDA build.
set(_ed_bench_targets
    bench_operator_apply
    bench_lanczos_ground_state
    bench_full_diagonalization
)
if(WITH_CUDA)
    function(ed_add_gpu_benchmark bench_name bench_source)
        add_executable(${bench_name} ${bench_source})
        target_link_libraries(${bench_name} PRIVATE
            ed_solvers_gpu
            ed_solvers_cpu
            benchmark::benchmark
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
        )
        target_include_directories(${bench_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_target_properties(${bench_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endfunction()

    ed_add_gpu_benchmark(bench_gpu_operator_apply       bench_gpu_operator_apply.cpp)
    ed_add_gpu_benchmark(bench_gpu_lanczos_ground_state bench_gpu_lanczos_ground_state.cpp)

    list(APPEND _ed_bench_targets
        bench_gpu_operator_apply
        bench_gpu_lanczos_ground_state
    )
endif()

# Convenience meta-target that pulls in everything so a single
# `cmake --build build --target ed_benchmarks` builds the lot.
add_custom_target(ed_benchmarks DEPENDS ${_ed_bench_targets})

message(STATUS
    "Benchmarks enabled: build with `cmake --build build --target ed_benchmarks`."
)
