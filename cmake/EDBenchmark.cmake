# =============================================================================
# cmake/EDBenchmark.cmake
#
# Bring in Google Benchmark via FetchContent (with a system-wide
# `find_package` fallback). Defines the imported targets:
#
#   benchmark::benchmark          core micro-benchmarking library
#   benchmark::benchmark_main     bundles a main() that runs the registered
#                                 BENCHMARK_*() macros (use this for
#                                 ed_add_benchmark).
#
# Pinned to v1.8.5 (released 2024-08, last release tested on this codebase).
# Disable Google Benchmark's own self-tests + Google Test dep by default --
# we only want the library, not its CI infrastructure.
#
# Audit ref: P2.13.
# =============================================================================

option(ED_FETCH_BENCHMARK
    "Always fetch Google Benchmark via FetchContent (bypass find_package)"
    OFF)

if(NOT ED_FETCH_BENCHMARK)
    find_package(benchmark 1.8 QUIET)
endif()

if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found system-wide; "
                   "fetching v1.8.5 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.5
        GIT_SHALLOW    TRUE
    )
    set(BENCHMARK_ENABLE_TESTING        OFF CACHE INTERNAL "")
    set(BENCHMARK_ENABLE_INSTALL        OFF CACHE INTERNAL "")
    set(BENCHMARK_ENABLE_GTEST_TESTS    OFF CACHE INTERNAL "")
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE INTERNAL "")
    set(BENCHMARK_USE_BUNDLED_GTEST     OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(benchmark)
else()
    message(STATUS "Google Benchmark found: ${benchmark_VERSION}")
endif()
