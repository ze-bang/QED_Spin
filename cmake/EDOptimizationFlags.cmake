# =============================================================================
# cmake/EDOptimizationFlags.cmake
#
# Builds the global CPU_OPT_FLAGS list used as $<COMPILE_LANGUAGE:CXX>
# compile options on the executable targets.
#
# Exposed cache option:
#   ED_ENABLE_FAST_MATH   default OFF.  See P0.16 / audit Q12: -ffast-math
#                         breaks NaN/Inf checks, signed-zero handling, and
#                         reduction associativity. Opt in only when you have
#                         a benchmark that justifies the numerical risk.
#
# Sets in parent (directory) scope:
#   CPU_OPT_FLAGS         the list of -O3 / -march=native / -funroll-loops /
#                         -ftree-vectorize / -fomit-frame-pointer / GCC- or
#                         Clang-specific extras / optionally -ffast-math.
#
# Application of CPU_OPT_FLAGS to specific targets is done in CMakeLists.txt
# (after the targets are defined) via target_compile_options(); this file
# only computes the list.
#
# P0.16 / P1.1 / audit Q5+Q12.
# =============================================================================

option(ED_ENABLE_FAST_MATH
    "Enable -ffast-math (UNSAFE: breaks NaN handling, signed zero, reductions). \
P0.16 / audit Q12."
    OFF)

# The instruction set the objects are compiled for. "native" is fastest but ties the
# binary to the BUILD host: a module built on one node type can die with SIGILL, or
# round differently, on another (login vs compute, CPU vs GPU partition). Set e.g.
# -DED_MARCH=x86-64-v3 for a binary that must run on every node of a mixed cluster.
set(ED_MARCH "native" CACHE STRING "Value passed to -march / -mtune (default: native)")

set(CPU_OPT_FLAGS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(ED_MARCH STREQUAL "native")
        set(_ed_mtune native)
    else()
        set(_ed_mtune generic)
    endif()
    list(APPEND CPU_OPT_FLAGS
        -O3                          # Maximum optimization
        -march=${ED_MARCH}           # Instruction set (see ED_MARCH above)
        -mtune=${_ed_mtune}          # Scheduling model
        -funroll-loops               # Unroll loops for better performance
        -ftree-vectorize             # Enable auto-vectorization
        -fomit-frame-pointer         # Remove frame pointer for extra register
    )

    if(ED_ENABLE_FAST_MATH)
        list(APPEND CPU_OPT_FLAGS -ffast-math)
        message(WARNING
            "ED_ENABLE_FAST_MATH=ON: -ffast-math added to CPU_OPT_FLAGS. "
            "Krylov breakdown checks and NaN tracking may misbehave; verify "
            "ctest is still green and that physical observables agree to "
            "your reference precision before trusting results.")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND CPU_OPT_FLAGS
            -fprefetch-loop-arrays   # Enable prefetching
            -fipa-pta                # Inter-procedural pointer analysis
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND CPU_OPT_FLAGS
            -fvectorize              # Explicit vectorization flag
        )
    endif()

    message(STATUS "  CPU Optimization Flags: ${CPU_OPT_FLAGS}")
endif()
