# =============================================================================
# cmake/EDPlatform.cmake
#
# Host platform detection (CPU vendor + reportable CPU features).
#
# Sets / advertises:
#   CPU_VENDOR  - "AMD" | "Intel" | "Unknown"
#   CPU_COUNT   - logical CPU count   (x86_64 only, may be empty otherwise)
#   CPU_MODEL   - human-readable model string (x86_64 only, may be empty)
# Prints whether the build host advertises AVX2 / FMA in /proc/cpuinfo.
#
# This file is included from the top-level CMakeLists.txt; it intentionally
# does no find_package() and sets no link-time state -- it just gathers
# information used by EDBlasProfile.cmake (default profile selection) and by
# the build-summary block at the end of CMakeLists.txt.
#
# P1.1 / audit Q5.
# =============================================================================

# -----------------------------------------------------------------------------
# CPU vendor (used by BLAS_PROFILE auto-resolution).
# -----------------------------------------------------------------------------
set(CPU_VENDOR "Unknown")
if(EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" CPUINFO_RAW)
    if(CPUINFO_RAW MATCHES "AuthenticAMD")
        set(CPU_VENDOR "AMD")
    elseif(CPUINFO_RAW MATCHES "GenuineIntel")
        set(CPU_VENDOR "Intel")
    endif()
endif()
message(STATUS "Detected CPU vendor: ${CPU_VENDOR}")

# -----------------------------------------------------------------------------
# Reportable CPU features (purely informational; no compile-time dispatch).
# Guarded so that non-Linux hosts and non-x86 CPUs don't try to grep
# /proc/cpuinfo.
# -----------------------------------------------------------------------------
function(_ed_report_cpu_features)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        return()
    endif()
    if(NOT EXISTS "/proc/cpuinfo")
        return()
    endif()

    execute_process(COMMAND grep -c "^processor" /proc/cpuinfo
        OUTPUT_VARIABLE _cpu_count OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND grep "model name" /proc/cpuinfo
                    COMMAND head -1
                    COMMAND sed "s/.*: //"
        OUTPUT_VARIABLE _cpu_model OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CPU_COUNT "${_cpu_count}" PARENT_SCOPE)
    set(CPU_MODEL "${_cpu_model}" PARENT_SCOPE)
    message(STATUS "  CPU: ${_cpu_model} (${_cpu_count} cores)")

    execute_process(COMMAND grep -q "avx2" /proc/cpuinfo
        RESULT_VARIABLE _avx2_rc OUTPUT_QUIET ERROR_QUIET)
    if(_avx2_rc EQUAL 0)
        message(STATUS "  AVX2: Available")
    endif()

    execute_process(COMMAND grep -q "fma" /proc/cpuinfo
        RESULT_VARIABLE _fma_rc OUTPUT_QUIET ERROR_QUIET)
    if(_fma_rc EQUAL 0)
        message(STATUS "  FMA: Available")
    endif()
endfunction()
