# =============================================================================
# cmake/EDCatch2.cmake
#
# Bring in Catch2 v3 via FetchContent (with a system-wide find_package
# fallback). Defines the imported targets:
#
#   Catch2::Catch2          core library, no main()
#   Catch2::Catch2WithMain  bundles a main() that runs the registered
#                           TEST_CASEs (use this for ed_add_test).
#
# Why FetchContent first? Most lab clusters and HPC images do not ship
# Catch2 v3 (Debian bookworm, RHEL 8/9 ship v2). Pinning to v3.5.4 makes
# the test build reproducible without imposing a system dependency.
#
# P1.8 / audit Q12.
# =============================================================================

option(ED_FETCH_CATCH2 "Always fetch Catch2 v3 via FetchContent (bypass find_package)" OFF)

if(NOT ED_FETCH_CATCH2)
    find_package(Catch2 3.0 QUIET)
endif()

if(NOT Catch2_FOUND)
    message(STATUS "Catch2 v3 not found system-wide; fetching v3.5.4 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4
        GIT_SHALLOW    TRUE
    )
    set(CATCH_INSTALL_DOCS    OFF CACHE INTERNAL "")
    set(CATCH_INSTALL_EXTRAS  OFF CACHE INTERNAL "")
    set(CATCH_BUILD_TESTING   OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
else()
    message(STATUS "Catch2 found: ${Catch2_VERSION}")
    list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
endif()

include(Catch)
