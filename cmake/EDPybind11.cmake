# =============================================================================
# cmake/EDPybind11.cmake
#
# Discovers pybind11 (with FetchContent fallback) and exposes the imported
# target `pybind11::module` for building Python extensions. Used by
# python/quantum_ed/CMakeLists.txt when ED_BUILD_PYTHON=ON.
#
# P2.7 / audit "modern python interface".
# =============================================================================

# Pin a known-good pybind11 version. Bumping requires re-testing scikit-build-core
# integration end-to-end (pip install --editable .).
set(_ED_PYBIND11_TAG "v2.13.6")

find_package(pybind11 2.10 QUIET CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found system-wide; fetching ${_ED_PYBIND11_TAG} via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        ${_ED_PYBIND11_TAG}
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "pybind11 found: ${pybind11_VERSION}")
endif()
