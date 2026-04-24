# =============================================================================
# python/quantum_ed/CMakeLists.txt
#
# Build target: pybind11 module `quantum_ed._core`.
#
# Linked against the project's CPU static-library stack (ed_solvers_cpu) so
# the binding inherits BLAS/LAPACK/ARPACK/HDF5/OpenMP transitively. The GPU
# stack is NOT linked here -- the Python module stays CPU-only for portability
# (a future P2.x task will introduce a separate `_core_gpu` extension).
#
# When built via scikit-build-core (pip install [-e] .), the binary lands in
# `python/quantum_ed/_core*.so` next to `__init__.py`. When built via the
# top-level CMake with -DED_BUILD_PYTHON=ON, the binary lands in
# `<build>/python/quantum_ed/_core*.so` for in-tree experimentation.
# =============================================================================

include(EDPybind11)

pybind11_add_module(_core MODULE
    _bindings/quantum_ed_bindings.cpp
)

target_link_libraries(_core PRIVATE
    ed_solvers_cpu
    ed_dssf
)

target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)

# Ensure the .so ends up beside the package's __init__.py so editable
# installs and in-tree imports both work.
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# scikit-build-core installs the extension into the wheel under
# quantum_ed/_core*.so; ED_PYTHON_INSTALL_DIR is the wheel install prefix
# scikit-build-core selects (defaults to site-packages/quantum_ed).
if(DEFINED SKBUILD_PROJECT_NAME)
    install(TARGETS _core LIBRARY DESTINATION quantum_ed)
endif()
