# =============================================================================
# python/qed/CMakeLists.txt
#
# Build target: pybind11 module `qed._core`.
#
# Linked against the project's CPU static-library stack (ed_solvers_cpu) so
# the binding inherits BLAS/LAPACK/HDF5/OpenMP transitively. When the
# top-level build sets ``WITH_CUDA=ON`` we additionally PRIVATE-link
# ``ed_solvers_gpu``, which holds the device lanes (the operators'
# ``bind_cuda()`` CudaMatVecBackend mirrors, the rep-sector mirror, the
# little-group GPU eigensolver) that the workflow bindings dispatch into.
#
# When built via scikit-build-core (pip install [-e] .), the binary lands in
# `python/qed/_core*.so` next to `__init__.py`. When built via the
# top-level CMake with -DED_BUILD_PYTHON=ON, the binary lands in
# `<build>/python/qed/_core*.so` for in-tree experimentation.
# =============================================================================

include(EDPybind11)

# When building with CUDA we must NOT let pybind11 turn on LTO (the
# default behavior of `pybind11_add_module`): GCC's LTO discards the
# synthetic ``cmake_device_link.o`` produced by NVCC's `nvlink` step,
# leaving an unresolvable ``fatbinData`` reference in `_core.so`. The
# `NO_EXTRAS` keyword keeps the module fully functional otherwise --
# we just lose visibility=hidden and strip-on-build, neither of which
# matter for an in-process Python extension. CPU-only builds keep the
# default extras (smallest possible wheel, hidden symbols).
if(WITH_CUDA AND TARGET ed_solvers_gpu)
    pybind11_add_module(_core MODULE NO_EXTRAS
        _bindings/qed_bindings.cpp
        _bindings/input_bindings.cpp
        _bindings/sectors_bindings.cpp
    )
else()
    pybind11_add_module(_core MODULE
        _bindings/qed_bindings.cpp
        _bindings/input_bindings.cpp
        _bindings/sectors_bindings.cpp
    )
endif()

target_link_libraries(_core PRIVATE
    ed_solvers_cpu
    ed_dssf
    ed_symmetry
    ed_input
)

# When the project was configured with WITH_CUDA=ON, the bindings reference
# the device-lane symbols in ed_solvers_gpu (gated by `#ifdef WITH_CUDA`,
# which is set via `add_definitions(-DWITH_CUDA)` in
# `cmake/EDDependencies.cmake`). We must PRIVATE-link ed_solvers_gpu so the
# linker can resolve them.
if(WITH_CUDA AND TARGET ed_solvers_gpu)
    target_link_libraries(_core PRIVATE ed_solvers_gpu)
    # ed_solvers_gpu is built with CUDA_SEPARABLE_COMPILATION=ON (device
    # functions across .cu translation units link via cudadevrt). Python
    # loads `_core.so` standalone, so we must perform device linking here
    # to bake the device code into the module -- otherwise dlopen fails
    # with `undefined symbol: fatbinData`. We also disable IPO/LTO on
    # this target because LTO discards the synthetic ``cmake_device_link.o``
    # object that defines ``fatbinData``, putting us right back where we
    # started.
    set_target_properties(_core PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION OFF
    )
endif()

target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)

# The extension is a BUILD product and never lands in the source tree: two
# configurations (CPU and CUDA, say) would otherwise overwrite each other's
# python/qed/_core*.so, and whichever built last would silently win. It goes to
# <build>/python/qed/, and the source package finds it through QED_CORE_DIR
# (see python/qed/__init__.py):
#
#     export PYTHONPATH=<src>/python  QED_CORE_DIR=<build>/python/qed
#
# `cmake --install` and the wheel place it inside the installed package, where
# no environment variable is needed.
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/qed
)

if(DEFINED SKBUILD_PROJECT_NAME)
    # scikit-build-core: the wheel install prefix is site-packages.
    install(TARGETS _core LIBRARY DESTINATION qed)
else()
    install(TARGETS _core LIBRARY DESTINATION python/qed)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION python/qed
            FILES_MATCHING PATTERN "*.py"
            PATTERN "_bindings" EXCLUDE PATTERN "__pycache__" EXCLUDE)
endif()
