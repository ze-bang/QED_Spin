# =============================================================================
# python/qed/CMakeLists.txt
#
# Build target: pybind11 module `qed._core`.
#
# Linked against the project's CPU static-library stack (ed_solvers_cpu) so
# the binding inherits BLAS/LAPACK/HDF5/OpenMP transitively. When the
# top-level build sets ``WITH_CUDA=ON`` we additionally PRIVATE-link
# ``ed_solvers_gpu`` so the dispatcher header (`ed_wrapper.h`) -- which
# `dispatcher_bindings.cpp` ``#include``s and which references
# ``GPUEDWrapper::*`` symbols inside ``#ifdef WITH_CUDA`` blocks at
# compile time -- can resolve them at link time. The Python module
# itself never instantiates GPU code from Python; it just routes
# ``params.use_gpu = True`` calls into the directory / streaming-symmetry
# dispatchers, which then call into the CUDA kernels.
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
        _bindings/dispatcher_bindings.cpp
        _bindings/sector_bindings.cpp
        _bindings/little_group_bindings.cpp
        _bindings/workflow_bindings.cpp
    )
else()
    pybind11_add_module(_core MODULE
        _bindings/qed_bindings.cpp
        _bindings/input_bindings.cpp
        _bindings/dispatcher_bindings.cpp
        _bindings/sector_bindings.cpp
        _bindings/little_group_bindings.cpp
        _bindings/workflow_bindings.cpp
    )
endif()

target_link_libraries(_core PRIVATE
    ed_solvers_cpu
    ed_dssf
    ed_symmetry
    ed_input
)

# WP9.8: `_core.dssf_run` runs the `ED dssf` subcommand in-process through
# `ed::dssf::run_cli` (src/cli/dssf_engine.cpp), so the extension carries the
# CLI workflow bodies and the EDConfig command-line parser. ed_cli is a
# static, PIC library with no `main` (that stays in src/apps/ed_main.cpp).
# In CUDA builds ed_cli PUBLIC-links ed_solvers_gpu, which is linked just
# below anyway, so no new GPU dependency enters the module.
target_link_libraries(_core PRIVATE ed_cli)

# When the project was configured with WITH_CUDA=ON, the dispatcher header
# `ed_wrapper.h` references GPUEDWrapper::* symbols (gated by
# `#ifdef WITH_CUDA`, which is set via `add_definitions(-DWITH_CUDA)` in
# `cmake/EDDependencies.cmake`). We must PRIVATE-link ed_solvers_gpu (and
# ed_distributed_gpu when NCCL is present) so the linker can resolve them.
if(WITH_CUDA AND TARGET ed_solvers_gpu)
    target_link_libraries(_core PRIVATE ed_solvers_gpu)
    if(TARGET ed_distributed_gpu)
        target_link_libraries(_core PRIVATE ed_distributed_gpu)
    endif()
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
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../edlib/ DESTINATION python/edlib
            FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
endif()
