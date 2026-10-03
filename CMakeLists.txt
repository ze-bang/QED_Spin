cmake_minimum_required(VERSION 3.18)

# The QED_Spin engine: one static library (qed_engine), the pybind11 module qed._core
# (ED_BUILD_PYTHON), the Catch2 unit tests (BUILD_ED_TESTS) and the docs (ED_BUILD_DOCS).
# Entry points: scripts/build.sh on a cluster, CMakePresets.json in CI, pyproject.toml for pip.
#
#   -DWITH_CUDA=ON          build the device lanes (default OFF)
#   -DBLAS_PROFILE=...      AUTO | FLEXIBLAS | OPENBLAS | MKL (cmake/EDBlas.cmake)
#   -DED_MARCH=...          -march value (cmake/EDOptimizationFlags.cmake; default native)

option(WITH_CUDA "Build the CUDA lanes" OFF)

# CUDA architectures: H100 (sm_90) unless the caller chose them (-DCMAKE_CUDA_ARCHITECTURES
# or the CUDAARCHS environment variable). Compute capability 6.0+ is required for native
# double atomicAdd. Seeded into the cache before project(), which would otherwise cache the
# compiler default (sm_52) and have every later re-configure pick that up.
if(WITH_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures to build for")
endif()

# The version is pyproject.toml's, and only there: the project, qed._core.__version__ (and so
# qed.__version__) and the docs read it from that line.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" QED_VERSION_LINE REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$")
string(REGEX REPLACE "^version = \"([0-9.]+)\"$" "\\1" QED_VERSION "${QED_VERSION_LINE}")
if(NOT QED_VERSION)
    message(FATAL_ERROR "pyproject.toml has no line version = \"X.Y.Z\"")
endif()
if(WITH_CUDA)
    project(QED VERSION ${QED_VERSION} LANGUAGES CXX CUDA)
else()
    project(QED VERSION ${QED_VERSION} LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(WITH_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -fPIC -std=c++17")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)
include(EDBlas)                 # BLAS / LAPACK / LAPACKE
include(EDDependencies)         # Eigen3; CUDAToolkit under WITH_CUDA
include(EDOptimizationFlags)    # CPU_OPT_FLAGS
find_package(OpenMP REQUIRED)
include(EDLibraries)            # qed_engine

# =============================================================================
# Install: headers, the library and QEDConfig.cmake, so a downstream project can
#   find_package(QED CONFIG REQUIRED)
#   target_link_libraries(myapp PRIVATE QED::qed_engine)
# =============================================================================
include(CMakePackageConfigHelpers)
install(TARGETS qed_engine
    EXPORT QEDTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ed
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)
install(EXPORT QEDTargets
        FILE QEDTargets.cmake
        NAMESPACE QED::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QED
)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/QEDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QED
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QED
)

message(STATUS "QED ${PROJECT_VERSION}: ${CMAKE_BUILD_TYPE}, WITH_CUDA=${WITH_CUDA} (${CMAKE_CUDA_ARCHITECTURES}), "
               "BLAS ${BLAS_PROFILE}, flags ${CPU_OPT_FLAGS}")

# =============================================================================
# Python bindings (qed._core). `pip install .` (scikit-build-core, pyproject.toml) turns them
# on; scripts/build.sh does unless --no-python.
# =============================================================================
option(ED_BUILD_PYTHON "Build the qed pybind11 Python bindings" OFF)
if(SKBUILD)
    set(ED_BUILD_PYTHON ON CACHE BOOL "" FORCE)
endif()
if(ED_BUILD_PYTHON)
    add_subdirectory(python/qed)
endif()

# =============================================================================
# Documentation site (Doxygen + Sphinx + Breathe; pyproject.toml's docs extra): targets `doxygen`
# and `sphinx`. CI publishes it (.github/workflows/docs.yml).
# =============================================================================
option(ED_BUILD_DOCS "Build the Doxygen + Sphinx documentation site" OFF)
if(ED_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# =============================================================================
# Unit tests (Catch2 v3): one executable per file, each TEST_CASE a ctest test.
# =============================================================================
option(BUILD_ED_TESTS "Build the ED regression test suite" ON)
if(BUILD_ED_TESTS)
    enable_testing()
    include(EDCatch2)               # Catch2::Catch2, catch_discover_tests

    # Catch2's session with an exit code ctest can trust (tests/common/catch2_main.cpp): 1 on any
    # failure, 77 when every test case run was skipped. Catch2's own main exits with the failed
    # assertion count, so four failures read as its all-skipped code 4.
    add_library(ed_catch2_main OBJECT tests/common/catch2_main.cpp)
    target_link_libraries(ed_catch2_main PUBLIC Catch2::Catch2)

    function(ed_add_test test_name test_source)
        add_executable(${test_name} ${test_source})
        target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries(${test_name} PRIVATE qed_engine ed_catch2_main Catch2::Catch2)
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>)
        # SKIP_RETURN_CODE 77: a TEST_CASE that calls SKIP(...) (no device, ...) and nothing failed.
        catch_discover_tests(${test_name} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
    endfunction()

    # The exit code itself: four failed assertions exit 1, a skip alone 77.
    add_executable(test_harness_exit tests/unit/test_harness_exit.cpp)
    target_link_libraries(test_harness_exit PRIVATE ed_catch2_main Catch2::Catch2)
    add_test(NAME harness_four_failures_exit_1
             COMMAND sh -c "\"$<TARGET_FILE:test_harness_exit>\" \"[harness_four]\" > /dev/null; test $? -eq 1")
    add_test(NAME harness_skip_alone_exits_77
             COMMAND sh -c "\"$<TARGET_FILE:test_harness_exit>\" \"[harness_skip]\" > /dev/null; test $? -eq 77")

    ed_add_test(test_operator_apply             tests/unit/test_operator_apply.cpp)
    # One Lanczos algorithm shared by CPU / GPU via the abstract Backend
    # interface: pins agreement with a sequential-MGS reference and
    # orthogonality of the CGS2 basis.
    ed_add_test(test_lanczos_kernel             tests/unit/test_lanczos_kernel.cpp)
    ed_add_test(test_kernel_facades            tests/unit/test_kernel_facades.cpp)
    # The per-block solve driver: place() (the device decision for every block), the dense
    # lane, the LinearOperator surface, the Backend-templated block lanes on CPU and CUDA, and
    # the sampled-thermal dense crossover.
    ed_add_test(test_block_solve                tests/unit/test_block_solve.cpp)
    target_include_directories(test_block_solve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)   # internal.h
    # THE reduced-CSR budget decision (shared by the abelian + little-group
    # lanes): serial admission, and the aggregate split across concurrent
    # sector builders that keeps ED_SYM_SECTOR_PARALLEL=1 from OOMing.
    ed_add_test(test_sector_csr_budget          tests/unit/test_sector_csr_budget.cpp)
    # Numerical accuracy: FTLM and mTPQ through ed::sectors::thermal against a dense
    # reference, without symmetry and over the Sz sectors, on the CPU and (WITH_CUDA) the
    # device. test_block_solve [thermal] pins the dense crossover these runs switch off.
    ed_add_test(test_thermal_dense_ref          tests/unit/test_thermal_dense_ref.cpp)
    # The Backend-templated FTLM body (both lanes) honours full
    # reorthogonalisation and fills ground_state_estimate.
    ed_add_test(test_ftlm_backend_body          tests/unit/test_ftlm_backend_body.cpp)
    # The FTLM body draws the shared per-sample seed recipe
    # (sample_seed.h), is run-to-run bit-identical, and is what the CPU
    # front door returns.
    ed_add_test(test_ftlm_sample_seed           tests/unit/test_ftlm_sample_seed.cpp)
    # OpenMP thread-pinning hook (env-knob default + parsing,
    # idempotent pinning).
    ed_add_test(test_numa                       tests/unit/test_numa.cpp)
    # The environment registry: one meaning of "set" per value type, unique rows,
    # the unknown-name scan.
    ed_add_test(test_env_registry               tests/unit/test_env_registry.cpp)
    # The available-memory probe honours the job's cgroup limit (Krylov basis caps).
    ed_add_test(test_mem_guard                  tests/unit/test_mem_guard.cpp)

    # Lattice generators.
    ed_add_test(test_input_library              tests/unit/test_input_library.cpp)
    # Symmetry engine: compiled byte-LUT group action bit-identity
    ed_add_test(test_compiled_group             tests/unit/test_compiled_group.cpp)
    # Fused OrbitTable (hash, Burnside sum rule)
    ed_add_test(test_orbit_table                tests/unit/test_orbit_table.cpp)
    # The in-process OrbitTable registry
    ed_add_test(test_symmetry_cache             tests/unit/test_symmetry_cache.cpp)
    # SU(2): S^2 Casimir in the TermStorage schema + labeling
    ed_add_test(test_casimir_operator           tests/unit/test_casimir_operator.cpp)
    # SU(2): S-resolved dims (highest-weight Burnside differencing)
    ed_add_test(test_su2_dims                   tests/unit/test_su2_dims.cpp)
    # SU(2): Lowdin total-spin projector + targeting wrapper
    ed_add_test(test_casimir_projector          tests/unit/test_casimir_projector.cpp)
    # Numerical irrep decomposition (non-abelian symmetry engine)
    ed_add_test(test_irreps                     tests/unit/test_irreps.cpp)

    # Lock down the programmatic symmetry DSL (ed::sym::Permutation
    # algebra + generate_group closure).
    ed_add_test(test_symmetry_dsl               tests/unit/test_symmetry_dsl.cpp)

    # The operator algebra (masked terms: products, adjoints, images against dense matrices) and
    # the compiled programs swept in symmetry sectors (against the full-space expansion).
    ed_add_test(test_masked_terms               tests/unit/test_masked_terms.cpp)
    ed_add_test(test_rep_matrix_elements        tests/unit/test_rep_matrix_elements.cpp)
    target_include_directories(test_rep_matrix_elements PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)   # engine/options.h
    # The symmetry verdicts on the canonical terms (invariance.h): flip, conjugation, Sz content,
    # SU(2), permutations, against the dense truth.
    ed_add_test(test_invariance                 tests/unit/test_invariance.cpp)
    # Every lane that applies H (full-space gather, scatter, CSR; rep-sector CSR, walks, device
    # gather) against the dense block of its canonical terms.
    ed_add_test(test_row_walk                   tests/unit/test_row_walk.cpp)
    target_include_directories(test_row_walk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)   # engine/internal.h

    # -------------------------------------------------------------------------
    # CUDA backend tests. Only added when WITH_CUDA is ON so non-GPU
    # builds skip compilation entirely. The TEST_CASEs themselves SKIP at
    # runtime when no CUDA device is detected (cudaGetDeviceCount), which
    # keeps the build-only `linux-cuda-build` CI lane green even though it
    # has no NVIDIA hardware attached.
    # -------------------------------------------------------------------------
    if(WITH_CUDA)
        # CudaBackend BLAS-1 round-trips + agreement of
        # `lanczos_kernel<Backend&>` between CpuBackend and CudaBackend on a
        # 6-site Heisenberg chain.
        ed_add_test(test_cuda_backend           tests/unit/test_cuda_backend.cpp)
    endif()

    # Backend BLAS-3 surface (gemm / gemv / trsm / qr_thin) lockdown across CpuBackend
    # (+ CudaBackend when WITH_CUDA).
    ed_add_test(test_backend_blas3              tests/unit/test_backend_blas3.cpp)
endif()
