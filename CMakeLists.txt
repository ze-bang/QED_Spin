cmake_minimum_required(VERSION 3.18)

# Detect host CPU vendor to pick optimized BLAS/LAPACK defaults
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

set(DEFAULT_WITH_CUDA ON)
set(DEFAULT_WITH_MPI ON)
set(DEFAULT_WITH_MKL ON)
set(DEFAULT_USE_ONEMKL ON)
set(DEFAULT_USE_AOCL_BLIS OFF)

if(CPU_VENDOR STREQUAL "AMD")
    set(DEFAULT_WITH_MKL OFF)
    set(DEFAULT_USE_ONEMKL OFF)
    set(DEFAULT_USE_AOCL_BLIS ON)
elseif(CPU_VENDOR STREQUAL "Intel")
    # keep MKL defaults
else()
    # Unknown vendor: keep generic defaults (MKL optional, AOCL disabled)
endif()

# Options (define before project)
option(WITH_CUDA "Build with CUDA support" ${DEFAULT_WITH_CUDA})
option(WITH_MPI "Build with MPI support" ${DEFAULT_WITH_MPI})
option(WITH_MKL "Build with Intel MKL support" ${DEFAULT_WITH_MKL})
option(USE_ONEMKL "Use Intel oneMKL instead of traditional MKL" ${DEFAULT_USE_ONEMKL})
option(USE_AOCL_BLIS "Use AMD AOCL-BLIS optimized BLAS/LAPACK" ${DEFAULT_USE_AOCL_BLIS})

if(USE_AOCL_BLIS AND WITH_MKL)
    message(STATUS "AOCL-BLIS requested; disabling MKL to avoid conflicting BLAS backends")
    set(WITH_MKL OFF)
endif()

if(USE_AOCL_BLIS)
    set(USE_ONEMKL OFF)
endif()

if(NOT WITH_MKL)
    set(USE_ONEMKL OFF)
endif()

# Set languages based on options
if(WITH_CUDA)
    project(ExactDiagonalization LANGUAGES CXX CUDA)
    enable_language(CUDA)
else()
    project(ExactDiagonalization LANGUAGES CXX)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard if CUDA is enabled
if(WITH_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    
    # Add CUDA compiler flags to handle C++ standard library compatibility
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    
    # Separate host and device compilation to avoid C++17 issues
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Source directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CORE_DIR ${SRC_DIR}/core)
set(CPU_SOLVERS_DIR ${SRC_DIR}/cpu_solvers)
set(GPU_DIR ${SRC_DIR}/gpu)

include_directories(${SRC_DIR})
include_directories(${CORE_DIR})
include_directories(${CPU_SOLVERS_DIR})
include_directories(${GPU_DIR})

# Configure BLAS/LAPACK backends
set(BLAS_LIBRARIES "")
set(LAPACK_LIBRARIES "")
set(LAPACKE_LIBRARIES "")
set(EXTRA_LINALG_LIBRARIES "")

if(USE_AOCL_BLIS)
    message(STATUS "Configuring AMD AOCL-BLIS libraries")

    set(_AOCL_HINT_DIRS)
    if(DEFINED ENV{AOCL_ROOT})
        list(APPEND _AOCL_HINT_DIRS $ENV{AOCL_ROOT})
    endif()
    list(APPEND _AOCL_HINT_DIRS
        /opt/AMD/aocl
        /opt/AMD/aocl/aocl-linux-x86_64
        /opt/AMD/aocl/aocl-linux-aocc-5.0.0
        /opt/AMD/aocl/aocl-linux-aocc-5.0.0/aocc
        /opt/aocl
        /usr/lib/aocl
    )

    find_path(AOCL_INCLUDE_DIR cblas.h
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES include include/aocl include/amd-blis include/blis include/BLIS
    )

    find_library(AOCL_BLIS_LIBRARY NAMES blis-mt blis
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES lib lib64 lib/aocl/lib
    )

    find_library(AOCL_LAPACK_LIBRARY NAMES aoclflapack flame lapack
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES lib lib64 lib/aocl/lib
    )

    find_library(AOCL_LAPACKE_LIBRARY NAMES lapacke
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES lib lib64 lib/aocl/lib
    )

    if(AOCL_INCLUDE_DIR AND AOCL_BLIS_LIBRARY AND AOCL_LAPACK_LIBRARY AND AOCL_LAPACKE_LIBRARY)
        include_directories(SYSTEM ${AOCL_INCLUDE_DIR})
        set(BLAS_LIBRARIES ${AOCL_BLIS_LIBRARY})
        set(LAPACK_LIBRARIES ${AOCL_LAPACK_LIBRARY})
        set(LAPACKE_LIBRARIES ${AOCL_LAPACKE_LIBRARY})
        add_definitions(-DUSE_AOCL_BLIS)
        find_library(AOCL_OMP_LIBRARY NAMES omp
            HINTS ${_AOCL_HINT_DIRS}
            PATH_SUFFIXES "" lib lib64 lib/aocl/lib aocc/lib lib/llvm-18/lib
        )
        if(AOCL_OMP_LIBRARY)
            list(APPEND EXTRA_LINALG_LIBRARIES ${AOCL_OMP_LIBRARY})
            message(STATUS "  AOCL OpenMP runtime: ${AOCL_OMP_LIBRARY}")
        else()
            message(WARNING "AOCL-BLIS OpenMP runtime (libomp) not found; linking may fail. Set AOCL_ROOT if needed.")
        endif()
        message(STATUS "  AOCL include dir: ${AOCL_INCLUDE_DIR}")
        message(STATUS "  AOCL BLIS library: ${AOCL_BLIS_LIBRARY}")
        message(STATUS "  AOCL LAPACK library: ${AOCL_LAPACK_LIBRARY}")
        message(STATUS "  AOCL LAPACKE library: ${AOCL_LAPACKE_LIBRARY}")
    else()
        message(FATAL_ERROR "AOCL-BLIS requested but required components were not found. Set AOCL_ROOT to your AOCL installation path or disable USE_AOCL_BLIS.")
    endif()
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    if(WITH_MKL)
        if(USE_ONEMKL)
            # Use oneMKL from oneAPI
            find_package(MKL CONFIG QUIET)
            if(MKL_FOUND)
                message(STATUS "Using Intel oneMKL from: ${MKLROOT}")
            else()
                message(STATUS "Intel oneMKL not found, falling back to traditional MKL detection")
                set(USE_ONEMKL OFF)
            endif()
        endif()

        if(NOT USE_ONEMKL)
            # Use traditional MKL - add include directories
            find_path(MKL_INCLUDE_DIR mkl.h
                HINTS
                    /usr/include/mkl
                    /opt/intel/mkl/include
                    /opt/intel/oneapi/mkl/latest/include
                    /opt/intel/compilers_and_libraries/linux/mkl/include
                PATH_SUFFIXES mkl
            )

            if(MKL_INCLUDE_DIR)
                include_directories(SYSTEM ${MKL_INCLUDE_DIR})
                message(STATUS "Found MKL headers at: ${MKL_INCLUDE_DIR}")
            else()
                message(WARNING "MKL headers not found, disabling MKL support")
                set(WITH_MKL OFF)
                set(USE_ONEMKL OFF)
            endif()
        endif()

        if(WITH_MKL)
            add_definitions(-DWITH_MKL)
        endif()
    endif()

    message(STATUS "BLAS library: ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK library: ${LAPACK_LIBRARIES}")

    if(BLAS_LIBRARIES MATCHES "mkl")
        message(STATUS "BLAS/LAPACK vendor: Intel MKL")
    elseif(BLAS_LIBRARIES MATCHES "openblas")
        message(STATUS "BLAS vendor: OpenBLAS")
    elseif(BLAS_LIBRARIES MATCHES "atlas")
        message(STATUS "BLAS vendor: ATLAS")
    elseif(BLAS_LIBRARIES MATCHES "Accelerate")
        message(STATUS "BLAS/LAPACK vendor: Apple Accelerate Framework")
    elseif(BLAS_LIBRARIES MATCHES "FlexiBLAS")
        message(STATUS "BLAS vendor: FlexiBLAS")
    else()
        message(STATUS "BLAS/LAPACK vendor: Unknown/Generic")
    endif()

    find_package(LAPACKE QUIET)
    if(NOT LAPACKE_FOUND)
        # Try to find it manually
        find_library(LAPACKE_LIBRARY NAMES lapacke)
        if(LAPACKE_LIBRARY)
            set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
        else()
            message(FATAL_ERROR "LAPACKE library not found")
        endif()
    endif()
endif()

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Find HDF5 (with C++ bindings)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
message(STATUS "HDF5 found: ${HDF5_VERSION}")
message(STATUS "HDF5 include directories: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")
add_definitions(${HDF5_DEFINITIONS})

# CUDA setup
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DWITH_CUDA)
    add_definitions(-DTPQ_HAVE_CUDA)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit include directories: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# Find ARPACK
find_library(ARPACK_LIBRARY NAMES arpack arpack-ng)
if(NOT ARPACK_LIBRARY)
    message(FATAL_ERROR "ARPACK library not found")
endif()

# MPI setup
if(WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    add_definitions(-DWITH_MPI)
endif()

# New elegant main sources
set(SOURCES
    ${CORE_DIR}/ed_main.cpp
    ${CORE_DIR}/ed_config.cpp
    ${CPU_SOLVERS_DIR}/arpack.cpp
    ${CPU_SOLVERS_DIR}/observables.cpp
    ${CPU_SOLVERS_DIR}/lanczos.cpp
    ${CPU_SOLVERS_DIR}/CG.cpp
    ${CPU_SOLVERS_DIR}/dynamics.cpp
    ${CPU_SOLVERS_DIR}/TPQ.cpp
    ${CPU_SOLVERS_DIR}/ftlm.cpp
    ${CPU_SOLVERS_DIR}/ltlm.cpp
    ${CPU_SOLVERS_DIR}/hybrid_thermal.cpp
)

# Header-only sources that should be included in compilation
set(HEADER_SOURCES
    ${SRC_DIR}/*.h
)

# GPU sources for CUDA-enabled builds
if(WITH_CUDA)
    set(GPU_SOURCES
        ${GPU_DIR}/gpu_operator.cu
        ${GPU_DIR}/gpu_kernels.cu
        ${GPU_DIR}/gpu_fixed_sz_operator.cu
        ${GPU_DIR}/gpu_lanczos.cu
        ${GPU_DIR}/gpu_ed_wrapper.cu
        ${GPU_DIR}/gpu_tpq.cu
        ${GPU_DIR}/gpu_cg.cu
        ${GPU_DIR}/gpu_dynamics.cu
    )
    
    set(GPU_HEADERS
        ${GPU_DIR}/kernel_config.h
        ${GPU_DIR}/bit_operations.cuh
        ${GPU_DIR}/gpu_operator.cuh
        ${GPU_DIR}/gpu_lanczos.cuh
        ${GPU_DIR}/gpu_ed_wrapper.h
        ${GPU_DIR}/gpu_tpq.cuh
        ${GPU_DIR}/gpu_cg.cuh
        ${GPU_DIR}/gpu_dynamics.cuh
    )
endif()

# Define executables

# New elegant main (add GPU sources if CUDA is enabled)
if(WITH_CUDA)
    add_executable(ED ${SOURCES} ${GPU_SOURCES})
    target_include_directories(ED PRIVATE ${GPU_DIR})
else()
    add_executable(ED ${SOURCES})
endif()

# Add TPQ_DSSF executable (only if MPI is enabled since it uses MPI)
add_executable(TPQ_DSSF ${CORE_DIR}/TPQ_DSSF.cpp ${CPU_SOLVERS_DIR}/dynamics.cpp ${CPU_SOLVERS_DIR}/TPQ.cpp ${CPU_SOLVERS_DIR}/ftlm.cpp)


# Link libraries for new main
target_link_libraries(ED
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${EXTRA_LINALG_LIBRARIES}
    ${ARPACK_LIBRARY}
    ${HDF5_LIBRARIES}
)

# Add CUDA libraries to ED if CUDA is enabled
if(WITH_CUDA)
    target_link_libraries(ED
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
        CUDA::cusolver
    )
    
endif()

# Link libraries for TPQ_DSSF executable (only if it exists)
if(TARGET TPQ_DSSF)
    target_link_libraries(TPQ_DSSF
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
        ${HDF5_LIBRARIES}
    )
endif()


# Link libraries for GPU Lanczos example
if(TARGET GPU_LANCZOS_EXAMPLE)
    target_link_libraries(GPU_LANCZOS_EXAMPLE
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
    )
endif()


# Link libraries for CUDA example if it exists
if(TARGET CUDA_EXAMPLE)
    target_link_libraries(CUDA_EXAMPLE
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
endif()

# Link libraries for CUDA test if it exists
if(TARGET CUDA_TEST)
    target_link_libraries(CUDA_TEST
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
endif()


if(WITH_MKL)
    if(USE_ONEMKL AND MKL_FOUND)
        # Link oneMKL using its modern CMake targets
        target_link_libraries(ED MKL::MKL)
        
        if(TARGET TPQ_DSSF)
            target_link_libraries(TPQ_DSSF MKL::MKL)
        endif()
        
        # Link MKL to CUDA example if it exists
        if(TARGET CUDA_EXAMPLE)
            target_link_libraries(CUDA_EXAMPLE MKL::MKL)
        endif()
        
        # Link MKL to CUDA test if it exists
        if(TARGET CUDA_TEST)
            target_link_libraries(CUDA_TEST MKL::MKL)
        endif()
        
    else()
        # Link traditional MKL (already linked via BLAS_LIBRARIES)
        # No additional linking needed since MKL is detected as BLAS/LAPACK
    endif()
endif()

# Add CUDA libraries if enabled
if(WITH_CUDA)
    target_link_libraries(ED
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
        CUDA::cusolver
    )
    
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
            CUDA::cusolver
        )
    endif()
    
endif()

# Add MPI if enabled
if(WITH_MPI)
    target_link_libraries(ED ${MPI_CXX_LIBRARIES})
    
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF ${MPI_CXX_LIBRARIES})
    endif()
endif()

# Install targets
install(TARGETS ED DESTINATION bin)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
if(WITH_CUDA)
    message(STATUS "  CUDA Support: YES")
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    if(CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    message(STATUS "  CUDA Support: NO")
endif()
message(STATUS "  MPI Support: ${WITH_MPI}")

# Build configuration summary and performance options
message(STATUS "")
message(STATUS "=== Performance Configuration Summary ===")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Aggressive Optimizations: ${ENABLE_AGGRESSIVE_OPTS}")
message(STATUS "  Profile-Guided Optimization: ${ENABLE_PROFILE_GUIDED_OPTS}")
message(STATUS "  Native Architecture: ${ENABLE_NATIVE_ARCH}")
message(STATUS "  CUDA Support: ${WITH_CUDA}")
message(STATUS "  MPI Support: ${WITH_MPI}")
message(STATUS "  OpenMP Support: ${OpenMP_CXX_FOUND}")

# CPU information
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    execute_process(COMMAND grep -c "^processor" /proc/cpuinfo OUTPUT_VARIABLE CPU_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND grep "model name" /proc/cpuinfo COMMAND head -1 COMMAND sed "s/.*: //" OUTPUT_VARIABLE CPU_MODEL OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "  CPU: ${CPU_MODEL} (${CPU_COUNT} cores)")
    
    # Check for specific CPU features
    execute_process(COMMAND grep -q "avx2" /proc/cpuinfo RESULT_VARIABLE AVX2_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(AVX2_RESULT EQUAL 0)
        message(STATUS "  AVX2: Available")
    endif()
    
    execute_process(COMMAND grep -q "fma" /proc/cpuinfo RESULT_VARIABLE FMA_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(FMA_RESULT EQUAL 0)
        message(STATUS "  FMA: Available")
    endif()
endif()

# CPU optimization flags
set(CPU_OPT_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND CPU_OPT_FLAGS
        -O3                          # Maximum optimization
        -march=native                # Optimize for the build machine's CPU
        -mtune=native                # Tune for the build machine's CPU
        -ffast-math                  # Aggressive floating-point optimizations
        -funroll-loops               # Unroll loops for better performance
        -ftree-vectorize             # Enable auto-vectorization
        -fomit-frame-pointer         # Remove frame pointer for extra register
    )
    
    # Additional flags for GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND CPU_OPT_FLAGS
            -fprefetch-loop-arrays   # Enable prefetching
            -fipa-pta                # Inter-procedural pointer analysis
        )
    endif()
    
    # Additional flags for Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND CPU_OPT_FLAGS
            -fvectorize              # Explicit vectorization flag
        )
    endif()
    
    message(STATUS "  CPU Optimization Flags: ${CPU_OPT_FLAGS}")
endif()

# Apply CPU optimization flags to all targets
target_compile_options(ED PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)

if(TARGET TPQ_DSSF)
    target_compile_options(TPQ_DSSF PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    )
endif()

# CUDA-specific compile options
if(WITH_CUDA)

    set_property(TARGET ED PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(ED PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    
    if(TARGET TPQ_DSSF)
        set_property(TARGET TPQ_DSSF PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(TPQ_DSSF PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
    
endif()

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ED OpenMP::OpenMP_CXX)
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF OpenMP::OpenMP_CXX)
    endif()
    message(STATUS "  OpenMP Support: YES")
else()
    message(STATUS "  OpenMP Support: NO")
endif()

# Debug test for projected matrix (commented out - file missing)
# add_executable(debug_projected_matrix test/debug_projected_matrix.cpp)
# target_link_libraries(debug_projected_matrix 
#     ${BLAS_LIBRARIES}
#     ${LAPACK_LIBRARIES}
#     ${LAPACKE_LIBRARIES}
#     ${EXTRA_LINALG_LIBRARIES}
#     ${ARPACK_LIBRARY}
# )
# target_link_libraries(debug_projected_matrix OpenMP::OpenMP_CXX)

# Streaming symmetry test (only build if file exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_symmetry.cpp")
    set(TEST_STREAMING_SOURCES
        test_streaming_symmetry.cpp
        ${CORE_DIR}/ed_config.cpp
        ${CPU_SOLVERS_DIR}/arpack.cpp
        ${CPU_SOLVERS_DIR}/lanczos.cpp
        ${CPU_SOLVERS_DIR}/CG.cpp
        ${CPU_SOLVERS_DIR}/dynamics.cpp
        ${CPU_SOLVERS_DIR}/TPQ.cpp
        ${CPU_SOLVERS_DIR}/ftlm.cpp
        ${CPU_SOLVERS_DIR}/ltlm.cpp
        ${CPU_SOLVERS_DIR}/hybrid_thermal.cpp
        ${CPU_SOLVERS_DIR}/observables.cpp
    )

    if(WITH_CUDA)
        add_executable(test_streaming_symmetry ${TEST_STREAMING_SOURCES} ${GPU_SOURCES})
        target_include_directories(test_streaming_symmetry PRIVATE ${GPU_DIR})
    else()
        add_executable(test_streaming_symmetry ${TEST_STREAMING_SOURCES})
    endif()

    target_link_libraries(test_streaming_symmetry 
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
        ${HDF5_LIBRARIES}
    )

    if(WITH_CUDA)
        target_link_libraries(test_streaming_symmetry
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
            CUDA::cusolver
        )
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_streaming_symmetry OpenMP::OpenMP_CXX)
    endif()
endif()
