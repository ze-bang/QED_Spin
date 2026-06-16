// =============================================================================
// src/symmetry/sector_operator_gpu.cpp
//
// Non-CUDA throwing stub for ``ed::symmetry::SectorOperator::bind_cuda()``.
// Compiled into ed_core. On a WITH_CUDA build this TU is empty (the real
// definition is in ``sector_operator_gpu.cu`` -> ed_solvers_gpu).
//
// Callers reach ``bind_cuda()`` only through ``ed::select_backend``, which
// never picks the GPU lane unless ``Geometry::supports_device_matvec`` is
// set -- and ``SectorOperator::geometry()`` only sets that flag under
// WITH_CUDA. So on a non-CUDA build this throw is effectively unreachable;
// it exists to satisfy the out-of-line override declaration.
//
// Phase A of the operator-collapse GPU-parity work (Jun 2026).
// =============================================================================

// Operator-collapse Phase 4: SectorOperator is now the alias
// SubspaceOperator<SymmetryBasisPolicy, Host>, whose bind_cuda() is an INLINE
// template member. Under !WITH_CUDA it throws inline (no out-of-line key
// function needed); under WITH_CUDA the strong bind_cuda_impl_ specialization
// lives in sector_operator_gpu.cu (ed_solvers_gpu). This TU therefore no
// longer defines any symbol -- it is intentionally empty.
