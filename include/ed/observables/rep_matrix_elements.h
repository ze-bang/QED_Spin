// =============================================================================
// include/ed/observables/rep_matrix_elements.h -- <bra|O|ket> for many observables and
// many state pairs in one sweep over a symmetry-reduced (representative) basis.
//
// Basis convention (RepSectorData, expand_rep_vector_to_computational): the rep vector
// u of a sector with characters chi is the state
//
//     |u> = |G|^{-1/2} sum_r u_r inv_norm[r] sum_g conj(chi(g)) U_g |s_r>,
//
// an isometry from the rep basis. For an operator already lambda-projected by
// compile_program (U_g O U_g^dagger = lambda(g) O, lambda = chi_bra conj(chi_ket)) the
// group sum on the ket side collapses and
//
//     <a|O|b> = sum_r b_r inv_norm_src[r] sum_t <t|O|s_r> proj_tgt(t) conj(a_{j(t)}),
//
// where j(t) and proj_tgt(t) are the target sector's index_and_projection(t). The sweep
// runs over KET representatives only and applies each term to one state, so O need not
// be Hermitian and the same code serves diagonal, cross-irrep and cross-S^z elements.
//
// Reduction: OpenMP static schedule with per-thread accumulators summed in thread
// order, so the result is bitwise reproducible for a fixed thread count.
// =============================================================================
#pragma once

#include <ed/observables/masked_program.h>

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

namespace ed::symmetry { struct RepSectorData; }

namespace ed::observables {

/// Read-only view of one vector in a sector's rep basis (length = sector dim()).
struct RepVectorView {
    const std::complex<double>* data = nullptr;
    std::size_t size = 0;
};

struct RepMEOptions {
    /// Run the sweep on the GPU (WITH_CUDA builds; throws otherwise). Sums are then not
    /// bitwise reproducible (shared-memory atomics); agreement with the CPU is ~1e-13.
    bool use_gpu = false;
    /// Shared-memory budget per block for the GPU accumulators; observables are swept in
    /// chunks when n_pairs * n_obs does not fit.
    std::size_t gpu_shared_bytes = 96 * 1024;
    /// Diagonal constraint projector P: keep only basis states in which EVERY mask holds
    /// exactly half of its bits set (hexagon masks: Q_h = 0 on every hexagon). The sweep then
    /// returns <bra| P O P |ket>; with the identity observable, <P>. P must commute with the
    /// group -- checked: every element must map the mask set onto itself (a spin flip keeps
    /// "half set" by construction). Empty: no constraint.
    std::vector<std::uint64_t> balanced_masks;
};

/// out[p * prog.n_obs + o] = <bras[pairs[p].first] | O_o | kets[pairs[p].second]>,
/// kets in sector `src`, bras in sector `tgt`, O_o the observables `prog` was compiled
/// from (compile_program(ops, src, tgt)). Throws if the program was compiled for other
/// sectors, a vector has the wrong length, or a pair index is out of range.
[[nodiscard]] std::vector<std::complex<double>>
rep_matrix_elements(const ed::symmetry::RepSectorData& src,
                    const ed::symmetry::RepSectorData& tgt,
                    const MaskedProgram& prog,
                    const std::vector<RepVectorView>& kets,
                    const std::vector<RepVectorView>& bras,
                    const std::vector<std::pair<int, int>>& pairs,
                    const RepMEOptions& opt = {});

/// GPU implementation (rep_matrix_elements_gpu.cu). Arguments are validated by
/// rep_matrix_elements(); call that instead.
[[nodiscard]] std::vector<std::complex<double>>
rep_matrix_elements_gpu(const ed::symmetry::RepSectorData& src,
                        const ed::symmetry::RepSectorData& tgt,
                        const MaskedProgram& prog,
                        const std::vector<RepVectorView>& kets,
                        const std::vector<RepVectorView>& bras,
                        const std::vector<std::pair<int, int>>& pairs,
                        const RepMEOptions& opt);

/// True when this is a WITH_CUDA build and a CUDA device is visible.
[[nodiscard]] bool rep_matrix_elements_gpu_available();

}  // namespace ed::observables
