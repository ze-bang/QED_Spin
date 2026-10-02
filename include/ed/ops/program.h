// =============================================================================
// include/ed/ops/program.h -- a list of observables compiled into a flat program, and its
// sweep over a symmetry-reduced (representative) basis: <bra|O|ket> for many observables and
// many state pairs at once.
//
// compile_program() projects every observable onto the component that can connect the two
// sectors (the "lambda projection"):
//
//     O_lambda = (1/|G|) sum_g conj(lambda(g)) U_g O U_g^dagger,
//     lambda(g) = chi_bra(g) * conj(chi_ket(g)),
//
// so <bra|O|ket> == <bra|O_lambda|ket> for ANY O: a single bond, plaquette or string may be
// passed, no invariance is required. U_g acts as the engine's group element (site permutation
// in the apply_perm convention, then XOR flip_masks[g]).
//
// Basis convention (RepSectorData, expand_rep_vector_to_computational): the rep vector u of a
// sector with characters chi is the state
//
//     |u> = |G|^{-1/2} sum_r u_r inv_norm[r] sum_g conj(chi(g)) U_g |s_r>,
//
// an isometry from the rep basis. For a lambda-projected operator the group sum on the ket side
// collapses and
//
//     <a|O|b> = sum_r b_r inv_norm_src[r] sum_t <t|O|s_r> proj_tgt(t) conj(a_{j(t)}),
//
// where j(t) and proj_tgt(t) are the target sector's index_and_projection(t). The sweep runs over
// KET representatives only and applies each term to one state, so O need not be Hermitian and the
// same code serves diagonal, cross-irrep and cross-S^z elements. Reduction: OpenMP static schedule
// with per-thread accumulators summed in thread order (bitwise reproducible for a fixed thread
// count).
// =============================================================================
#pragma once

#include <ed/ops/algebra.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ed::symmetry { struct RepSectorData; }

namespace ed::ops {

/// Pointers into a program's arrays: what a row walk (row_walk.h) reads, on the host or, with
/// the arrays copied to a device, in a kernel. C is the complex type of the reading side
/// (std::complex<double> on the host); every array is laid out as in MaskedProgram.
template <class C>
struct ProgramView {
    std::uint32_t        n_groups      = 0;
    const std::uint64_t* group_flip    = nullptr;
    const int*           group_setbits = nullptr;   ///< -1: the group's subgroups differ in popcount
    const std::uint32_t* group_vbegin  = nullptr;
    const std::uint64_t* vsub_val      = nullptr;
    const std::uint32_t* vsub_tbegin   = nullptr;
    const std::uint64_t* term_sign     = nullptr;
    const C*             term_coeff    = nullptr;
};

/// Flat program over which the sector sweep runs. Terms are grouped by flip mask (one
/// target state and ONE rep lookup per group per ket rep, shared by every observable that
/// flips those bits), then by the required values on the flipped bits (for a given ket
/// state at most one value subgroup matches: v == s & flip).
struct MaskedProgram {
    int n_obs = 0;
    int delta_set_bits = 0;                        ///< n_set(bra) - n_set(ket)
    std::vector<std::uint64_t> group_flip;         ///< flip mask per group
    std::vector<int>           group_setbits;      ///< popcount(v) of every subgroup, or -1 if they differ
    std::vector<std::uint32_t> group_vbegin;       ///< n_groups + 1 offsets into vsub_*
    std::vector<std::uint64_t> vsub_val;           ///< required values, sorted within a group
    std::vector<std::uint32_t> vsub_tbegin;        ///< n_vsub + 1 offsets into term_*
    std::vector<std::uint64_t> term_sign;          ///< sign_mask per term
    std::vector<std::complex<double>> term_coeff;  ///< coefficient per term
    std::vector<std::uint32_t> term_obs;           ///< observable index per term
    std::vector<std::size_t>   terms_per_obs;      ///< after projection (0: selection-rule zero)
    /// Characters the program was compiled for (checked by rep_matrix_elements).
    std::vector<std::complex<double>> src_characters, tgt_characters;
    int src_n_up = -1, tgt_n_up = -1;

    [[nodiscard]] std::size_t n_groups() const noexcept { return group_flip.size(); }
    [[nodiscard]] std::size_t n_terms() const noexcept { return term_sign.size(); }
    [[nodiscard]] ProgramView<std::complex<double>> view() const noexcept {
        return {static_cast<std::uint32_t>(group_flip.size()), group_flip.data(), group_setbits.data(),
                group_vbegin.data(), vsub_val.data(), vsub_tbegin.data(), term_sign.data(), term_coeff.data()};
    }
};

struct CompileOptions {
    /// Apply the lambda projection (default). Only turn off for operators the caller
    /// guarantees are already covariant (U_g O U_g^dagger = lambda(g) O).
    bool project = true;
    /// Projected coefficients below drop * max|coeff| of the UNPROJECTED observable are
    /// discarded after merging (selection-rule zeros compile to no terms).
    double drop = 1e-14;
};

/// Compile `ops` for matrix elements <bra|O|ket> with ket in sector `src` and bra in
/// sector `tgt`. Both sectors must come from the same group (identical group_size,
/// n_sites, perms_flat and flip_masks). Terms whose change of the set-bit count does not
/// match tgt.n_up - src.n_up cannot connect the sectors and are dropped.
[[nodiscard]] MaskedProgram compile_program(const std::vector<MaskedOperator>& ops,
                                            const ed::symmetry::RepSectorData& src,
                                            const ed::symmetry::RepSectorData& tgt,
                                            const CompileOptions& opt = {});



/// One operator's canonical terms in the flip-grouped layout, for the row walks (row_walk.h):
/// no sector and no projection (n_obs = 1, n_up = -1). Groups are in ascending flip mask, so
/// the diagonal (F = 0) comes first; within a subgroup the terms keep their key order.
[[nodiscard]] MaskedProgram compile_operator(const MaskedOperator& O);

/// The operator a program was compiled from (one observable): every term, coefficients exact.
[[nodiscard]] MaskedOperator program_operator(const MaskedProgram& P, int n_sites);

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

}  // namespace ed::ops
