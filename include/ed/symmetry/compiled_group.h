#pragma once
// =============================================================================
// include/ed/symmetry/compiled_group.h
//
// CompiledGroup: the compiled action of a (unitary) symmetry group on
// computational basis states -- Stage 1 of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md).
//
// Each group element acts as
//
//     s'  =  permute_bits(s, perm)  XOR  flip_mask
//
// which covers every unitary symmetry this package projects on:
// translations, arbitrary point-group elements, their products, and
// (Stage 5) internal Z2 flips such as the global spin flip
// (flip_mask = (1<<N)-1). U(1) Sz is a Subspace, not a group element;
// time reversal is antiunitary and is handled as sector-pairing
// metadata, never through this class.
//
// The permutation is compiled to a byte-decomposition lookup table:
// input byte b with value v contributes ``lut[(g*BPW + b)*256 + v]``
// to the output word, so ``apply`` is BPW (= ceil(N/8), 4 at N=32)
// L1/L2-resident loads + ORs + one XOR instead of an N-iteration
// scalar bit scatter. This generalizes the uint32 N<=32 LUT that
// ``RepSectorData::build_perm_lut`` introduced for the SpMV rep walk
// (same layout convention, widened to uint64 / N<=64) so construction
// and matvec share one compiled form.
//
// ``content_hash()`` is a canonical FNV-1a over (n_sites, elements):
// the key for the Stage-3 persistent basis cache. It deliberately does
// NOT include Hamiltonian couplings -- the symmetry-adapted basis
// depends only on the group and the subspace.
//
// Bit-identity contract: ``apply(s, g) == applyPermutation(s, perm[g])
// ^ flip[g]`` for every state and element; pinned by
// tests/unit/test_compiled_group.cpp.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <ed/core/basis_utils.h>  // applyPermutation (reference semantics)

namespace ed::symmetry {

class CompiledGroup {
public:
    CompiledGroup() = default;

    /// Compile pure site permutations (no flips). ``perms[g]`` maps output
    /// bit i to input bit perms[g][i], matching ``applyPermutation``.
    [[nodiscard]] static CompiledGroup
    from_permutations(const std::vector<std::vector<int>>& perms, int n_sites)
    {
        return build_(perms, std::vector<std::uint64_t>(perms.size(), 0ULL),
                      n_sites);
    }

    /// Compile permutation + XOR-flip elements (Stage 5: spin-flip Z2 and
    /// sublattice flips ride the same table).
    [[nodiscard]] static CompiledGroup
    from_elements(const std::vector<std::vector<int>>& perms,
                  const std::vector<std::uint64_t>&    flip_masks,
                  int                                  n_sites)
    {
        if (flip_masks.size() != perms.size()) {
            throw std::invalid_argument(
                "CompiledGroup: perms/flip_masks size mismatch");
        }
        return build_(perms, flip_masks, n_sites);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] int n_sites() const noexcept { return n_sites_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Action of element ``g`` on state ``s``. BPW table loads + one XOR.
    [[nodiscard]] inline std::uint64_t
    apply(std::uint64_t s, std::size_t g) const noexcept {
        const std::uint64_t* lut_g = lut_.data() + g * stride_;
        std::uint64_t out = 0;
        for (int b = 0; b < bpw_; ++b) {
            out |= lut_g[(static_cast<std::size_t>(b) << 8)
                         + ((s >> (8 * b)) & 0xFFu)];
        }
        return out ^ flips_[g];
    }

    /// True iff element ``g`` fixes every state (identity permutation and
    /// zero flip). Useful for skip-identity micro-optimizations; callers
    /// that require bit-identity with legacy loops should not skip.
    [[nodiscard]] bool is_identity(std::size_t g) const noexcept {
        return identity_[g] != 0;
    }

    /// Canonical FNV-1a content hash over (n_sites, per-element perm+flip).
    /// Stage-3 cache key material.
    [[nodiscard]] std::uint64_t content_hash() const noexcept { return hash_; }

private:
    [[nodiscard]] static CompiledGroup
    build_(const std::vector<std::vector<int>>& perms,
           const std::vector<std::uint64_t>&    flip_masks,
           int                                  n_sites)
    {
        if (n_sites <= 0 || n_sites > 64) {
            throw std::invalid_argument(
                "CompiledGroup: n_sites must be in [1, 64]");
        }
        CompiledGroup cg;
        cg.n_sites_ = n_sites;
        cg.size_    = perms.size();
        cg.bpw_     = (n_sites + 7) / 8;
        cg.stride_  = static_cast<std::size_t>(cg.bpw_) * 256;
        cg.lut_.assign(cg.size_ * cg.stride_, 0ULL);
        cg.flips_   = flip_masks;
        cg.identity_.assign(cg.size_, 0);

        std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
        auto mix = [&h](std::uint64_t v) {
            for (int i = 0; i < 8; ++i) {
                h ^= (v >> (8 * i)) & 0xFFu;
                h *= 1099511628211ULL;
            }
        };
        mix(static_cast<std::uint64_t>(n_sites));
        mix(static_cast<std::uint64_t>(cg.size_));

        for (std::size_t g = 0; g < cg.size_; ++g) {
            const std::vector<int>& p = perms[g];
            if (static_cast<int>(p.size()) != n_sites) {
                throw std::invalid_argument(
                    "CompiledGroup: permutation size != n_sites");
            }
            // Invert: output bit i reads input bit p[i]  =>  input bit j
            // scatters to output bit p_inv[j].
            int p_inv[64];
            bool ident = (flip_masks[g] == 0ULL);
            for (int i = 0; i < n_sites; ++i) {
                if (p[i] < 0 || p[i] >= n_sites) {
                    throw std::invalid_argument(
                        "CompiledGroup: permutation entry out of range");
                }
                p_inv[p[i]] = i;
                if (p[i] != i) ident = false;
                mix(static_cast<std::uint64_t>(p[i]));
            }
            mix(flip_masks[g]);
            cg.identity_[g] = ident ? 1 : 0;

            std::uint64_t* lut_g = cg.lut_.data() + g * cg.stride_;
            for (int byte_idx = 0; byte_idx < cg.bpw_; ++byte_idx) {
                const int bit_base = byte_idx * 8;
                for (int byte_val = 0; byte_val < 256; ++byte_val) {
                    std::uint64_t out = 0;
                    for (int b = 0; b < 8 && bit_base + b < n_sites; ++b) {
                        if ((byte_val >> b) & 1) {
                            out |= (1ULL << p_inv[bit_base + b]);
                        }
                    }
                    lut_g[(static_cast<std::size_t>(byte_idx) << 8)
                          + static_cast<std::size_t>(byte_val)] = out;
                }
            }
        }
        cg.hash_ = h;
        return cg;
    }

    std::vector<std::uint64_t> lut_;       // size_ * stride_ entries
    std::vector<std::uint64_t> flips_;     // per element XOR mask
    std::vector<std::uint8_t>  identity_;  // per element identity flag
    std::size_t                size_    = 0;
    std::size_t                stride_  = 0;  // bpw_ * 256
    int                        bpw_     = 0;
    int                        n_sites_ = 0;
    std::uint64_t              hash_    = 0;
};

}  // namespace ed::symmetry
