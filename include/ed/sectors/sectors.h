#pragma once
// =============================================================================
// include/ed/sectors/sectors.h
//
// The symmetry sectors of a Hamiltonian and the lowest-level eigensolve over them.
//
// A Spec names the symmetry to use: a closed abelian group A of site permutations
// (the translations), coset representatives of the point group (residues), the Sz
// content, and the spin-flip / time-reversal toggles. subspaces() turns the Sz content
// into the diagonal subspaces to walk -- each Sz sector, each Sz-parity half, or the
// full space -- and folds n_up <-> N - n_up into one solve when H is spin-flip
// symmetric. Inside one subspace the little-group engine splits by momentum star and
// little-group irrep; for_each_star() streams those blocks one star at a time.
//
// eigs() is the lowest-k eigensolve over every block: each block's lowest levels,
// weighted by multiplicity (|star| x irrep dimension x time-reversal fold x flip
// mirror), merged across blocks and subspaces, and cut at k. A block that could not
// certify the levels it owes below the cut makes the window incomplete; eigs() throws
// unless the caller allows a partial window.
//
// Vectors stay in the basis the block was solved in (a group sector or a momentum
// sector, both RepSectorData). multiplet() expands one level's vector into the Sz
// sector or the full space and completes it to its whole degenerate multiplet by
// applying the symmetry operations, so a caller asking for the lowest k vectors gets
// k orthonormal eigenvectors even when a level is degenerate.
// =============================================================================

#include <ed/core/operator.h>
#include <ed/solvers/little_group_blocks.h>
#include <ed/symmetry/rep_sector_data.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace ed::sectors {

using Complex = std::complex<double>;
using Perm    = std::vector<int>;

struct Spec {
    std::vector<Perm> abelian;      ///< closed abelian group; empty = identity only
    std::vector<Perm> residues;     ///< point-group coset representatives
    int  n_up          = -1;        ///< one Sz sector (set-bit count); -1 = every sector
    int  sz_parity     = -1;        ///< one Sz-parity half when H breaks U(1); -1 = both
    bool use_sz        = true;      ///< decompose by Sz / parity when H conserves it
    int  spin_flip     = -1;        ///< -1 auto, 0 off, 1 require
    int  time_reversal = -1;        ///< -1 auto, 0 off, 1 require
    /// Restrict to total spin S = two_S / 2 (-1: no restriction). H must be SU(2)
    /// symmetric; the walk runs the Sz = S sector with every block projected onto the
    /// spin-S tower, and each level counts 2S + 1 times.
    int  two_S         = -1;
    std::vector<int> only_k0;       ///< restrict to these star representatives
    std::vector<int> only_irrep;    ///< restrict to these little-group irreps
};

/// What H conserves along the Sz axis.
enum class SzContent { U1, Parity, None };
[[nodiscard]] SzContent sz_content(const ::Operator& H);

struct Subspace {
    int n_up      = -1;
    int sz_parity = -1;
    int mirror    = 1;   ///< 2 when the flip image of this subspace is folded in
};

[[nodiscard]] std::vector<Subspace> subspaces(const ::Operator& H, int n_sites, const Spec& s);

using StarFn = std::function<void(const ed::solvers::LittleGroupStarInfo&,
                                  std::vector<ed::solvers::LittleGroupBlock>&)>;

/// Stream one subspace's blocks, one star resident at a time.
void for_each_star(const ::Operator& H, int n_sites, const Spec& s, const Subspace& sub,
                   const StarFn& fn);

// -----------------------------------------------------------------------------
// Lowest-level eigensolve
// -----------------------------------------------------------------------------

struct EigsOptions {
    int  k             = 1;
    bool vectors       = false;
    int  dense_max_dim = 64;    ///< per-block dense crossover
    int  block_size    = 1;     ///< >1: block Krylov-Schur inside each block
    bool allow_partial = false; ///< return an incomplete window instead of throwing
};

/// One eigenvalue of one block. The level occurs `multiplicity` times in the spectrum.
struct Level {
    double        energy       = 0.0;
    ed::solvers::LittleGroupBlockTag tag;   ///< the block's quantum numbers
    int           mirror       = 1;         ///< flip fold of the subspace
    std::uint64_t multiplicity = 1;         ///< tag.multiplicity x mirror
    int           vector       = -1;        ///< index into EigsResult::vectors, -1 = none
};

/// A certified eigenvector in the rep basis of the block's sector.
struct BlockVector {
    std::shared_ptr<const ed::symmetry::RepSectorData> basis;
    std::vector<Complex> amplitudes;        ///< normalized
};

struct EigsResult {
    std::vector<Level>       levels;        ///< ascending, cut so that multiplicities reach k
    std::vector<BlockVector> vectors;
    std::uint64_t            total_dim = 0; ///< sum of dim x multiplicity over the walk
    std::size_t              partial_blocks = 0;  ///< blocks that could not certify their rows
    bool                     complete = true;     ///< no uncertified level can lie below the cut
    bool                     flip_engaged = false;
    bool                     tr_engaged   = false;

    /// Energies with multiplicities expanded, the lowest k.
    [[nodiscard]] std::vector<double> energies(int k) const;
};

[[nodiscard]] EigsResult eigs(const ::Operator& H, int n_sites, const Spec& s,
                              const EigsOptions& o);

/// The complete spectrum: every block diagonalised densely.
struct SpectrumResult {
    std::vector<Level> levels;              ///< ascending; one row per block eigenvalue
    std::uint64_t      total_dim = 0;       ///< sum of multiplicities (the Hilbert-space dimension)
    bool               flip_engaged = false;
    bool               tr_engaged   = false;

    /// Every eigenvalue, multiplicities expanded, ascending.
    [[nodiscard]] std::vector<double> expanded() const;
};

[[nodiscard]] SpectrumResult spectrum(const ::Operator& H, int n_sites, const Spec& s);

/// Expand a rep-basis vector into the Sz sector n_up (the C(N, n_up) states in
/// ascending integer order) or, for n_up < 0, the full 2^N space.
[[nodiscard]] std::vector<Complex>
expand(const ed::symmetry::RepSectorData& rd, const std::vector<Complex>& u, int n_up);

/// The degenerate multiplet of `level` in the Sz sector n_up (n_up < 0: full space):
/// the span of the expanded vector under the abelian group, the residues, complex
/// conjugation when the level is time-reversal folded, and the global flip when the
/// basis holds both flip partners. Orthonormal; at most the level's multiplicity
/// vectors (fewer in an Sz sector, which holds only part of a flip-mirrored multiplet).
[[nodiscard]] std::vector<std::vector<Complex>>
multiplet(const Spec& s, int n_sites, const Level& level, const BlockVector& v, int n_up);

}  // namespace ed::sectors
