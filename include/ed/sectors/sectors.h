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
// little-group irrep, streamed one momentum star at a time.
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

#include <ed/core/device.h>
#include <ed/core/operator.h>
#include <ed/solvers/little_group_blocks.h>
#include <ed/symmetry/rep_sector_data.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ed::sectors {

using Complex = std::complex<double>;
using Perm    = std::vector<int>;

/// Where the blocks run (ed::Device, include/ed/core/device.h). place() (select_backend.h)
/// decides each block: Cpu never touches CUDA; Gpu runs every Krylov solve on the device or
/// raises naming the block (isotypic W blocks, OFTLM, blocks without a device kernel); blocks
/// the verb solves densely stay on the host under every device; Auto uses the device above the
/// task's floor in the 'auto' table (auto_row). Results count where the solves ran.
using Device = ed::Device;

/// Things a caller should know about a result that did not stop it: (code, message) pairs,
/// e.g. ("partial_window", ...) for an eigs window returned incomplete under allow_partial.
using Diagnostics = std::vector<std::pair<std::string, std::string>>;

struct Spec {
    std::vector<Perm> abelian;      ///< closed abelian group; empty = identity only
    std::vector<Perm> residues;     ///< point-group coset representatives; each must normalise
                                    ///< `abelian` (p A p^-1 = A), else ed::InvalidRequest
    int  n_up          = -1;        ///< one Sz sector (set-bit count); -1 = every sector
    int  sz_parity     = -1;        ///< one Sz-parity half (set-bit count parity); -1 = both
    bool use_sz        = true;      ///< decompose by Sz / parity when H conserves it
    int  spin_flip     = -1;        ///< -1 auto, 0 off, 1 require
    int  time_reversal = -1;        ///< -1 auto, 0 off, 1 require
    /// Restrict to total spin S = two_S / 2 (-1: no restriction). H must be SU(2)
    /// symmetric; the walk runs the Sz = S sector with every block projected onto the
    /// spin-S tower, and each level counts 2S + 1 times.
    int  two_S         = -1;
    std::vector<int> only_k0;       ///< restrict to these star representatives
    std::vector<int> only_irrep;    ///< restrict to these little-group irreps
    /// A list of character constraints [(element index, chi)]: a constraint on `abelian`
    /// keeps the stars holding a momentum with chi_k(abelian[i]) = chi for every pair; one on
    /// `residues` (-1: the identity) keeps the blocks whose little-co-group irrep has
    /// chi_sigma(residues[i]) = chi, and needs every listed residue in the little group.
    /// A block passes when it meets any one constraint of the list (empty list: all pass).
    using CharConstraint = std::vector<std::pair<int, Complex>>;
    std::vector<CharConstraint> only_momentum;
    std::vector<CharConstraint> only_irrep_chars;
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


// -----------------------------------------------------------------------------
// Lowest-level eigensolve
// -----------------------------------------------------------------------------

struct EigsOptions {
    int  k             = 1;
    bool vectors       = false;
    /// Per-block dense crossover: blocks up to this dimension are diagonalised densely,
    /// larger ones by Krylov. -1: automatic (4x the iteration cap, 1600 for k <= 10).
    int  dense_max_dim = -1;
    bool allow_partial = false; ///< return an incomplete window instead of throwing
    Device device      = Device::Cpu;
    /// Rows each block contributes (0: enough for k given its multiplicity). With `cut`
    /// false every block's rows are returned, not only the lowest k across blocks.
    int  per_block     = 0;
    bool cut           = true;
    /// Solve only the blocks whose 40-step Lanczos estimate lies within prune_margin
    /// (relative) of the k-th level found so far. False: solve every block.
    bool   prune        = true;
    double prune_margin = 0.02;
    /// Also keep every level within `window` (absolute) above the k-th; pruning never drops
    /// a block that could hold one. Each block still contributes only its quota of rows, so
    /// this finds the partners of a level in OTHER blocks (e.g. a degenerate ground state).
    double window       = 0.0;
};

/// Where the solves of a verb ran: a Krylov or a dense solve, on the device or the host, one
/// count per lane place() chose. Under Device::Gpu no Krylov solve runs on the host (a block
/// that cannot run on the device raises); small blocks may still be solved densely there.
/// Units: one per block solve. Prune estimates and setup work (projector shifts, seed
/// projections) are not counted; dynamics counts one per continued fraction at T = 0 and one
/// per source sector at T > 0. A host Krylov solve whose H apply is the device gather on host
/// vectors (the hybrid lane, dim >= kHostGatherFloor) counts host_krylov, since its vectors
/// and Krylov work are on the host; its block_stats lane says "gpu-gather".
struct Placement {
    std::size_t device_krylov = 0, device_dense = 0, host_krylov = 0, host_dense = 0;
    void add(ed::Lane lane) {
        switch (lane) {
            case ed::Lane::HostDense:    ++host_dense;    break;
            case ed::Lane::HostKrylov:   ++host_krylov;   break;
            case ed::Lane::DeviceDense:  ++device_dense;  break;
            case ed::Lane::DeviceKrylov: ++device_krylov; break;
        }
    }
    void add(bool device, bool dense) {
        ++(device ? (dense ? device_dense : device_krylov) : (dense ? host_dense : host_krylov));
    }
    Placement& operator+=(const Placement& o) {
        device_krylov += o.device_krylov; device_dense += o.device_dense;
        host_krylov += o.host_krylov; host_dense += o.host_dense;
        return *this;
    }
};

/// One eigenvalue of one block. The level occurs `multiplicity` times in the spectrum.
struct Level {
    double        energy       = 0.0;
    ed::solvers::LittleGroupBlockTag tag;   ///< the block's quantum numbers
    /// chi_k(a) for every a of Spec::abelian: the momentum of the star representative (the
    /// other members of the star are isospectral and counted in the multiplicity).
    std::vector<Complex> momentum;
    /// (residue index, chi_sigma) over the little co-group, -1 the identity; empty for a
    /// block without a co-group decomposition.
    std::vector<std::pair<int, Complex>> irrep_characters;
    int           mirror       = 1;         ///< flip fold of the subspace
    std::uint64_t multiplicity = 1;         ///< tag.multiplicity x mirror
    int           vector       = -1;        ///< index into EigsResult::vectors, -1 = none
};

/// A certified eigenvector in the rep basis of the block's sector.
struct BlockVector {
    std::shared_ptr<const ed::symmetry::RepSectorData> basis;
    std::vector<Complex> amplitudes;        ///< normalized
};

/// Where one solved block spent its time (one entry per block eigensolve, also logged at Info).
struct BlockStats {
    int           k0 = 0, irrep = -1, flip_parity = -1, n_up = -1;
    std::uint64_t dim = 0;
    std::string   kind;              ///< "group" (full little group), "isotypic" (W), "plain" (k-sector)
    /// How H was applied: "dense" (materialised), "csr" (reduced CSR), "walk" (CSR-free gather),
    /// "gpu-gather" (device kernel on host vectors), "device" (whole solve on the device).
    std::string   lane;
    double        context_orbit_s = 0.0;  ///< the walk's abelian orbit table (shared by its blocks)
    double        star_orbit_s    = 0.0;  ///< the star's own group orbit table
    double        star_build_s    = 0.0;  ///< the whole star build (sector, co-group, blocks)
    double        build_s         = 0.0;  ///< CSR / device-mirror build charged to this block
    std::uint64_t nnz             = 0;    ///< reduced-CSR entries (0 without a CSR)
    std::uint64_t csr_bytes       = 0;
    std::uint64_t applies         = 0;    ///< H applies (device: Krylov iterations)
    double        apply_s         = 0.0;  ///< seconds inside those applies
    /// Solve time outside applies and builds: Krylov vector work (BLAS-1, reorthogonalisation),
    /// tridiagonal / dense eigensolves.
    double        other_s         = 0.0;
    double        solve_s         = 0.0;  ///< the whole block solve
};

struct EigsResult {
    std::vector<Level>       levels;        ///< ascending, cut so that multiplicities reach k
    std::vector<BlockVector> vectors;
    std::uint64_t            total_dim = 0; ///< sum of dim x multiplicity over the walk
    std::size_t              partial_blocks = 0;  ///< blocks that could not certify their rows
    bool                     complete = true;     ///< no uncertified level can lie below the cut
    bool                     flip_engaged = false;
    bool                     tr_engaged   = false;
    std::size_t              device_blocks = 0;   ///< blocks solved on a GPU
    std::size_t              pruned_blocks = 0;   ///< blocks skipped by the estimate test
    Placement                placement;
    Diagnostics              diagnostics;
    std::vector<BlockStats>  block_stats;   ///< one per solved block, in solve order

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
    std::size_t        device_blocks = 0;   ///< blocks diagonalised on a GPU
    Placement          placement;
    Diagnostics        diagnostics;

    /// Every eigenvalue, multiplicities expanded, ascending.
    [[nodiscard]] std::vector<double> expanded() const;
};

[[nodiscard]] SpectrumResult spectrum(const ::Operator& H, int n_sites, const Spec& s,
                                      Device device = Device::Cpu);

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
