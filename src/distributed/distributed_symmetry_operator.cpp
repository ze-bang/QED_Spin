// =============================================================================
// src/distributed/distributed_symmetry_operator.cpp    (Phase 3b #7, stage 2)
//
// Implementation of `ed::distributed::DistributedSymmetryOperator`. See
// the header for the design rationale and honest-scope notes.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_symmetry_operator.h>

// Pulls in the full Operator + SymmetryGroupInfo definitions.
#include <ed/core/construct_ham.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <queue>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::distributed {

namespace {

constexpr std::int64_t kNoOrbit = -1;

// `SectorMetadata::phase_factors` in this codebase is stored per-group-element
// (length |G|), with `phase_factors[a] = chi_q(max_clique[a])` already
// pre-multiplied through the power representation by the sector
// builder (`ed::sym::group_from_generators` in src/symmetry/group.cpp,
// L221-230; matches the QuSpin/Bloch convention test in
// tests/unit/test_symmetry_dsl.cpp L127-139). We therefore do NOT
// re-raise to per-generator powers here -- that's a stale convention
// inherited by the deprecated `Operator::createSymmetrizedVector`
// (which the public symmetry path no longer uses).

}  // namespace

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
DistributedSymmetryOperator::DistributedSymmetryOperator(
    std::shared_ptr<Operator> op,
    std::size_t sector_index,
    MPI_Comm comm)
    : op_(std::move(op)),
      comm_(comm),
      sector_index_(sector_index)
{
    if (!op_) {
        throw std::invalid_argument(
            "DistributedSymmetryOperator: null operator");
    }
    if (comm_ == MPI_COMM_NULL) {
        throw std::invalid_argument(
            "DistributedSymmetryOperator: comm == MPI_COMM_NULL");
    }
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    const std::uint64_t n_bits = op_->getNumBits();
    if (n_bits >= 64) {
        throw std::runtime_error(
            "DistributedSymmetryOperator: n_bits >= 64 unsupported");
    }
    const std::uint64_t dim = (n_bits == 0) ? 1ULL : (1ULL << n_bits);

    const auto& info = op_->symmetry_info;
    if (info.max_clique.empty()) {
        throw std::invalid_argument(
            "DistributedSymmetryOperator: symmetry_info.max_clique is empty "
            "(no group elements -- did you populate symmetry_info?)");
    }
    if (info.power_representation.size() != info.max_clique.size()) {
        throw std::invalid_argument(
            "DistributedSymmetryOperator: power_representation size ("
            + std::to_string(info.power_representation.size())
            + ") != max_clique size ("
            + std::to_string(info.max_clique.size()) + ")");
    }
    if (sector_index_ >= info.sectors.size()) {
        throw std::invalid_argument(
            "DistributedSymmetryOperator: sector_index "
            + std::to_string(sector_index_) + " out of range [0, "
            + std::to_string(info.sectors.size()) + ")");
    }
    const auto& sector = info.sectors[sector_index_];
    if (sector.phase_factors.size() != info.max_clique.size()) {
        throw std::invalid_argument(
            "DistributedSymmetryOperator: sector.phase_factors size ("
            + std::to_string(sector.phase_factors.size())
            + ") != |G| (max_clique size = "
            + std::to_string(info.max_clique.size())
            + "). The codebase convention (per `ed::sym` and "
            "`SectorMetadata` in test_symmetry_dsl.cpp) is one phase "
            "factor PER GROUP ELEMENT, not per generator.");
    }

    op_->commitPendingTransforms();

    // -------------------------------------------------------------------------
    // Step 1: Enumerate orbits + projection coefficients.
    //
    //   For each state b in [0, dim), find its orbit's lex-min representative
    //   `rep` via BFS over generators.  The orbit's contribution to state b
    //   in the unnormalised projected basis is:
    //
    //     ~|orbit(rep)>_b = sum_{g: g(rep) = b} chi_q(g)*
    //
    //   We compute this by iterating g in `max_clique` once per representative
    //   (each rep is visited exactly when we first encounter it via BFS from
    //   `b == rep`).
    // -------------------------------------------------------------------------
    const std::size_t n_group = info.max_clique.size();

    // Per-group-element character: chi_q(max_clique[g]) = phase_factors[g].
    // (See src/symmetry/group.cpp L221-230 for the build-time formula.)
    const std::vector<Complex>& chi = sector.phase_factors;

    // Per-state orbit id (-1 = not in any kept orbit) and projection coeff
    // (~|orbit(rep)>_b; 0 in a zero-norm orbit). Written directly (no raw_*
    // scratch): orbits are disjoint, so the parallel build below is race-free.
    std::vector<std::int64_t> state_to_orbit(dim, kNoOrbit);
    std::vector<Complex>      phi(dim, Complex(0.0, 0.0));

    // ---- Step 1 (parallel) --------------------------------------------------
    // Phase A: parallel lex-min representative scan. A state is its orbit's
    // canonical rep iff it is the numeric minimum over all group images. Skip
    // popcount-mismatched orbits entirely (a whole orbit shares popcount, so it
    // is not in this Sz sector). Per-thread ascending chunks concatenated in
    // thread order keep `reps` globally ascending. O(#reps) extra memory.
    std::vector<std::uint64_t> reps;
    {
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        if (dim < (std::uint64_t{1} << 14)) nthreads = 1;
        if (nthreads < 1) nthreads = 1;
        std::vector<std::vector<std::uint64_t>> local(
            static_cast<std::size_t>(nthreads));
#ifdef _OPENMP
#       pragma omp parallel num_threads(nthreads)
#endif
        {
            int tid = 0, nt = 1;
#ifdef _OPENMP
            tid = omp_get_thread_num();
            nt  = omp_get_num_threads();
#endif
            const std::uint64_t chunk = dim / static_cast<std::uint64_t>(nt);
            const std::uint64_t rem   = dim % static_cast<std::uint64_t>(nt);
            const std::uint64_t b0 = static_cast<std::uint64_t>(tid) * chunk +
                std::min<std::uint64_t>(static_cast<std::uint64_t>(tid), rem);
            const std::uint64_t cnt = chunk +
                (static_cast<std::uint64_t>(tid) < rem ? 1u : 0u);
            auto& out = local[static_cast<std::size_t>(tid)];
            for (std::uint64_t i = 0; i < cnt; ++i) {
                const std::uint64_t s = b0 + i;
                if (info.n_up >= 0 &&
                    static_cast<int>(__builtin_popcountll(s)) != info.n_up) continue;
                bool is_rep = true;
                for (std::size_t g = 0; g < n_group; ++g)
                    if (applyPermutation(s, info.max_clique[g]) < s) { is_rep = false; break; }
                if (is_rep) out.push_back(s);
            }
        }
        std::size_t tot = 0; for (const auto& v : local) tot += v.size();
        reps.reserve(tot);
        for (auto& v : local) reps.insert(reps.end(), v.begin(), v.end());
    }

    // Phase B: parallel per-rep orbit build. Orbits are DISJOINT, so concurrent
    // writes to state_to_orbit[s]/phi[s] for different reps never collide.
    // oid = index in the ascending `reps` list.
    const std::size_t n_raw = reps.size();
    std::vector<std::uint64_t> raw_orbit_sizes(n_raw, 0);
    std::vector<double>        raw_orbit_norms_sq(n_raw, 0.0);
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 64)
#endif
    for (long long oidx = 0; oidx < static_cast<long long>(n_raw); ++oidx) {
        const std::uint64_t rep = reps[static_cast<std::size_t>(oidx)];
        std::set<std::uint64_t> orbit; orbit.insert(rep);
        std::queue<std::uint64_t> q; q.push(rep);
        while (!q.empty()) {
            const std::uint64_t cur = q.front(); q.pop();
            for (const auto& gen : info.generators) {
                const std::uint64_t nxt = applyPermutation(cur, gen);
                if (orbit.insert(nxt).second) q.push(nxt);
            }
        }
        // Project: phi[g(rep)] += conj(chi_q(g)) over all group elements g.
        for (std::size_t g = 0; g < n_group; ++g)
            phi[applyPermutation(rep, info.max_clique[g])] += std::conj(chi[g]);
        double N_i = 0.0;
        for (std::uint64_t s : orbit) {
            N_i += std::norm(phi[s]);
            state_to_orbit[s] = oidx;
        }
        raw_orbit_sizes[static_cast<std::size_t>(oidx)] =
            static_cast<std::uint64_t>(orbit.size());
        raw_orbit_norms_sq[static_cast<std::size_t>(oidx)] = N_i;
    }

    // ---- Step 2: drop zero-norm orbits + renumber dense --------------------
    std::vector<std::int64_t> raw_to_dense(n_raw, kNoOrbit);
    orbit_reps_.clear(); orbit_sizes_.clear(); orbit_norms_sq_.clear();
    orbit_reps_.reserve(n_raw);
    orbit_sizes_.reserve(n_raw);
    orbit_norms_sq_.reserve(n_raw);
    for (std::size_t i = 0; i < n_raw; ++i) {
        if (raw_orbit_norms_sq[i] > kZeroNormTolerance) {
            raw_to_dense[i] = static_cast<std::int64_t>(orbit_reps_.size());
            orbit_reps_.push_back(reps[i]);
            orbit_sizes_.push_back(raw_orbit_sizes[i]);
            orbit_norms_sq_.push_back(raw_orbit_norms_sq[i]);
        }
    }
    // Renumber state_to_orbit into the dense space (parallel; zero filtered).
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (long long b = 0; b < static_cast<long long>(dim); ++b) {
        const std::int64_t raw = state_to_orbit[static_cast<std::size_t>(b)];
        if (raw == kNoOrbit) continue;
        const std::int64_t dense = raw_to_dense[static_cast<std::size_t>(raw)];
        state_to_orbit[static_cast<std::size_t>(b)] = dense;
        if (dense == kNoOrbit) phi[static_cast<std::size_t>(b)] = Complex(0.0, 0.0);
    }

    const std::size_t n_orbits = orbit_reps_.size();

    // -------------------------------------------------------------------------
    // Step 3: Build the LPT orbit partition. Weight = orbit size (proxy for
    // per-orbit row build cost; balances total H * |j> work, since each
    // column build is O(dim) and the projection step iterates all bits of
    // the orbit).
    // -------------------------------------------------------------------------
    partition_ = balanced_orbit_slab(orbit_sizes_, size_);

    // -------------------------------------------------------------------------
    // Step 4: Build the rank-local sparse rows of H_q.
    //
    //   For every orbit j, build dense ~|j> on the full Hilbert space,
    //   apply H, then project onto every orbit i. Keep `(j, H_ij)` only
    //   if `i` is locally owned.
    //
    //   H_ij = (1/sqrt(N_i N_j)) * sum_{b in orbit i} conj(phi^i[b]) * Hj[b]
    //
    //   where phi^i[b] is `phi[b]` (since b uniquely belongs to orbit
    //   state_to_orbit[b]).
    //
    // -- Phase 4 of the "Unified CPU/GPU symmetry architecture" plan -
    //    (May 2026, supersedes Wave 3 / "Unify all 16 matvec cells")
    //
    // The dense-probe construction below is O(n_orbits * 2^N) operator
    // matvecs at setup. The unified-plan replacement walks the orbit of
    // each rep_j directly and emits (i, j, H_ij) triplets via
    // ``ed::matvec::kernel::apply_term_to_state`` (NOW IMPLEMENTED in
    // ``include/ed/matvec/term_kernels.h`` -- the helper yields all
    // (s', h) pairs for a single source state, mirroring the per-bin
    // scan inside ``apply_terms``). Total cost
    // O(n_orbits * |orbit| * num_terms) -- the SAME complexity as the
    // streaming SectorView SpMV. The mathematics is identical: H_ij
    // is bilinear in the per-orbit phi coefficients, so the dense
    // probe and the orbit-walk emit the same triplets up to
    // floating-point order.
    //
    // The orbit-walk replacement is gated behind
    // ``ED_DISTRIBUTED_LEGACY_PROBE``: when unset (default) the legacy
    // probe still runs while the new path bakes in cross-checks for
    // one release. Setting it to ``0`` activates the orbit-walk path
    // (follow-up PR).
    //
    // Implementation skeleton (carried into the orbit-walk PR;
    // ``apply_term_to_state`` is the helper -- everything else exists):
    //
    //   std::vector<std::vector<uint64_t>> orbit_members(n_orbits);
    //   for (uint64_t b = 0; b < dim; ++b) {
    //       const auto i = state_to_orbit[b];
    //       if (i != kNoOrbit) orbit_members[i].push_back(b);
    //   }
    //   for (std::size_t j = 0; j < n_orbits; ++j) {
    //       if (partition_.owner_rank(j) != rank_) continue;
    //       const double norm_j_inv = 1.0 / std::sqrt(orbit_norms_sq_[j]);
    //       std::unordered_map<std::size_t, Complex> col_acc;
    //       for (const auto s_src : orbit_members[j]) {
    //           const Complex pre = phi[s_src] * norm_j_inv;
    //           ed::matvec::kernel::apply_term_to_state<Complex>(
    //               s_src, spin_l_,
    //               op_->terms().diag_one_body,
    //               op_->terms().offdiag_one_body,
    //               op_->terms().diag_two_body,
    //               op_->terms().mixed_two_body,
    //               op_->terms().offdiag_two_body,
    //               op_->terms().three_body,
    //               [&](uint64_t s_dst, Complex h) {
    //                   const auto i = state_to_orbit[s_dst];
    //                   if (i == kNoOrbit) return;
    //                   const double norm_i_inv =
    //                       1.0 / std::sqrt(orbit_norms_sq_[i]);
    //                   col_acc[i] += std::conj(phi[s_dst])
    //                               * norm_i_inv * h * pre;
    //               });
    //       }
    //       // Emit triplets from col_acc into row_col_idx_/row_coeff_;
    //       // populate needed_orbits exactly as the legacy probe.
    //   }
    //
    // Validation gate (bit-exact cross-check at the triplet level)
    // lives in ``tests/unit/test_distributed_symmetry_construction.cpp``;
    // the orbit-walk path is correct iff every emitted (i,j,H_ij)
    // matches the probe path to 1e-12.
    // -------------------------------------------------------------------------
    const std::size_t local_n = partition_.local_size(rank_);
    row_col_idx_.assign(local_n, {});
    row_is_local_.assign(local_n, {});
    row_coeff_.assign(local_n, {});

    // For local orbits, set up a global -> local index lookup. Since the
    // partition's owner_local_index is O(1), we just call it.

    // Track which remote orbits we need (any j we touch with owner != rank_).
    // (j is "needed" only if some local row has a nonzero in column j; we
    // determine that only after computing the row's nonzero pattern.)
    std::set<std::size_t> needed_orbits;

    std::vector<Complex> tilde_j(dim, Complex(0.0, 0.0));
    std::vector<Complex> H_tilde_j(dim, Complex(0.0, 0.0));

    // Precompute 1/sqrt(N_i) for the projection normalisation step.
    std::vector<double> inv_sqrt_N(n_orbits, 0.0);
    for (std::size_t i = 0; i < n_orbits; ++i) {
        inv_sqrt_N[i] = 1.0 / std::sqrt(orbit_norms_sq_[i]);
    }

    // Per-column scratch: accumulated <~|i | H | ~|j>> for every orbit i.
    std::vector<Complex> col_proj(n_orbits, Complex(0.0, 0.0));

    for (std::size_t j = 0; j < n_orbits; ++j) {
        // Build dense ~|j>.
        std::fill(tilde_j.begin(), tilde_j.end(), Complex(0.0, 0.0));
        // Iterate states in orbit j.  We don't have an explicit orbit
        // membership list, but we have state_to_orbit, so we can either
        // (a) walk all states (O(dim) per column -- already paid for the
        //     apply, so order-of-magnitude same), or
        // (b) regenerate the orbit via BFS (O(|orbit| * |gen|)). Both
        //     are bounded; (a) is simpler and lets the compiler vectorise.
        // Use (a) for code simplicity.
        for (std::uint64_t b = 0; b < dim; ++b) {
            if (state_to_orbit[b] == static_cast<std::int64_t>(j)) {
                tilde_j[b] = phi[b];
            }
        }

        // Apply H: H_tilde_j = H * tilde_j.
        op_->apply(tilde_j.data(), H_tilde_j.data(), static_cast<size_t>(dim));

        // Project onto every orbit i: col_proj[i] = sum_{b in orbit i}
        //   conj(phi[b]) * H_tilde_j[b].
        std::fill(col_proj.begin(), col_proj.end(), Complex(0.0, 0.0));
        for (std::uint64_t b = 0; b < dim; ++b) {
            const std::int64_t i = state_to_orbit[b];
            if (i == kNoOrbit) continue;
            if (std::abs(H_tilde_j[b]) < 1e-300) continue;
            col_proj[static_cast<std::size_t>(i)]
                += std::conj(phi[b]) * H_tilde_j[b];
        }

        // Normalise + record local-row entries.
        const double norm_j = inv_sqrt_N[j];
        for (std::size_t i = 0; i < n_orbits; ++i) {
            const Complex H_ij = col_proj[i] * (inv_sqrt_N[i] * norm_j);
            if (std::abs(H_ij) < kSparsityTolerance) continue;

            const int owner = partition_.owner_rank(i);
            if (owner != rank_) continue;
            const std::size_t local_row = partition_.owner_local_index(i);

            row_col_idx_[local_row].push_back(j);
            row_coeff_[local_row].push_back(H_ij);
            // is_local resolved after we know which j are remote.
            const int j_owner = partition_.owner_rank(j);
            const bool j_local = (j_owner == rank_);
            row_is_local_[local_row].push_back(j_local ? 1u : 0u);
            if (!j_local) {
                needed_orbits.insert(j);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: Build the orbit halo plan + remap remote column indices to
    // halo buffer offsets.
    // -------------------------------------------------------------------------
    const std::vector<std::size_t> needed_vec(needed_orbits.begin(),
                                              needed_orbits.end());
    halo_plan_ = std::make_unique<OrbitHaloPlan>(partition_, needed_vec, comm_);

    // Phase 8: pre-size the persistent halo recv buffer. The plan's
    // recv_total() is fixed for the lifetime of *this, so apply() never
    // needs to (re)allocate. Empty vector for ranks with no incoming halo
    // is fine -- exchange() handles that case symmetrically (see apply()).
    halo_buf_.assign(halo_plan_->recv_total(), Complex(0.0, 0.0));

    // Build halo_index_ table: orbit_id -> position in halo recv buffer.
    halo_index_.assign(n_orbits, kHaloMissing);
    {
        const auto& recv_ids = halo_plan_->recv_orbit_id();
        for (std::size_t k = 0; k < recv_ids.size(); ++k) {
            const std::uint64_t oid = recv_ids[k];
            if (oid < halo_index_.size()) {
                halo_index_[static_cast<std::size_t>(oid)] = k;
            }
        }
    }

    // Rewrite per-row col indices: local entries get owner_local_index(j),
    // remote entries get halo_index_[j].
    for (std::size_t r = 0; r < local_n; ++r) {
        for (std::size_t k = 0; k < row_col_idx_[r].size(); ++k) {
            const std::size_t j = row_col_idx_[r][k];
            if (row_is_local_[r][k]) {
                row_col_idx_[r][k] = partition_.owner_local_index(j);
            } else {
                const std::size_t h = halo_index_[j];
                if (h == kHaloMissing) {
                    throw std::logic_error(
                        "DistributedSymmetryOperator: remote orbit "
                        + std::to_string(j) + " missing from halo plan "
                        "after construction (rank=" + std::to_string(rank_)
                        + ")");
                }
                row_col_idx_[r][k] = h;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Apply: distributed symmetry-projected SpMV.
// -----------------------------------------------------------------------------
void DistributedSymmetryOperator::apply(const Complex* x_local,
                                        Complex* y_local) const {
    const std::size_t local_n = row_col_idx_.size();

    // Halo exchange first so subsequent local SpMV can read freely.
    // halo_buf_ was pre-sized in the constructor; we reuse it on every
    // apply() to avoid per-matvec heap activity (Phase 8). exchange()
    // tolerates an empty recv buffer when peers still have sends to us.
    if (halo_plan_) {
        halo_plan_->exchange(x_local, halo_buf_.data());
    }
    const Complex* const halo = halo_buf_.data();

    // Rank-local CSR-style SpMV.
    for (std::size_t r = 0; r < local_n; ++r) {
        const auto& cols  = row_col_idx_[r];
        const auto& isloc = row_is_local_[r];
        const auto& coefs = row_coeff_[r];
        Complex acc(0.0, 0.0);
        for (std::size_t k = 0; k < cols.size(); ++k) {
            const Complex x = isloc[k] ? x_local[cols[k]] : halo[cols[k]];
            acc += coefs[k] * x;
        }
        y_local[r] = acc;
    }
}

// -----------------------------------------------------------------------------
// Geometry / diagnostics
// -----------------------------------------------------------------------------
std::size_t DistributedSymmetryOperator::dim() const {
    // Phase 2 MatVecOperator: local orbit count (per-rank).
    return partition_.local_size(rank_);
}

std::size_t DistributedSymmetryOperator::global_dim() const {
    // Phase 2 MatVecOperator: total # of nonzero-norm orbits.
    return orbit_reps_.size();
}

std::string DistributedSymmetryOperator::description() const {
    return "DistributedSymmetryOperator(sector=" + std::to_string(sector_index_)
        + ", local=" + std::to_string(dim())
        + ", global=" + std::to_string(global_dim())
        + ", nproc=" + std::to_string(size_) + ")";
}

std::uint64_t DistributedSymmetryOperator::local_size() const noexcept {
    return static_cast<std::uint64_t>(partition_.local_size(rank_));
}

std::uint64_t DistributedSymmetryOperator::local_offset() const noexcept {
    if (rank_ < 0 || rank_ >= static_cast<int>(partition_.rank_offsets.size()))
        return 0;
    return static_cast<std::uint64_t>(partition_.rank_offsets[rank_]);
}

std::size_t DistributedSymmetryOperator::local_nnz() const noexcept {
    std::size_t n = 0;
    for (const auto& row : row_coeff_) n += row.size();
    return n;
}

}  // namespace ed::distributed

#endif  // WITH_MPI
