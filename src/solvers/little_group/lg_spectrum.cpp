// =============================================================================
// src/solvers/little_group/lg_spectrum.cpp -- full / lowest spectra and the public block-set factory
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

namespace ed::solvers {

using namespace lg_detail;

std::vector<double> LittleGroupSpectrum::expanded() const {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(total_dim));
    for (std::size_t i = 0; i < eigenvalues.size(); ++i)
        for (int r = 0; r < multiplicities[i]; ++r)
            out.push_back(eigenvalues[i]);
    std::sort(out.begin(), out.end());
    return out;
}

namespace {

// Full-spectrum sum rule: multiplicities must tile the subspace.
void check_sum_rule(const LittleGroupSpectrum& out, int n_sites,
                    const LittleGroupOptions& opt) {
    std::uint64_t want;
    if (opt.n_up >= 0) {
        long double c = 1.0L;
        const int kk = std::min(opt.n_up, n_sites - opt.n_up);
        for (int i = 0; i < kk; ++i)
            c = c * (n_sites - i) / (i + 1);
        want = static_cast<std::uint64_t>(c + 0.5L);
    } else if (opt.sz_parity >= 0) {
        want = std::uint64_t{1} << (n_sites - 1);
    } else {
        want = std::uint64_t{1} << n_sites;
    }
    if (out.total_dim != want) {
        // A star filter (env job-split, plan mode, or a caller naming a
        // momentum via opt.only_k0) means the walk deliberately covered a
        // SUBSET, so the tiling identity cannot hold -- skip with a loud note
        // rather than raise a false bookkeeping error. Restricted to exactly
        // these two causes: the sum rule is THE tripwire that catches a real
        // covering bug, so it must still fire for every unrestricted call.
        const bool restricted =
            !ed::env::text("ED_SYM_LG_ONLY_K0").empty()
            || !opt.only_k0.empty() || !opt.only_irrep.empty()
            || opt.plan_only;
        if (restricted) {
            std::fprintf(stderr,
                "[little_group] sum rule SKIPPED: star filter active "
                "(covered %llu of %llu)\n",
                static_cast<unsigned long long>(out.total_dim),
                static_cast<unsigned long long>(want));
            return;
        }
        throw std::runtime_error(
            "little_group_full_spectrum: multiplicity sum "
            + std::to_string(out.total_dim) + " != subspace dim "
            + std::to_string(want) + " (internal bookkeeping error).");
    }
}

}  // namespace

// =============================================================================
// U1a: the public block-set factory -- the same star walk as run_little_group
// with the solves omitted. Consumed by the U1b thermal lane, the unit tests,
// and (U2a) the projected ground-state path.
// =============================================================================
LittleGroupBlockSet build_little_group_blocks(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const LittleGroupOptions&            opt)
{
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, residue_perms, n_sites, opt,
                        cx, tr_on);
    const auto stars = star_partition(cx, tr_on);

    // Honour the job-splitting env override like every other consumer.
    // "plan" is meaningless here (this factory never solves) -- ignored.
    bool ignore_plan = false;
    std::set<int> only_k0(opt.only_k0.begin(), opt.only_k0.end());
    parse_only_k0_env(only_k0, ignore_plan);

    LittleGroupBlockSet set;
    set.meta.flip_engaged = cx.flip_half;
    set.meta.tr_engaged   = tr_on;
    set.meta.irrep_characters.reserve(static_cast<std::size_t>(cx.n_irr_raw));
    for (int kk = 0; kk < cx.n_irr_raw; ++kk)
        set.meta.irrep_characters.push_back(
            cx.giA.irreps[static_cast<std::size_t>(kk)].character);
    for (const auto& [k0, members] : stars) {
        if (!only_k0.empty() && only_k0.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(
            op, cx, tr_on, k0, members, opt, /*plan_print=*/false,
            nullptr, nullptr, nullptr);
        for (auto& bi : sb.blocks) {
            set.meta.total_dim += bi->tag.dim * bi->tag.multiplicity;
            set.blocks.emplace_back(std::move(bi));
        }
        set.meta.stars.push_back(std::move(sb.info));
    }
    return set;
}

LittleGroupSpectrum little_group_full_spectrum(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    const LittleGroupOptions&            opt)
{
    // GPU lane (restored 2026-07-20; the Family-6 removal of the SAB engine
    // took the batched eigensolver down with it and silently no-op'd
    // opt.use_gpu -- the kernel now lives in little_group_gpu.cu, SAB-free):
    // materialise every block on the host (same engine sampling as the CPU
    // path, the callback defers), then ONE batched cuSOLVER stream-pool call
    // solves them all. Any GPU failure degrades to the CPU path below.
#ifdef WITH_CUDA
    if (lg_gpu_eigensolve_enabled(opt)) {
        try {
            ed::solvers::LgBlocksPacked P;
            std::vector<int> pack_mult;
            std::vector<LittleGroupLabel> pack_label;
            auto out = run_little_group(
                op, abelian_group, residue_perms, n_sites, opt,
                [&](const ed::matvec::MatVecOperator& mv, int mult,
                    LittleGroupLabel& lab) -> std::vector<double> {
                    const std::size_t nb = mv.dim();
                    if (nb == 0) return {};
                    const Eigen::MatrixXcd Hb = materialize(mv);
                    P.offset.push_back(P.data.size());
                    P.block_dim.push_back(static_cast<int>(nb));
                    P.block_irrep_dim.push_back(1);
                    for (std::size_t col = 0; col < nb; ++col)
                        for (std::size_t row = 0; row < nb; ++row)
                            P.data.push_back(Hb(static_cast<Eigen::Index>(row),
                                                static_cast<Eigen::Index>(col)));
                    pack_mult.push_back(mult);
                    pack_label.push_back(lab);
                    return {};       // deferred: recorded above
                });
            const std::vector<double> eigs =
                ed::solvers::lg_blocks_batched_eigenvalues_gpu(P);
            // Same ED_SYM_PROFILE marker the rep-gather lane carries: this
            // lane was previously silent unless it FAILED, so "the GPU
            // eigensolve ran" could only be assumed, never observed -- and an
            // assumed device lane is how CPU numbers get recorded as GPU ones.
            if (ed::env::flag("ED_SYM_PROFILE", false)) {
                std::fprintf(stderr,
                             "[sym_profile] little-group batched GPU "
                             "eigensolve engaged (%zu blocks)\n",
                             P.block_dim.size());
            }
            std::size_t off = 0;
            for (std::size_t b = 0; b < P.block_dim.size(); ++b) {
                for (int i = 0; i < P.block_dim[b]; ++i) {
                    out.eigenvalues.push_back(
                        eigs[off + static_cast<std::size_t>(i)]);
                    out.multiplicities.push_back(pack_mult[b]);
                    out.labels.push_back(pack_label[b]);
                }
                off += static_cast<std::size_t>(P.block_dim[b]);
            }
            out.total_dim = 0;
            for (int m : out.multiplicities)
                out.total_dim += static_cast<std::uint64_t>(m);
            // Audit 2026-07-31: truthful only when the deferred batch
            // actually held blocks -- an all-Lanczos walk hands the GPU
            // eigensolve an EMPTY pack (it returns without touching the
            // device), and stamping true regardless made every GPU-lane
            // assertion keyed on this flag toothless.
            if (!P.block_dim.empty()) out.gpu_engaged = true;
            check_sum_rule(out, n_sites, opt);
            return out;
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "[little_group] GPU batched eigensolve declined "
                         "(%s); using the CPU path\n", e.what());
        }
    }
#endif
    auto out = run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [](const ed::matvec::MatVecOperator& mv, int /*mult*/,
           LittleGroupLabel& /*lab*/) {
            return solve_block_full(mv);
        });
    check_sum_rule(out, n_sites, opt);
    return out;
}

LittleGroupSpectrum little_group_lowest_spectrum(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    int                                  k,
    const LittleGroupOptions&            opt)
{
    // GPU lane (2026-07-20): dense-eligible blocks (the same crossover
    // decision as solve_block_lowest, via lowest_dense_floor) are DEFERRED
    // into one packed batch and eigensolved in a single cuSOLVER stream-pool
    // call; Lanczos-sized blocks solve inline on the CPU kernel exactly as
    // before (at >= 2^20 reps their matvec engages the GPU rep-gather). Any
    // GPU failure degrades to the all-CPU path below.
#ifdef WITH_CUDA
    if (lg_gpu_eigensolve_enabled(opt)) {
        try {
            ed::solvers::LgBlocksPacked P;
            std::vector<int> pack_mult;
            std::vector<int> pack_keep;   // lowest-k kept per deferred block
            std::vector<LittleGroupLabel> pack_label;
            auto out = run_little_group(
                op, abelian_group, residue_perms, n_sites, opt,
                [&](const ed::matvec::MatVecOperator& mv, int mult,
                    LittleGroupLabel& lab) -> std::vector<double> {
                    const std::uint64_t nb = mv.dim();
                    if (nb == 0) return {};
                    const std::size_t kk =
                        static_cast<std::size_t>(std::max<std::uint64_t>(
                            1u, std::min<std::uint64_t>(
                                    static_cast<std::uint64_t>(k), nb)));
                    if (nb <= lowest_dense_floor(kk, opt.dense_max_dim)
                            || nb <= 2) {
                        const Eigen::MatrixXcd Hb = materialize(mv);
                        P.offset.push_back(P.data.size());
                        P.block_dim.push_back(static_cast<int>(nb));
                        P.block_irrep_dim.push_back(1);
                        for (std::size_t col = 0; col < nb; ++col)
                            for (std::size_t row = 0; row < nb; ++row)
                                P.data.push_back(
                                    Hb(static_cast<Eigen::Index>(row),
                                       static_cast<Eigen::Index>(col)));
                        pack_mult.push_back(mult);
                        pack_keep.push_back(static_cast<int>(kk));
                        lab.converged = true;   // dense spectrum is exact
                        pack_label.push_back(lab);
                        return {};   // deferred
                    }
                    bool conv = true;
                    auto ev = solve_block_lowest(mv, k, opt.dense_max_dim,
                                                 &conv, opt.block_size);
                    lab.converged = conv;
                    return ev;
                });
            const std::vector<double> eigs =
                ed::solvers::lg_blocks_batched_eigenvalues_gpu(P);
            // Same ED_SYM_PROFILE marker the rep-gather lane carries: this
            // lane was previously silent unless it FAILED, so "the GPU
            // eigensolve ran" could only be assumed, never observed -- and an
            // assumed device lane is how CPU numbers get recorded as GPU ones.
            if (ed::env::flag("ED_SYM_PROFILE", false)) {
                std::fprintf(stderr,
                             "[sym_profile] little-group batched GPU "
                             "eigensolve engaged (%zu blocks)\n",
                             P.block_dim.size());
            }
            std::size_t off = 0;
            for (std::size_t b = 0; b < P.block_dim.size(); ++b) {
                const int keep = std::min(pack_keep[b], P.block_dim[b]);
                for (int i = 0; i < keep; ++i) {   // ascending per block
                    out.eigenvalues.push_back(
                        eigs[off + static_cast<std::size_t>(i)]);
                    out.multiplicities.push_back(pack_mult[b]);
                    out.labels.push_back(pack_label[b]);
                    out.total_dim +=
                        static_cast<std::uint64_t>(pack_mult[b]);
                }
                off += static_cast<std::size_t>(P.block_dim[b]);
            }
            // Audit 2026-07-31: truthful only when the deferred batch
            // actually held blocks -- an all-Lanczos walk hands the GPU
            // eigensolve an EMPTY pack (it returns without touching the
            // device), and stamping true regardless made every GPU-lane
            // assertion keyed on this flag toothless.
            if (!P.block_dim.empty()) out.gpu_engaged = true;
            return out;
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "[little_group] GPU batched eigensolve declined "
                         "(%s); using the CPU path\n", e.what());
        }
    }
#endif
    // CPU deferred dense batch (audit 2026-07-30) -- the CPU twin of the
    // GPU packed-batch lane above. The serial star walk solved each
    // dense-eligible block inline, and at the dense crossover (~1e3-1e4
    // dims) a threaded zheevd is LATENCY-bound (OpenBLAS pthread pool
    // spin-wait per small zgemv inside zhetrd; Eigen before it was the
    // same wall single-threaded): measured 26-31 s for the N=18 ring walk
    // whose abelian twin runs in 0.7 s. Defer dense-eligible blocks
    // (materialized under a byte budget, env ED_SYM_LG_DENSE_BATCH_GIB,
    // default 8), then eigensolve them in PARALLEL across blocks with the
    // BLAS pinned serial inside each task. Lanczos-sized blocks still
    // solve inline (their kernels are internally parallel). Row ordering
    // matches the GPU lane's convention: inline rows during the walk,
    // deferred rows appended after (consumers sort / read parallel
    // arrays; neither lane guarantees walk order).
    struct CpuDeferred {
        Eigen::MatrixXcd  Hb;
        std::size_t       keep;
        int               mult;
        LittleGroupLabel  lab;
    };
    std::vector<CpuDeferred> defer;
    std::uint64_t defer_bytes  = 0;
    std::uint64_t defer_budget = std::uint64_t{8} << 30;
    if (const double g = ed::env::real("ED_SYM_LG_DENSE_BATCH_GIB", -1.0); g >= 0.0)
        defer_budget = static_cast<std::uint64_t>(g * (1ULL << 30));
    auto out = run_little_group(
        op, abelian_group, residue_perms, n_sites, opt,
        [&](const ed::matvec::MatVecOperator& mv, int mult,
            LittleGroupLabel& lab) -> std::vector<double> {
            const std::uint64_t nb = mv.dim();
            if (nb == 0) return {};
            const std::size_t kk =
                static_cast<std::size_t>(std::max<std::uint64_t>(
                    1u, std::min<std::uint64_t>(
                            static_cast<std::uint64_t>(k), nb)));
            const bool dense =
                nb <= lowest_dense_floor(kk, opt.dense_max_dim) || nb <= 2;
            if (dense) {
                const std::uint64_t bytes = nb * nb * sizeof(Complex);
                if (defer_bytes + bytes <= defer_budget) {
                    lab.converged = true;   // dense spectrum is exact
                    defer.push_back({materialize(mv), kk, mult, lab});
                    defer_bytes += bytes;
                    return {};              // deferred
                }
                // Budget exhausted: solve inline (exact, just serial).
            }
            bool conv = true;
            auto ev = solve_block_lowest(mv, k, opt.dense_max_dim, &conv,
                                         opt.block_size);
            lab.converged = conv;
            return ev;
        });
    if (!defer.empty()) {
        std::vector<std::vector<double>> evs(defer.size());
        std::exception_ptr eptr;
#ifdef _OPENMP
        const int batch_threads = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(omp_get_max_threads()), defer.size()));
        // Pin the BLAS pool serial for the batch: each task runs its own
        // single-threaded zheevd; block-level parallelism replaces the
        // (latency-dominated) intra-solve threading. The explicit
        // num_threads clause overrides the scope's OMP cap for THIS
        // region; nested regions inside the solve collapse to serial.
        ed::parallel::ThreadBudgetScope blas_serial(1);
#       pragma omp parallel for schedule(dynamic) \
            num_threads(batch_threads)
#endif
        for (long long i = 0; i < static_cast<long long>(defer.size()); ++i) {
            try {
                evs[static_cast<std::size_t>(i)] = dense_eigenvalues_inplace(
                    defer[static_cast<std::size_t>(i)].Hb);
            } catch (...) {
#ifdef _OPENMP
#               pragma omp critical(lg_dense_batch_eptr)
#endif
                { if (!eptr) eptr = std::current_exception(); }
            }
        }
        if (eptr) std::rethrow_exception(eptr);
        for (std::size_t b = 0; b < defer.size(); ++b) {
            const auto& d = defer[b];
            const std::size_t keep =
                std::min<std::size_t>(d.keep, evs[b].size());
            for (std::size_t i = 0; i < keep; ++i) {   // ascending per block
                out.eigenvalues.push_back(evs[b][i]);
                out.multiplicities.push_back(d.mult);
                out.labels.push_back(d.lab);
                // run_little_group summed total_dim over the INLINE rows
                // only (the walk saw `{}` for deferred blocks); mirror the
                // GPU packed lane's per-row accounting here.
                out.total_dim += static_cast<std::uint64_t>(d.mult);
            }
        }
    }
    return out;
}

std::vector<double> little_group_lowest_eigenvalues(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    const std::vector<std::vector<int>>& residue_perms,
    int                                  n_sites,
    int                                  k,
    const LittleGroupOptions&            opt)
{
    auto spec = little_group_lowest_spectrum(
        op, abelian_group, residue_perms, n_sites, k, opt);
    // A flat list has nowhere to carry a per-block convergence flag, and an
    // unconverged block returns only the converged PREFIX of its levels -- so its
    // missing levels could belong in the global lowest k. Returning the list anyway
    // (or an empty one) is how this verb used to fail silently; refuse instead.
    const std::size_t unconverged = spec.unconverged_blocks;
    if (unconverged > 0) {
        throw std::runtime_error(
            "little_group_lowest_eigenvalues: " + std::to_string(unconverged)
            + " block(s) did not converge their lowest levels within the iteration "
            "budget, so the global lowest-" + std::to_string(k) + " list cannot be "
            "certified. little_group_lowest_eigenvalues_labeled reports which blocks; "
            "raise ED_SYM_LG_LOWEST_MAX_ITER to give them more iterations.");
    }
    std::vector<double> flat = spec.expanded();
    if (static_cast<int>(flat.size()) > k)
        flat.resize(static_cast<std::size_t>(k));
    return flat;
}

}  // namespace ed::solvers
