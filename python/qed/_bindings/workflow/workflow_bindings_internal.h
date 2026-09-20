// =============================================================================
// python/qed/_bindings/workflow/workflow_bindings_internal.h
//
// Internal (binding-private) header for the `bind_workflows` translation
// units in this directory. Carries the include set every workflow binding
// TU needs plus the helpers that used to live in the anonymous namespace of
// the former monolithic `workflow_bindings.cpp` (WP11 split, Sep 2026):
// the probe loader / composition resolver, the GPU-lane probes and the
// silent-fallback warning, the SU(2) hoists, the symmetric-source variant
// and the cross-irrep ground-state scan.
//
// The helpers live in `workflow_bindings_detail` (external linkage, `inline`
// so the definitions merge across TUs); each TU pulls them in with a
// file-scope `using namespace workflow_bindings_detail;`. Nothing here is
// part of the Python surface -- that is declared in `workflow_bindings.h`.
//
// NOT installed and NOT included from outside `python/qed/_bindings`.
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <Eigen/Dense>                   // degenerate-multiplet S^2 rotation (2026-09-11)
#include <ed/config/env_registry.h>
#include <ed/core/hdf5_io.h>             // isDisabledOutputPath
#include <ed/core/fixed_sz_operator.h>   // FixedSzOperator (bound pybind type)
#include <ed/core/linear_operator.h>
#include <ed/core/make_operator.h>
#include <ed/symmetry/canonical_thermo.h>      // single canonical-thermo impl
#include <ed/symmetry/spin_flip.h>
#include <ed/symmetry/observable_character.h>  // Stage 8d probe classifiers
#include <ed/symmetry/commute_check.h>         // A2 [H,g]=0 validation
#include <ed/symmetry/sector_plan.h>
#include <ed/symmetry/time_reversal.h>
#include <ed/symmetry/sym_profile.h>
#include <ed/core/operator.h>
#include <ed/core/results.h>
#include <ed/core/sector_loop.h>          // filter_sectors + resolve_target_sector
#include <ed/core/sector_thermo.h>        // combine_sector_thermodynamics (SOTA)
#include <ed/core/select_backend.h>
#include <ed/dssf/cross_sector_orbit_observable.h>  // SOTA cross-irrep observable
#include <ed/matvec/backends/cpu_backend.h>          // CpuBackend for cf_spectral_from_vector
#include <ed/krylov/lanczos_kernel.h>                // CGS2 GS refinement (ensure_gs_residual)
#include <ed/solvers/lanczos.h>                      // full_diagonalization (Stage 12f exact tower route)
#include <ed/core/blas_lapack_wrapper.h>             // LAPACKE_dstevd (GS refinement)
#include <ed/observables/cf_spectral_kernel.h>      // cf_spectral_from_vector
#include <ed/observables/ftlm_cross_irrep_kernel.h>  // SOTA finite-T cross-irrep
#include <ed/orchestrator.h>
#include <ed/operators/casimir.h>                    // Stage 12: S^2 carrier + labels
#include <ed/symmetry/casimir_projector.h>           // Stage 12: Lowdin targeting
#include <ed/symmetry/su2.h>                         // Stage 12: SU(2) detection
#include <ed/solvers/little_group_solve.h>           // make_rep_sector_matvec
#include <ed/solvers/kpm_dos.h>                      // Wave B3: estimate_spectral_bounds

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <exception>
#include <thread>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace py = pybind11;

namespace workflow_bindings_detail {

using Complex = std::complex<double>;

// (rank, size) on MPI_COMM_WORLD, or (0,1) when MPI is unavailable / not
// initialized. Lets the in-process symmetry sector loops distribute their
// independent per-sector work across ranks when launched under
// ``mpirun -n N python ...`` (mpi4py / a launcher initializes MPI).
inline std::pair<int, int> binding_mpi_rank_size() {
    int rank = 0, size = 1;
#ifdef WITH_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
#endif
    return {rank, size};
}

// B6: resolve the sector-level OMP gate. An explicit ED_SYM_SECTOR_PARALLEL
// (0/1) always wins. When UNSET, auto-enable only in the many-tiny-sectors
// regime (composed parity x flip x spatial => hundreds of ~tiny blocks),
// where each inner solve is trivial so outer parallelism is a clear win with
// negligible oversubscription. Few-large-sector runs (max_dim > cap) stay
// serial-outer, preserving the inner Lanczos/BLAS threading.
inline bool resolve_sector_parallel(std::size_t   num_sectors,
                                    std::uint64_t max_sector_dim,
                                    bool          gpu_lane) {
    // Audit 2026-07-31: a single-sector call gains NOTHING from a
    // 1-iteration parallel region but loses everything to its side
    // effect -- full_diagonalization sees omp_in_parallel() and pins
    // the dense eigensolve to ONE thread. qed.full_spectrum's forced
    // ED_SYM_SECTOR_PARALLEL=1 hit exactly this on trivial-symmetry
    // clusters (one identity-irrep sector per Sz call): 17 serial
    // blocks x single-threaded dsyevd, measured 283 s at N=16 where
    // threaded LAPACK does it in a fraction. Declining here is
    // semantics-preserving (there is no parallelism to enable), so it
    // outranks even the explicit env.
    if (num_sectors <= 1) return false;
    if (const char* env = ed::env::raw("ED_SYM_SECTOR_PARALLEL"))
        return env[0] == '1';   // explicit override always wins (user's risk)
    // NEVER auto-enable on the GPU lane: the per-sector solves launch CUDA
    // kernels / build device mirrors, which are not safe to call concurrently
    // from OMP threads on the default stream.
    if (gpu_lane) return false;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return num_sectors >= 4u * static_cast<std::size_t>(hw)
        && max_sector_dim > 0 && max_sector_dim <= 4096;
}

// Largest sector dim across a filtered index list of a SectorSetView (O(1)
// dim() per sector, no materialisation).
template <class HandleT, class IdxContainer>
inline std::uint64_t max_sector_dim(const HandleT& handle,
                                    const IdxContainer& indices) {
    std::uint64_t m = 0;
    for (auto k : indices) {
        auto s = handle.sector(static_cast<std::size_t>(k));
        if (s) m = std::max<std::uint64_t>(m, s->dim());
    }
    return m;
}

// ---------------------------------------------------------------------------
// Stage 8d (SymmetryEngine v2) DSSF helpers.
// ---------------------------------------------------------------------------

// Decode the 6-tuple probe-transform rows shared by every cross-irrep
// spectral binding (op_type, site, coeff, is_two_body, op_type_2, site_2).
inline std::vector<Operator::TransformData>
decode_probe_transforms(const std::vector<py::tuple>& rows, const char* who) {
    std::vector<Operator::TransformData> tlist;
    tlist.reserve(rows.size());
    for (const auto& row : rows) {
        if (row.size() < 6) {
            throw std::invalid_argument(
                std::string(who) + ": each transform must be a 6-tuple "
                "(op_type, site, coeff, is_two_body, op_type_2, site_2).");
        }
        Operator::TransformData t;
        t.op_type      = static_cast<uint8_t>(row[0].cast<int>());
        t.site_index   = row[1].cast<std::uint64_t>();
        t.coefficient  = row[2].cast<std::complex<double>>();
        t.is_two_body  = row[3].cast<bool>();
        t.op_type_2    = static_cast<uint8_t>(row[4].cast<int>());
        t.site_index_2 = row[5].cast<std::uint64_t>();
        tlist.push_back(t);
    }
    return tlist;
}

// Pick the observable ref for a sector: the CSR-free RepSectorData when the
// sector runs the rep-lazy lane (flip-extended / Sz-parity sectors NEVER have
// an orbit CSR; ordinary fixed-Sz sectors avoid materialising one), otherwise
// the materialised orbit basis (eager small sectors).
inline ed::dssf::CrossSectorOrbitObservable::OperatorRef
make_cross_sector_ref(ed::symmetry::SectorOperator* sec,
                      std::uint64_t                 num_sites) {
    using Ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef;
    if (sec->rep_lazy()) {
        const auto& rd = sec->basis().ensureRepData();
        if (rd.usable()) return Ref::from_rep(rd, num_sites);
    }
    if (!sec->csr_available()) {
        throw std::runtime_error(
            "cross-irrep spectral: sector carries neither a usable "
            "RepSectorData nor a materialisable orbit CSR.");
    }
    return Ref::from(sec->materialized_basis(), num_sites);
}

// ---------------------------------------------------------------------------
// Structural cleanup (Jul 2026): ONE source probe per binding call.
//
// Every symmetry binding needs (a) the term-level TermStorage for detection
// and (b) the loaded symmetry_info for composition -- and then hands the
// SAME loaded content to ``make_sector_operators_tagged``. This helper loads
// the carrier once (``ed::detail::load_symmetric_base``: directory parse or
// in-memory copy, per the spec's source); the caller passes ``probe.base``
// into the factory so the source is loaded exactly once per call
// (previously: probe + factory each parsed it).
// ---------------------------------------------------------------------------
struct DirectoryProbe {
    std::shared_ptr<Operator>  base;   // terms + symmetry_info loaded
    ed::matvec::TermStorage    soa;    // classified term SoA (detection)
};

inline DirectoryProbe
load_probe(const ed::OperatorSpec& spec) {
    DirectoryProbe p;
    p.base = ed::detail::load_symmetric_base(spec);
    ed::matvec::TermStorage::classify_route(
        p.soa, p.base->transform_data_, p.base->three_body_data_,
        [](const std::complex<double>& c) { return c; });
    return p;
}

// Composition resolution over a loaded probe, with the Stage-7a star maps
// folded in (shared by the GS / thermal / flat-pool bindings).
template <class OptsT>
inline ed::symmetry::SymmetryComposition
resolve_comp_with_stars(const DirectoryProbe& probe, const OptsT& opts) {
    auto comp = ed::symmetry::resolve_symmetry_composition(
        probe.soa, probe.base->symmetry_info, opts.backend.allow_gpu,
        ed::symmetry::sym_toggle_from_int(opts.spin_flip),
        ed::symmetry::sym_toggle_from_int(opts.time_reversal));
    const std::size_t nsec = probe.base->symmetry_info.sectors.size();
    for (const auto& m : opts.star_maps) {
        if (m.size() != nsec) continue;
        comp.star_maps.emplace_back(m.begin(), m.end());
    }
    return comp;
}

// Synthetic-slot selection rule for the probe: how the trailing
// (parity[, flip]) tag labels of the source sector map to the target's.
// Throws when the probe cannot be routed through the engaged slots.
struct SlottedSelection {
    std::size_t      n_slots = 0;
    std::vector<int> signs;
};

inline SlottedSelection
slotted_selection_for(const ed::OperatorSpec&                     spec,
                      const std::vector<Operator::TransformData>& tlist,
                      const char*                                 who) {
    SlottedSelection s;
    if (spec.sz_parity.has_value()) {
        ++s.n_slots;
        const int dp = ed::symmetry::delta_n_up_parity(tlist);
        if (dp < 0) {
            throw std::invalid_argument(
                std::string(who) + ": probe mixes even and odd Sz "
                "selection rules and cannot be routed through Sz-parity "
                "sectors; disable the parity lane for this probe.");
        }
        s.signs.push_back(dp == 1 ? -1 : +1);
    }
    if (spec.flip_sectors_full) {
        ++s.n_slots;
        const int fc = ed::symmetry::spin_flip_character(tlist);
        if (fc == 0) {
            throw std::invalid_argument(
                std::string(who) + ": probe has no definite spin-flip "
                "character (X O X != +-O) and cannot be routed through "
                "flip sectors; pass flip_sectors=False for this probe.");
        }
        s.signs.push_back(fc);
    }
    return s;
}

#ifdef WITH_MPI
// Across-sector finite-T recombination: every rank holds the per-sector
// ThermodynamicData for ITS sectors only; Allgather the combine-relevant
// arrays (temperatures, energy, specific_heat, entropy, free_energy -- the
// fields ed::core::combine_sector_thermodynamics reads) so every rank ends with
// the FULL per-sector list and computes an identical combined result. gs_E is
// min-reduced. No-op when single-rank. Returns the gathered full list in-place.
inline void mpi_allgather_sector_thermo(
    std::vector<ThermodynamicData>&  per_sector_thermo,
    std::vector<std::uint64_t>&      per_sector_dims,
    const std::vector<std::uint64_t>& raw_indices,  // parallel to per_sector_thermo
    double&                          gs_E,
    int                              mpi_size) {
    if (mpi_size <= 1) return;

    // Agree on the temperature-grid length (a rank that owns no sector has 0).
    int nT = per_sector_thermo.empty()
                 ? 0 : static_cast<int>(per_sector_thermo.front().temperatures.size());
    int nT_global = nT;
    MPI_Allreduce(&nT, &nT_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (nT_global <= 0) {  // no rank produced any thermo
        double gmin = gs_E;
        MPI_Allreduce(&gs_E, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        gs_E = gmin;
        return;
    }

    // Flatten local sectors: [raw_index, temps, energy, Cv, S, F] = 1 + 5*nT
    // doubles each. The raw index travels with the block so the gathered list
    // can be re-sorted into the canonical (single-node) order -- otherwise the
    // F-based combine sums sectors in rank order and the result differs by FP
    // rounding from the single-rank run.
    const int per_block = 1 + 5 * nT_global;
    std::vector<double> send;
    send.reserve(per_sector_thermo.size() * static_cast<std::size_t>(per_block));
    auto put = [&](const std::vector<double>& v) {
        for (int t = 0; t < nT_global; ++t)
            send.push_back(t < static_cast<int>(v.size()) ? v[static_cast<std::size_t>(t)] : 0.0);
    };
    for (std::size_t s = 0; s < per_sector_thermo.size(); ++s) {
        send.push_back(static_cast<double>(s < raw_indices.size() ? raw_indices[s] : s));
        const auto& th = per_sector_thermo[s];
        put(th.temperatures); put(th.energy); put(th.specific_heat);
        put(th.entropy);      put(th.free_energy);
    }

    const int sendcount = static_cast<int>(send.size());
    std::vector<int> counts(static_cast<std::size_t>(mpi_size));
    MPI_Allgather(&sendcount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displs(static_cast<std::size_t>(mpi_size));
    int total = 0;
    for (int i = 0; i < mpi_size; ++i) { displs[i] = total; total += counts[i]; }
    std::vector<double> recv(static_cast<std::size_t>(total));
    MPI_Allgatherv(send.data(), sendcount, MPI_DOUBLE,
                   recv.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Sort the gathered blocks by raw sector index (canonical order == the
    // single-node sector loop order) so the combine is bit-identical.
    std::vector<int> block_off;
    for (int off = 0; off + per_block <= total; off += per_block) block_off.push_back(off);
    std::sort(block_off.begin(), block_off.end(),
              [&](int a, int b) { return recv[a] < recv[b]; });

    per_sector_thermo.clear();
    per_sector_dims.clear();
    for (int off : block_off) {
        ThermodynamicData th;
        auto take = [&](int slot) {  // slot 0 is raw_index; arrays start at 1
            const int base = off + 1 + slot * nT_global;
            return std::vector<double>(recv.begin() + base, recv.begin() + base + nT_global);
        };
        th.temperatures  = take(0);
        th.energy        = take(1);
        th.specific_heat = take(2);
        th.entropy       = take(3);
        th.free_energy   = take(4);
        per_sector_thermo.push_back(std::move(th));
        per_sector_dims.push_back(1);   // dims are unused by the F-based combine
    }

    double gmin = gs_E;
    MPI_Allreduce(&gs_E, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    gs_E = gmin;
}
#endif  // WITH_MPI

// Pybind11 cannot move a captured `std::unique_ptr<LinearOperator>` out of
// a Python-owned Operator easily; we instead accept the raw `Operator&` /
// `FixedSzOperator&` and rely on the fact that both inherit from
// `ed::LinearOperator`. The orchestrator calls take `const LinearOperator&`,
// so a simple reference upcast is enough.

// ---------------------------------------------------------------------------
// GPU lane (operator-collapse Phase 2a, Jun 2026).
//
// ``qed.solve/thermal/spectral(device='gpu')`` flips
// ``opts.backend.allow_gpu = true`` in the Python wrappers. The plain
// ``ed::Operator`` / ``ed::FixedSzOperator`` now advertise
// ``geometry().supports_device_matvec=true`` (WITH_CUDA), and their
// ``bind_cuda()`` lazily builds a ``CudaMatVecBackend`` device mirror (the
// SOTA no-atomic gather kernel). So ``ed::select_backend`` picks the
// ``CudaBackend`` lane straight off the host operator's capability flag --
// no bespoke ``GPUOperator`` / ``GPUFixedSzOperator`` promotion needed
// (that path is retired here; the legacy classes go away in Phase 2b).
//
// The streaming-symmetry directory binding is likewise capability-driven:
// the per-sector operators advertise the flag and wire their own lazy GPU
// mirror.
//
// What remains below: a runtime probe + a Python ``RuntimeWarning`` for the
// few lanes that genuinely have no GPU implementation (FullDiag for solve;
// any future host-only thermal/spectral method), so a ``device='gpu'``
// request that silently runs on CPU is visible at the call site.
// ---------------------------------------------------------------------------
inline bool gpu_runtime_available() noexcept {
#ifdef WITH_CUDA
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
#else
    return false;
#endif
}

/// FullDiag in ``solve_on<Backend>`` builds the dense matrix by calling
/// the bound matvec with HOST ``std::vector<Complex>`` storage and runs
/// the (host-only) LAPACK dense eigensolver; it explicitly pins the
/// column-build to ``bind_cpu()``. FullDiag is the auto-selected method
/// for ``global_dim <= 2^12 = 4096`` (see ``auto_solve_method`` in
/// orchestrator.cpp). At those dimensions the matvec is negligible
/// compared to the O(N^3) LAPACK solve and the GPU lane offers no
/// measurable win, so the binding pins ``allow_gpu=false`` for FullDiag
/// to avoid spinning up an unused CudaBackend (and a misleading "gpu"
/// lane label).
using ed::workflows::will_use_full_diag;  // hoisted (audit 2026-07-31)

// -----------------------------------------------------------------------
// Stage 12 (SU(2) rollout) shared helpers.
// -----------------------------------------------------------------------

/// Term-level SU(2) detection on an in-memory carrier operator.
using ed::workflows::op_is_su2_symmetric;  // hoisted (audit 2026-07-31)

/// Resolve the su2 engagement for a solve call: returns true when the
/// SU(2) machinery (targeting / labeling) should run. Throws when a
/// hard request (targeting, or label_total_spin == 1) cannot be met.
using ed::workflows::resolve_su2_engagement;  // hoisted

/// S^2 restricted to the SAME basis as `op` (dynamic-type probe):
/// FixedSzOperator -> a FixedSz twin sharing n_up; plain Operator ->
/// the full-space carrier. Symmetry sectors go through
/// `make_rep_sector_matvec` instead (rep-basis restriction).
inline std::shared_ptr<const ed::matvec::MatVecOperator>
make_s2_like(const Operator& op) {
    const std::uint64_t n = op.getNumBits();
    if (const auto* fsz = dynamic_cast<const FixedSzOperator*>(&op)) {
        auto s2 = std::make_shared<FixedSzOperator>(
            n, 0.5f, fsz->producer().n_up());
        s2->copyTermsFrom(*ed::ops::make_S2_carrier(n));
        return s2;
    }
    return ed::ops::make_S2_carrier(n);
}

/// Label a batch of host eigenvectors with certified two_S / raw <S^2>.
/// Appends one entry per vector to the result arrays (two_S = -1 when
/// certification fails).
inline void label_vectors_with_s2(
    const ed::matvec::MatVecOperator& s2,
    std::vector<std::vector<Complex>>& vecs,
    int n_sites, int n_up, int flip_parity,
    std::vector<double>& s2_out, std::vector<int>& two_S_out,
    const std::vector<double>* eigenvalues = nullptr) {
    // Correctness (2026-09-11): within a DEGENERATE multiplet the solver's
    // vectors are arbitrary mixtures of different total-spin components, so
    // the certification failed (e.g. E = 0 on the 4-site ring mixes S = 0 and
    // S = 1). Diagonalise S^2 inside each degenerate group first and rotate
    // the vectors -- they stay eigenvectors of H and become S^2 eigenstates.
    if (eigenvalues && eigenvalues->size() == vecs.size() && !vecs.empty()) {
        const std::size_t n = vecs.size();
        std::size_t i = 0;
        while (i < n) {
            std::size_t j = i + 1;
            const double e0 = (*eigenvalues)[i];
            while (j < n && std::abs((*eigenvalues)[j] - e0) <= 1e-8 * (1.0 + std::abs(e0))) ++j;
            const std::size_t g = j - i;
            const std::size_t dim = vecs[i].size();
            if (g > 1 && dim == s2.dim()) {
                std::vector<std::vector<Complex>> s2v(g, std::vector<Complex>(dim));
                for (std::size_t a = 0; a < g; ++a) s2.apply(vecs[i + a].data(), s2v[a].data(), dim);
                Eigen::MatrixXcd M(g, g);
                for (std::size_t a = 0; a < g; ++a)
                    for (std::size_t b = 0; b < g; ++b) {
                        Complex acc{0.0, 0.0};
                        for (std::size_t r = 0; r < dim; ++r) acc += std::conj(vecs[i + a][r]) * s2v[b][r];
                        M(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = acc;
                    }
                M = 0.5 * (M + M.adjoint().eval());
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
                if (es.info() == Eigen::Success) {
                    const Eigen::MatrixXcd U = es.eigenvectors();   // columns: S^2 eigenstates
                    std::vector<std::vector<Complex>> rot(g, std::vector<Complex>(dim, Complex{0.0, 0.0}));
                    for (std::size_t c = 0; c < g; ++c)
                        for (std::size_t a = 0; a < g; ++a) {
                            const Complex u = U(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(c));
                            if (std::abs(u) < 1e-15) continue;
                            for (std::size_t r = 0; r < dim; ++r) rot[c][r] += u * vecs[i + a][r];
                        }
                    for (std::size_t c = 0; c < g; ++c) vecs[i + c] = std::move(rot[c]);
                }
            }
            i = j;
        }
    }
    for (const auto& v : vecs) {
        if (v.size() != s2.dim()) {
            s2_out.push_back(-1.0);
            two_S_out.push_back(-1);
            continue;
        }
        double res = 0.0;
        const double s2_exp =
            ed::ops::s2_expectation(s2, v.data(), v.size(), &res);
        s2_out.push_back(s2_exp);
        two_S_out.push_back(
            res <= ed::ops::kS2CertifyTol
                ? ed::ops::snap_two_S(s2_exp, n_sites, n_up, flip_parity)
                : -1);
    }
}

/// Exact canonical thermodynamics of a (small) spectrum on an explicit
/// temperature grid. Twin of the orchestrator's static
/// ``compute_canonical_thermo_from_eigs`` (kept file-local there);
/// used by the tower binding's exact differencing route so every tower
/// lands on the SAME grid the recombiner expects.

/// Build the Lowdin projector + wrapped operator for one solve block.
/// Returns {wrapped_operator, projector}; the caller installs
/// ``seed_transform`` from the projector and solves the wrapper.
using ed::workflows::Su2Targeting;        // hoisted
using ed::workflows::make_su2_targeting;  // hoisted

/// Shared solve-one-block routine for the SU(2) targeting lane: installs
/// the projected seed, forces the Krylov lane (FullDiag has no seed and
/// would return every tower), and maps the "zero seed" refusal (empty
/// tower in this block) to an empty result.
using ed::workflows::solve_su2_targeted;  // hoisted

/// The thermal lane has uneven GPU coverage:
///   * FTLM         : CPU only (orchestrator throws on CUDA).
///   * LTLM, KpmDos : CPU or CUDA.
///   * mTPQ         : any backend.
///
/// As of Phase E of the "Close CPU/GPU Gaps" plan (May 2026), the
/// FTLM facade dispatches on Backend internally (see
/// ``include/ed/thermal/ftlm_kernel.h``) and accepts both ``CpuBackend``
/// and ``CudaBackend``, so it joins the GPU-eligible set.
inline bool thermal_method_supports_gpu(
    ed::workflows::ThermalOptions::Method m) noexcept {
    using M = ed::workflows::ThermalOptions::Method;
    return m == M::FTLM
        || m == M::LTLM
        || m == M::KpmDos
        || m == M::mTPQ;
}

/// Spectral lanes after Phases F + G of the "Close CPU/GPU Gaps"
/// plan (May 2026): every method now dispatches on Backend
/// internally, so the entire spectral lane is GPU-eligible.
///   * GroundStateCF : runs the inner solve + CF kernel through
///                     ``H.template bind<B>()``.
///   * FtlmDynamical : routes through
///                     ``detail::ftlm_dynamical_kernel_via_backend``.
///   * KpmDynamical  : routes through
///                     ``detail::kpm_dynamical_kernel_via_backend``.
inline bool spectral_method_supports_gpu(
    ed::workflows::SpectralOptions::Method /*m*/) noexcept {
    return true;
}

/// Emit a Python ``RuntimeWarning`` so the caller sees the silent
/// demotion ``device='gpu' -> CPU lane`` instead of finding out through
/// a profiler. The warning fires only when the GPU was actually
/// reachable (``allow_gpu=true`` AND ``gpu_runtime_available()``) -- if
/// the build is CPU-only or no NVIDIA device is visible there is no
/// "demotion" to report. Uses ``stacklevel=2`` so the warning blame
/// points at the user's ``qed.solve / qed.thermal / qed.spectral``
/// call site rather than at this binding.
inline void warn_silent_cpu_fallback(const char* what,
                                     const ed::BackendConstraints& c) {
    if (!c.allow_gpu) return;
#ifdef WITH_CUDA
    if (!gpu_runtime_available()) return;
#else
    return;
#endif
    try {
        py::module_::import("warnings").attr("warn")(
            std::string(what)
                + " requested device='gpu' but the chosen method has no GPU "
                  "implementation in the orchestrator. Falling back to the "
                  "CPU lane. Pass device='cpu' to silence this warning, or "
                  "switch to a GPU-clean method (Lanczos/BlockLanczos/"
                  "KrylovSchur for solve; LTLM/KPM_DOS/mTPQ for "
                  "thermal; GroundStateCF for spectral).",
            py::module_::import("builtins").attr("RuntimeWarning"),
            py::arg("stacklevel") = 2);
    } catch (const py::error_already_set&) {
        // Best-effort -- never let the warning machinery break the
        // workflow call. The caller still gets the correct CPU result.
    }
}


// ---------------------------------------------------------------------------
// Stage 10d: shared steps of the three cross-irrep spectral bindings
// (GS-CF, multi-Q, FTLM). These were byte-identical blocks restated per
// binding -- the A3 fail-safe had to be fixed TWICE because the GS scan
// existed twice. One definition each; the bindings keep only their
// genuinely different flows.
// ---------------------------------------------------------------------------

// Source-operator spec shared by all three cross-irrep bindings.
// Takes the ALREADY-DECODED optional: every caller runs this inside a
// ``py::gil_scoped_release`` block, and reading a ``py::object`` there
// (``is_none`` / ``cast<int>``) touches the interpreter without the GIL --
// a segfault in ``PyErr_Occurred`` on Python 3.11+ (CI, 2026-09-11) that the
// workstation build happened to survive.
[[nodiscard]] inline std::optional<int> decode_optional_n_up(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    return o.cast<int>();
}

// ---------------------------------------------------------------------------
// WP9: the symmetric source a streaming-symmetry binding body runs on -- the
// writer's directory (``*_directory`` bindings) or the same content held in
// memory (their in-memory twins). Both alternatives are cheap to copy, so a
// body can seed several specs (source + shifted-Sz target) from one source.
// ---------------------------------------------------------------------------
using SymmetricSource = std::variant<ed::DirectoryPath, ed::InMemorySymmetric>;

inline void set_symmetric_source(ed::OperatorSpec& spec,
                                 const SymmetricSource& source) {
    std::visit([&spec](const auto& s) { spec.source = s; }, source);
}

// In-memory twin input: a private copy of ``H``'s terms (the result never
// aliases the Python-owned operator) plus the group the directory writer
// would have serialised, rebuilt with the writer's phase convention
// (``SymmetryGroupInfo::from_memory``). ``group`` is the Python info dict
// ``_write_symmetry_directory`` consumes; only ``max_clique``,
// ``generators``, ``generator_orders`` and each sector's ``sector_id`` /
// ``quantum_numbers`` are read (the writer recomputes the phases from the
// quantum numbers too). Must run under the GIL.
[[nodiscard]] inline ed::InMemorySymmetric
in_memory_symmetric_source(const Operator& H, const py::dict& group,
                           const char* who) {
    auto entry = [&](const char* key) -> py::object {
        if (!group.contains(key)) {
            throw std::invalid_argument(
                std::string(who) + ": group dict has no '" + key
                + "' entry.");
        }
        return group[key];
    };
    auto max_clique = entry("max_clique").cast<std::vector<std::vector<int>>>();
    auto generators = entry("generators").cast<std::vector<std::vector<int>>>();
    auto orders     = entry("generator_orders").cast<std::vector<int>>();
    std::vector<std::pair<std::uint64_t, std::vector<int>>> sectors;
    for (py::handle h : entry("sectors")) {
        py::object s = py::reinterpret_borrow<py::object>(h);
        sectors.emplace_back(
            s.attr("get")("sector_id", 0).cast<std::uint64_t>(),
            s.attr("get")("quantum_numbers", py::list())
                .cast<std::vector<int>>());
    }
    auto op = std::make_shared<Operator>(H.getNumBits(), H.getSpin());
    op->copyTermsFrom(H);
    return ed::InMemorySymmetric{
        std::move(op),
        std::make_shared<const SymmetryGroupInfo>(
            SymmetryGroupInfo::from_memory(std::move(max_clique),
                                           std::move(generators),
                                           std::move(orders), sectors))};
}

[[nodiscard]] inline ed::OperatorSpec make_cross_irrep_src_spec(
    const SymmetricSource& source, std::uint64_t num_sites, double spin_l,
    std::optional<int> fixed_sz_n_up, int sz_parity, bool flip_sectors)
{
    ed::OperatorSpec spec;
    set_symmetric_source(spec, source);
    spec.num_sites          = num_sites;
    spec.spin_l             = static_cast<float>(spin_l);
    spec.streaming_symmetry = true;
    if (fixed_sz_n_up.has_value()) {
        spec.fixed_sz = *fixed_sz_n_up;
    } else {
        // Stage 8d: Sz-parity halves + full-space prod-sigma^x flip
        // sectors (the factory validates the closure rules; the probe's
        // slot routing is the caller's).
        if (sz_parity >= 0) spec.sz_parity = sz_parity;
        if (flip_sectors)   spec.flip_sectors_full = true;
    }
    return spec;
}

// Two-phase global-GS scan across the source sectors (Wave B2 + the A3
// fail-safes): cheap Phase-1 (krylov 40) everywhere, refine candidates
// within `gap` of the best Phase-1 minimum; any non-finite Phase-1
// estimate fails safe to refine-everything, and a throwing Phase-2
// candidate is skipped, never fatal.
struct GsScanResult {
    std::size_t gs_idx    = 0;
    double      gs_energy = std::numeric_limits<double>::infinity();
    bool        any_solved = false;
};

// Residual of a cross-irrep ground-state pair (Jul 2026; mirrors
// little_group_ground_state): the orchestrator returns a BEST-EFFORT pair
// when the Krylov iteration does not converge, and everything downstream --
// the scattered weights and the continued fraction -- would silently
// inherit the garbage. One extra matvec at sector dim measures it.
template <class SectorView>
[[nodiscard]] inline double gs_residual(SectorView& sec,
                                        const std::vector<Complex>& psi0,
                                        double E0)
{
    const std::size_t d = psi0.size();
    std::vector<Complex> hv(d, Complex(0.0, 0.0));
    sec.apply(psi0.data(), hv.data(), d);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        const Complex r = hv[i] - E0 * psi0[i];
        num += std::norm(r);
        den += std::norm(psi0[i]);
    }
    return (den > 0.0) ? std::sqrt(num / den)
                       : std::numeric_limits<double>::infinity();
}

// Refine-or-die (Jul 2026): the orchestrator's vector path has no stored-
// basis reorthogonalisation, so its Ritz vector carries ~1e-4 residual even
// on tiny sectors -- the guard alone dead-ended every cross-irrep spectral
// workflow (caught at 4x4 by the e2e diagnostic). When the first pair fails
// the bound, rerun a FullCGS2 Lanczos WITH a kept basis over the same
// sector matvec, SEEDED by the failed vector (the little_group
// solve_gs_vector construction), and re-guard. Still throws if even the
// reorthogonalised pair cannot meet 1e-8.
template <class SectorView>
inline void ensure_gs_residual(SectorView& sec,
                               std::vector<Complex>& psi0,
                               double& E0)
{
    double resid = gs_residual(sec, psi0, E0);
    if (resid < 1e-8) return;

    const std::size_t n = psi0.size();
    ed::matvec::CpuBackend be;
    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter   = std::min<std::size_t>(n, 300);
    kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
    kopts.keep_basis = true;
    kopts.dim_cap    = n;
    auto apply_H = [&sec](const Complex* in, Complex* out, std::size_t nn) {
        sec.apply(in, out, nn);
    };
    auto kres = ed::krylov::lanczos_kernel(be, apply_H, n, psi0.data(),
                                           kopts);
    const std::size_t m = kres.alpha.size();
    if (m == 0)
        throw std::runtime_error(
            "cross-irrep spectral: CGS2 GS refinement produced an empty "
            "tridiagonal.");
    std::vector<double> diag = kres.alpha;
    std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
    for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
    std::vector<double> z(m * m, 0.0);
    const lapack_int info = LAPACKE_dstevd(
        LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
        diag.data(), off.data(), z.data(), static_cast<lapack_int>(m));
    if (info != 0)
        throw std::runtime_error(
            "cross-irrep spectral: GS refinement tridiag eigensolve failed "
            "(dstevd info != 0).");
    E0 = diag[0];
    std::fill(psi0.begin(), psi0.end(), Complex(0.0, 0.0));
    for (std::size_t j = 0; j < m; ++j) {
        const Complex* vj = kres.basis[j].get();
        const double   yj = z[j];                    // column 0, row j
        if (std::abs(yj) < 1e-300) continue;
        for (std::size_t i = 0; i < n; ++i) psi0[i] += yj * vj[i];
    }
    double nrm = 1e-300;
    for (const auto& c : psi0) nrm += std::norm(c);
    const double inv = 1.0 / std::sqrt(nrm);
    for (auto& c : psi0) c *= inv;

    resid = gs_residual(sec, psi0, E0);
    if (!(resid < 1e-8)) {
        throw std::runtime_error(
            "cross-irrep spectral: ground-state pair failed the residual "
            "guard even after CGS2 refinement (|H psi - E psi|/|psi| = "
            + std::to_string(resid) + ").");
    }
}

[[nodiscard]] inline GsScanResult find_gs_sector_two_phase(
    ed::core::SectorSetView&              handle,
    const std::vector<std::size_t>&       sector_indices,
    const ed::workflows::SpectralOptions& opts)
{
    GsScanResult out;
    const bool enable_two_phase = sector_indices.size() > 2;
    std::vector<std::size_t> phase2_candidates;
    if (enable_two_phase) {
        std::vector<std::pair<double, std::size_t>> phase1_min;
        phase1_min.reserve(sector_indices.size());
        for (std::size_t k : sector_indices) {
            auto sec = handle.sector(k);
            if (!sec || sec->dim() == 0) continue;
            ed::workflows::SolveOptions p1;
            p1.num_eigs        = 1;
            p1.tolerance       = 1e-8;
            p1.backend         = opts.backend;
            p1.method          = ed::workflows::SolveMethod::Lanczos;
            p1.compute_vectors = false;
            p1.max_iter        = std::min<std::size_t>(40, sec->dim());
            try {
                auto sr = ed::workflows::solve(*sec, p1);
                if (!sr.eigenvalues.empty())
                    phase1_min.emplace_back(sr.eigenvalues.front(), k);
            } catch (...) {
                phase1_min.emplace_back(
                    -std::numeric_limits<double>::infinity(), k);
            }
        }
        if (!phase1_min.empty()) {
            std::sort(phase1_min.begin(), phase1_min.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });
            const bool phase1_nonfinite = std::any_of(
                phase1_min.begin(), phase1_min.end(),
                [](const auto& q) { return !std::isfinite(q.first); });
            if (phase1_nonfinite) {
                for (const auto& [E, k] : phase1_min) {
                    (void)E;
                    phase2_candidates.push_back(k);
                }
            } else {
                const double best_E = phase1_min.front().first;
                const double gap =
                    std::max(1e-2 * std::abs(best_E), 1e-4);
                for (const auto& [E, k] : phase1_min)
                    if (E <= best_E + gap) phase2_candidates.push_back(k);
            }
        }
    }
    const std::vector<std::size_t>& scan =
        enable_two_phase ? phase2_candidates : sector_indices;
    for (std::size_t k : scan) {
        auto sec = handle.sector(k);
        if (!sec || sec->dim() == 0) continue;
        ed::workflows::SolveOptions sopts;
        sopts.num_eigs        = 1;
        sopts.tolerance       = 1e-12;
        sopts.backend         = opts.backend;
        sopts.method          = ed::workflows::SolveMethod::Lanczos;
        sopts.compute_vectors = false;
        // Phase-2 refinement decides WHICH sector holds the global GS, so
        // its energies must be trusted at the inter-sector spacing. The
        // auto-tuned iteration budget resolved to ~32 here, which left
        // per-sector E0 estimates unconverged at the ~1e-3 level and was
        // observed (2026-07-18, kagome 2x3, sector spacing 4.3e-3) to pick
        // a WRONG sector non-deterministically. 300 iterations converges
        // a single extremal eigenvalue at every dim this scan sees.
        sopts.max_iter        = std::min<std::size_t>(sec->dim(), 300);
        ed::GroundStateResult sr;
        try {
            sr = ed::workflows::solve(*sec, sopts);
        } catch (...) { continue; }
        if (sr.eigenvalues.empty()) continue;
        if (sr.eigenvalues.front() < out.gs_energy) {
            out.gs_energy = sr.eigenvalues.front();
            out.gs_idx    = k;
        }
        out.any_solved = true;
    }
    return out;
}
}  // namespace workflow_bindings_detail

// ---------------------------------------------------------------------------
// Per-area registrars. `bind_workflows` (workflow_bindings.cpp) calls these
// in exactly this order -- the registration order decides the pybind11
// signature text of every later binding, so it is part of the surface.
// ---------------------------------------------------------------------------
void bind_workflows_types(py::module_& m);
void bind_workflows_symmetry(py::module_& m);
void bind_workflows_solve(py::module_& m);
void bind_workflows_thermal(py::module_& m);
void bind_workflows_spectral(py::module_& m);
void bind_workflows_solve_streaming(py::module_& m);
void bind_workflows_thermal_streaming(py::module_& m);
void bind_workflows_spectral_streaming(py::module_& m);
void bind_workflows_spectral_cross_irrep(py::module_& m);
void bind_workflows_spectral_multiq(py::module_& m);
void bind_workflows_spectral_ftlm(py::module_& m);
void bind_workflows_thermal_all_sz(py::module_& m);
