// =============================================================================
// src/symmetry/streaming_symmetry_gpu_mirror.cu
//
// Phase A of the "Backend x Symmetries x Workflows: close the full
// 48-cell matrix" plan (May 2026).
//
// Lazy per-sector GPU mirror for the SectorView::bind_cuda() path.
// Compiled into ed_solvers_gpu (so we have nvcc + thrust available);
// the matching ed_core .cpp file ``streaming_symmetry_gpu_mirror.cpp``
// only contributes the throwing stub when WITH_CUDA is OFF.
//
// On first bind, builds a fully device-resident snapshot of:
//
//   * Orbit CSR (orbit_elements, orbit_coefficients, orbit_offsets,
//     orbit_norms)
//   * State -> (basis_idx, pre-baked projection) hash table
//     (DeviceSymmetryHashEntry layout from device_basis_policy.cuh)
//   * Term SoA (DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
//     OffDiagTwoBody, ThreeBodyTerm) -- the same POD structs the host
//     uses, copied byte-for-byte
//
// The returned MatvecFn takes DEVICE pointers (per the bind_cuda()
// contract from <ed/core/linear_operator.h>) and dispatches the
// unified ``apply_terms_gpu_scatter<DeviceSymmetryBasisPolicy,
// cuDoubleComplex>`` kernel for both the StreamingSymmetryOperator
// (full Hilbert + symmetry, cell 3B) and the FixedSz variant (cell 4B).
//
// Validation: an end-to-end C++ test at
// tests/unit/test_streaming_symmetry_gpu_mirror.cpp asserts the GPU
// matvec result matches the CPU ``applySymmetrized`` to 1e-10 on a
// Heisenberg ring with Z_N translation symmetry.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/symmetry/symmetry_sector_data.h>
#include <ed/matvec/device_basis_policy.cuh>
#include <ed/matvec/term_kernels_gpu.cuh>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/sector_gpu_mirror.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <cmath>
#include <utility>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::symmetry::gpu_mirror {

namespace detail {
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("StreamingSymmetry GPU mirror: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}
}  // namespace detail

// (Stage 11c-2b: the legacy orbit-CSR device mirror -- GpuSectorMirror, its
// build_mirror, launch_symmetry_matvec, and the ED_GPU_SYMMETRY_MIRROR_V2
// tuning helpers -- was deleted together with make_sector_matvec_gpu and the
// ED_GPU_SYMMETRY_REP escape. The on-the-fly representative mirror below is
// THE device representation.)

// =============================================================================
// GpuRepSectorMirror -- on-the-fly representative SpMV device snapshot.
//
// "On-the-fly representative SpMV for streaming symmetry" plan (Jun 2026).
//
// Holds NO orbit CSR and NO O(full-Sz-dim) projection table. The
// resident footprint is:
//   * reps (dim x 8 B) + inv_norms (dim x 8 B)
//   * the |G| site permutations (group_size * n_sites ints)
//   * the per-sector character array (group_size complex)
//   * rep_index_of_rank: C(n_sites, n_up) int32 reverse lookup keyed by the
//     combinadic rank of an orbit REPRESENTATIVE (built from ``reps`` alone,
//     no orbit walk).
// The group action + projection are regenerated arithmetically inside the
// kernel; per-SpMV traffic is just the in/out vectors -> the genuine /|G|.
// =============================================================================
// Stage-4 device twin (Jul 2026): ONE rank -> shared-rep-index table per
// (N, n_up) subspace, co-owned by every irrep sector's mirror through a
// content-keyed weak registry. Kills the per-sector C(N, n_up) x int32
// duplication (2.4 GiB EACH at N=32 half filling).
struct GpuSharedRankTable {
    thrust::device_vector<std::int32_t> d_shared_of_rank;
};

[[nodiscard]] inline std::shared_ptr<GpuSharedRankTable>
acquire_gpu_shared_rank(
    const std::shared_ptr<const ed::symmetry::SharedRankLookup>& srl)
{
    static std::mutex mtx;
    static std::map<const void*, std::weak_ptr<GpuSharedRankTable>> registry;
    // Keep-alive FIFO: per-sector GPU mirrors are transient (rebuilt per
    // solve), so a pure weak registry would re-upload the table between
    // consecutive sector solves. A run touches at most a couple of
    // (N, n_up) subspaces, so a tiny strong cache pins the recent tables.
    static std::vector<std::pair<const void*,
                                 std::shared_ptr<GpuSharedRankTable>>> keep;
    // Jul 2026: BYTE-aware eviction. A count cap of 4 pinned up to 4 x 36 GB
    // at N >= 34 half filling -- guaranteed device OOM the moment a job
    // touched two subspaces. ED_GPU_SYM_CACHE_GIB (default 24) bounds the
    // strong cache; the weak registry still dedups concurrent co-owners.
    static const double kBudgetBytes = [] {
        double gib = 24.0;
        if (const char* v = std::getenv("ED_GPU_SYM_CACHE_GIB")) {
            const double parsed = std::atof(v);
            if (parsed > 0.0) gib = parsed;
        }
        return gib * 1073741824.0;
    }();

    std::lock_guard<std::mutex> lk(mtx);
    auto& slot = registry[static_cast<const void*>(srl.get())];
    if (auto sp = slot.lock()) return sp;
    auto sp = std::make_shared<GpuSharedRankTable>();
    sp->d_shared_of_rank = srl->shared_of_rank;   // one H2D per (N, n_up)
    if (std::getenv("ED_SYM_PROFILE") != nullptr) {
        std::fprintf(stderr,
                     "[sym_profile] GPU shared rank table uploaded: "
                     "%zu entries (N=%d, n_up=%d), co-owned by mirrors\n",
                     srl->shared_of_rank.size(), srl->n_sites, srl->n_up);
    }
    slot = sp;
    keep.emplace_back(static_cast<const void*>(srl.get()), sp);
    auto bytes_of = [](const std::shared_ptr<GpuSharedRankTable>& t) {
        return static_cast<double>(t->d_shared_of_rank.size())
             * sizeof(std::int32_t);
    };
    double total = 0.0;
    for (const auto& kv : keep) total += bytes_of(kv.second);
    while (keep.size() > 1 && total > kBudgetBytes) {
        total -= bytes_of(keep.front().second);
        keep.erase(keep.begin());
    }
    return sp;
}

struct GpuRepSectorMirror {
    thrust::device_vector<std::uint64_t>   d_reps;
    thrust::device_vector<double>          d_inv_norms;
    thrust::device_vector<int>             d_perms;
    thrust::device_vector<cuDoubleComplex> d_characters;
    thrust::device_vector<std::uint64_t>   d_flips;   // Stage 8b: flip masks
    thrust::device_vector<std::uint64_t>   d_perm_lut; // byte-LUT fast path
    int                                     perm_lut_bpw = 0;
    thrust::device_vector<std::int32_t>    d_rep_index_of_rank;
    // Stage-4 device twin: shared table (co-owned) + per-sector remap.
    std::shared_ptr<GpuSharedRankTable>    shared_rank_tab;
    thrust::device_vector<std::int32_t>    d_local_of_shared;

    int           group_size = 1;
    int           n_sites    = 0;
    int           n_up       = -1;
    std::uint64_t dim        = 0;
    double        spin_l     = 0.5;

    thrust::device_vector<ed::matvec::DiagOneBody>     d_diag_one_body;
    thrust::device_vector<ed::matvec::OffDiagOneBody>  d_offdiag_one_body;
    thrust::device_vector<ed::matvec::DiagTwoBody>     d_diag_two_body;
    thrust::device_vector<ed::matvec::MixedTwoBody>    d_mixed_two_body;
    thrust::device_vector<ed::matvec::OffDiagTwoBody>  d_offdiag_two_body;
    thrust::device_vector<ed::matvec::ThreeBodyTerm>   d_three_body;

    ed::matvec::basis::DeviceRepSymmetryBasisPolicy basis_view() const noexcept {
        ed::matvec::basis::DeviceRepSymmetryBasisPolicy v;
        v.reps              = thrust::raw_pointer_cast(d_reps.data());
        v.inv_norms         = thrust::raw_pointer_cast(d_inv_norms.data());
        v.perms             = thrust::raw_pointer_cast(d_perms.data());
        v.characters        = thrust::raw_pointer_cast(d_characters.data());
        v.flips             = d_flips.empty()
            ? nullptr : thrust::raw_pointer_cast(d_flips.data());
        v.perm_lut          = d_perm_lut.empty()
            ? nullptr : thrust::raw_pointer_cast(d_perm_lut.data());
        v.perm_lut_bpw      = perm_lut_bpw;
        v.rep_index_of_rank = d_rep_index_of_rank.empty()
            ? nullptr : thrust::raw_pointer_cast(d_rep_index_of_rank.data());
        if (shared_rank_tab && !d_local_of_shared.empty()) {
            v.shared_rank_of  = thrust::raw_pointer_cast(
                shared_rank_tab->d_shared_of_rank.data());
            v.local_of_shared = thrust::raw_pointer_cast(
                d_local_of_shared.data());
        }
        v.dim_              = dim;
        v.group_size        = group_size;
        v.n_sites           = n_sites;
        v.n_up              = n_up;
        return v;
    }

    ed::matvec::kernel::gpu::DeviceTermStorage terms_view() const noexcept {
        ed::matvec::kernel::gpu::DeviceTermStorage t;
        t.diag_one_body        = thrust::raw_pointer_cast(d_diag_one_body.data());
        t.num_diag_one_body    = static_cast<std::uint32_t>(d_diag_one_body.size());
        t.offdiag_one_body     = thrust::raw_pointer_cast(d_offdiag_one_body.data());
        t.num_offdiag_one_body = static_cast<std::uint32_t>(d_offdiag_one_body.size());
        t.diag_two_body        = thrust::raw_pointer_cast(d_diag_two_body.data());
        t.num_diag_two_body    = static_cast<std::uint32_t>(d_diag_two_body.size());
        t.mixed_two_body       = thrust::raw_pointer_cast(d_mixed_two_body.data());
        t.num_mixed_two_body   = static_cast<std::uint32_t>(d_mixed_two_body.size());
        t.offdiag_two_body     = thrust::raw_pointer_cast(d_offdiag_two_body.data());
        t.num_offdiag_two_body = static_cast<std::uint32_t>(d_offdiag_two_body.size());
        t.three_body           = thrust::raw_pointer_cast(d_three_body.data());
        t.num_three_body       = static_cast<std::uint32_t>(d_three_body.size());
        return t;
    }
};

namespace detail {

// Build a GpuRepSectorMirror from a CSR-free RepSectorData + term storage.
// Builds the reverse rank table from ``reps`` only (no orbit walk).
inline std::shared_ptr<GpuRepSectorMirror>
build_rep_mirror(const ed::symmetry::RepSectorData& data,
                 double spin_l,
                 const ed::matvec::TermStorage& terms)
{
    if (!data.usable()) {
        throw std::runtime_error(
            "build_rep_mirror: RepSectorData is not usable (need n_up >= 0, "
            "non-empty reps, and matching characters / perms sizes)");
    }
    const int n_sites = data.n_sites;
    const int n_up    = data.n_up;
    if (n_sites <= 0 || n_sites > 64 || n_up < -1 || n_up > n_sites) {
        throw std::runtime_error("build_rep_mirror: invalid n_sites / n_up");
    }
    if (n_up < 0 && n_sites > 31) {
        // Full-space sector: the reverse table is indexed by the state
        // itself (2^N int32 entries). 31 bits caps it at 8 GiB.
        throw std::runtime_error(
            "build_rep_mirror: full-space rep mirror needs n_sites <= 31 "
            "(dense state-indexed reverse table)");
    }

    auto mirror = std::make_shared<GpuRepSectorMirror>();
    mirror->spin_l     = spin_l;
    mirror->group_size = data.group_size;
    mirror->n_sites    = n_sites;
    mirror->n_up       = n_up;
    mirror->dim        = data.dim();

    // C(n_sites, n_up), capped at INT32_MAX (the rank-table value type).
    // Full-space sectors (n_up < 0): the rank space is the whole 2^N
    // (state-indexed identity rank).
    long double dv = 1.0L;
    if (n_up >= 0) {
        int kk = (n_up < n_sites - n_up) ? n_up : (n_sites - n_up);
        for (int i = 0; i < kk; ++i) {
            dv *= static_cast<long double>(n_sites - i);
            dv /= static_cast<long double>(i + 1);
        }
    } else {
        dv = static_cast<long double>(1ULL << n_sites);
    }
    // Ranks are 64-bit (C(36,18) ~ 9.1e9); only per-sector INDEX values
    // must fit int32 (they index the sector basis, capped below).
    const std::uint64_t dim_full_sz = static_cast<std::uint64_t>(dv + 0.5L);
    if (data.reps.size() > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(
            "build_rep_mirror: sector has more than INT32_MAX representatives; "
            "the reverse-lookup value type would overflow");
    }

    // Device combinadic rank() reads a Pascal triangle from constant memory.
    ed::gpu::combinadic::detail::upload_pascal();

    // Reverse table. Stage-4 device twin (Jul 2026): when the host sector
    // carries the two-level lookup, upload the small per-sector remap and
    // co-own ONE shared rank table per (N, n_up) -- the per-sector dense
    // table below is then never built (this was 2.4 GiB PER SECTOR at
    // N=32 half filling). Fallback: the dense per-sector table, built
    // from reps only (no orbit walk).
    // Reverse lookup (Jul 2026 consolidation): the shared two-level table
    // when the host sector carries one (Stage 4 -- the production abelian
    // lane), otherwise the device BINARY SEARCH over the resident sorted
    // ``reps``. The per-sector dense rank table this replaced was strictly
    // dominated (2.4 GiB per sector at N=32; 36 GiB impossible at N=36) and
    // its ED_SYM_GPU_NO_RANKTABLE test hook is retired with it -- binary
    // search is now the default-tested path wherever two-level is absent.
    if (data.has_two_level()) {
        mirror->shared_rank_tab = acquire_gpu_shared_rank(data.shared_rank);
        mirror->d_local_of_shared = data.local_of_shared;
    } else if (std::getenv("ED_SYM_PROFILE") != nullptr) {
        std::fprintf(stderr,
                     "[sym_profile] GPU rep mirror: binary-search lookup over "
                     "%zu reps (rank space %llu)\n",
                     data.reps.size(),
                     static_cast<unsigned long long>(dim_full_sz));
    }

    std::vector<cuDoubleComplex> h_characters(data.characters.size());
    for (std::size_t g = 0; g < data.characters.size(); ++g) {
        h_characters[g] = make_cuDoubleComplex(data.characters[g].real(),
                                               data.characters[g].imag());
    }

    mirror->d_reps              = data.reps;
    mirror->d_inv_norms         = data.inv_norms;
    mirror->d_perms             = data.perms_flat;
    mirror->d_characters        = h_characters;
    if (data.has_flips()) {   // Stage 8b: flip-extended sector
        mirror->d_flips         = data.flip_masks;
    }
    // Byte-LUT permutation fast path (Jul 2026): reuse the host-built table
    // when the caller carries one, else build it here from perms_flat --
    // the ~740 KB (N=36, |G|=72) upload replaces the serial n_sites-loop
    // walk in the device canonicalization hot path.
    if (!data.perm_lut_data.empty()) {
        mirror->d_perm_lut   = data.perm_lut_data;
        mirror->perm_lut_bpw = data.perm_lut_bpw;
    } else if (n_sites > 0 && n_sites <= 64 && !data.perms_flat.empty()) {
        ed::symmetry::RepSectorData tmp;
        tmp.n_sites    = n_sites;
        tmp.group_size = data.group_size;
        tmp.perms_flat = data.perms_flat;
        tmp.build_perm_lut();
        mirror->d_perm_lut   = tmp.perm_lut_data;
        mirror->perm_lut_bpw = tmp.perm_lut_bpw;
    }

    mirror->d_diag_one_body    = terms.diag_one_body;
    mirror->d_offdiag_one_body = terms.offdiag_one_body;
    mirror->d_diag_two_body    = terms.diag_two_body;
    mirror->d_mixed_two_body   = terms.mixed_two_body;
    mirror->d_offdiag_two_body = terms.offdiag_two_body;
    mirror->d_three_body       = terms.three_body;

    cuda_check(cudaDeviceSynchronize(), "synchronize after rep mirror upload");
    return mirror;
}

}  // namespace detail

void launch_rep_symmetry_matvec(const GpuRepSectorMirror& mirror,
                                const cuDoubleComplex* d_in,
                                cuDoubleComplex* d_out,
                                std::size_t dim,
                                double spin_l)
{
    using detail::cuda_check;
    if (dim == 0) return;
    const auto basis = mirror.basis_view();
    const auto terms = mirror.terms_view();

    // DEFAULT: SOTA lock-free row GATHER (one write per row, no atomics, no
    // pre-zero memset; the diagonal is fused inline). Bisection fallback to the
    // validated atomic scatter via ED_MATVEC_SCATTER=1.
    static const bool use_scatter = []() {
        const char* v = std::getenv("ED_MATVEC_SCATTER");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();

    cudaError_t err;
    if (use_scatter) {
        cuda_check(cudaMemsetAsync(d_out, 0, dim * sizeof(cuDoubleComplex)),
                   "zero output before rep scatter kernel");
        err = ed::matvec::kernel::gpu::launch_apply_terms_rep_symmetry_gpu<
            ed::matvec::basis::DeviceRepSymmetryBasisPolicy,
            cuDoubleComplex>(basis, spin_l, terms, d_in, d_out,
                             /*stream=*/0, /*threads_per_block=*/256);
    } else {
        err = ed::matvec::kernel::gpu::launch_apply_terms_rep_symmetry_gpu_gather<
            ed::matvec::basis::DeviceRepSymmetryBasisPolicy,
            cuDoubleComplex>(basis, spin_l, terms, d_in, d_out,
                             /*stream=*/0, /*threads_per_block=*/256);
    }
    cuda_check(err, "apply_terms_rep_symmetry kernel launch");
}

}  // namespace ed::symmetry::gpu_mirror

// =============================================================================
// make_sector_matvec_gpu_rep -- on-the-fly representative GPU matvec entry.
//
// "On-the-fly representative SpMV for streaming symmetry" plan (Jun 2026).
//
// Builds a resident GpuRepSectorMirror from a CSR-free RepSectorData (reps +
// inv_norms + |G| characters + group perms) and returns a DEVICE-pointer
// MatvecFn driving ``apply_terms_rep_symmetry_scatter``. No orbit CSR / no
// O(full-Sz-dim) projection table is allocated or streamed -- this is the
// resident N=32 Sz+Symm path.
// =============================================================================
ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep(const ed::symmetry::RepSectorData& rep,
                                         double                         spin_l,
                                         const ed::matvec::TermStorage& terms)
{
    using ed::symmetry::gpu_mirror::GpuRepSectorMirror;
    using ed::symmetry::gpu_mirror::detail::build_rep_mirror;
    using ed::symmetry::gpu_mirror::launch_rep_symmetry_matvec;

    // B7: memoise the resident device mirror across binds. A single GS solve
    // binds the operator several times (phase-1 scan, phase-2 refine, vector
    // pull); without this each rebuilds the mirror and re-uploads reps + terms
    // (~150 MB at an N=32 sector).
    //
    // The key MUST be content-based, NOT the RepSectorData address: sector
    // operators are transient, so a destroyed sector's rep_data_ address is
    // reused by the next sector, and within one Hamiltonian every sector
    // shares identical terms -- so an (address, terms_hash) key collides
    // across DIFFERENT sectors and returns the wrong mirror (GPU OOB on a
    // dim-equal parity/flip pair). Key on the sector's own content: the
    // per-sector characters (unique per irrep), the rep-list signature
    // (n_up / size / samples), spin, and the term footprint.
    auto content_key = [](const ed::symmetry::RepSectorData& r,
                          const ed::matvec::TermStorage& t, double sl) {
        std::uint64_t h = 1469598103934665603ULL;
        // Avalanche every word (splitmix64 finalizer) BEFORE the FNV fold:
        // the plain XOR-multiply mix is structurally degenerate on the
        // sign-patterned character values this key exists to separate --
        // on a Z8 ring, chi_{k+4}(g) = (-1)^g chi_k(g) hashed IDENTICALLY
        // to chi_k, so conjugate-partner sectors reused each other's
        // mirror (the exact wrong-mirror bug the content key was built to
        // prevent; caught by test_rep_symmetry_gpu Z8 sectors 5/7).
        auto mix = [&h](std::uint64_t v) {
            v += 0x9E3779B97F4A7C15ULL;
            v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
            v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
            v ^= v >> 31;
            h ^= v;
            h *= 1099511628211ULL;
        };
        mix(static_cast<std::uint64_t>(r.n_up + 2));
        mix(static_cast<std::uint64_t>(r.group_size));
        mix(r.reps.size());
        if (!r.reps.empty()) {
            mix(r.reps.front());
            mix(r.reps[r.reps.size() / 2]);
            mix(r.reps.back());
        }
        for (const auto& c : r.characters) {   // per-irrep, uniquely identifies k
            mix(static_cast<std::uint64_t>(std::llround(c.real() * 1e9)));
            mix(static_cast<std::uint64_t>(std::llround(c.imag() * 1e9)));
        }
        if (!r.flip_masks.empty()) mix(r.flip_masks.front());
        mix(static_cast<std::uint64_t>(std::llround(sl * 1e6)));
        mix(t.diag_one_body.size());    mix(t.offdiag_one_body.size());
        mix(t.diag_two_body.size());    mix(t.mixed_two_body.size());
        mix(t.offdiag_two_body.size()); mix(t.three_body.size());
        if (!t.offdiag_two_body.empty())
            mix(static_cast<std::uint64_t>(
                std::llround(t.offdiag_two_body.back().coefficient.real() * 1e9)));
        return h;
    };
    // FULL term-content signature. The sector fingerprint below identifies
    // the rep basis but NOT the operator terms -- fine when one operator
    // (H) is mirrored per sector, WRONG when several operators share a
    // sector (e.g. a q-mesh of transverse probes O_q on the GS sector): a
    // content_key collision would reuse the first operator's device mirror.
    // This hashes EVERY term field (sites, op types, coeffs) so distinct
    // operators never alias.
    auto term_signature = [](const ed::matvec::TermStorage& t) {
        std::uint64_t h = 1469598103934665603ULL;
        auto mix = [&h](std::uint64_t v) {
            v += 0x9E3779B97F4A7C15ULL;
            v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
            v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
            v ^= v >> 31; h ^= v; h *= 1099511628211ULL;
        };
        auto mixc = [&](const std::complex<double>& c) {
            mix(static_cast<std::uint64_t>(std::llround(c.real() * 1e9)));
            mix(static_cast<std::uint64_t>(std::llround(c.imag() * 1e9)));
        };
        for (const auto& d : t.diag_one_body)   { mix(d.site_index); mixc(d.coefficient); }
        for (const auto& o : t.offdiag_one_body){ mix(o.site_index); mix(o.op_type); mixc(o.coefficient); }
        for (const auto& d : t.diag_two_body)   { mix(d.site_index_1); mix(d.site_index_2); mixc(d.coefficient); }
        for (const auto& m : t.mixed_two_body)  { mix(m.sz_site); mix(m.flip_site); mix(m.flip_op_type); mixc(m.coefficient); }
        for (const auto& o : t.offdiag_two_body){ mix(o.site_index_1); mix(o.site_index_2); mix(o.op_type_1); mix(o.op_type_2); mixc(o.coefficient); }
        for (const auto& b : t.three_body)      { mix(b.site_index_1); mix(b.site_index_2); mix(b.site_index_3);
                                                  mix(b.op_type_1); mix(b.op_type_2); mix(b.op_type_3); mixc(b.coefficient); }
        return h;
    };
    const std::uint64_t terms_sig = term_signature(terms);
    // Registry entries carry a FULL fingerprint of what the mirror encodes:
    // reuse must never depend on hash quality (a silent wrong-mirror hit is
    // wrong PHYSICS with correct-looking norms). The fingerprint covers
    // everything the device tables are built from except the reps list,
    // which is fully determined by (n_up window, group action, characters)
    // and cross-checked by its (size, front, mid, back) signature.
    struct MirrorSlot {
        int                                       n_up;
        double                                    spin;
        std::uint64_t                             reps_sig[4];
        std::uint64_t                             terms_sig;
        std::vector<std::complex<double>>         chi;
        std::vector<int>                          perms;
        std::vector<std::uint64_t>                flips;
        std::weak_ptr<const GpuRepSectorMirror>   mirror;
    };
    auto fingerprint_matches = [](const MirrorSlot& s,
                                  const ed::symmetry::RepSectorData& r,
                                  double sl, std::uint64_t tsig) {
        return s.n_up == r.n_up && s.spin == sl
            && s.terms_sig == tsig            // operator terms, not just sector
            && s.reps_sig[0] == r.reps.size()
            && s.reps_sig[1] == (r.reps.empty() ? 0 : r.reps.front())
            && s.reps_sig[2] == (r.reps.empty() ? 0
                                   : r.reps[r.reps.size() / 2])
            && s.reps_sig[3] == (r.reps.empty() ? 0 : r.reps.back())
            && s.chi == r.characters
            && s.perms == r.perms_flat
            && s.flips == r.flip_masks;
    };

    std::shared_ptr<const GpuRepSectorMirror> mirror;
    {
        static std::mutex mtx;
        static std::map<std::uint64_t, std::vector<MirrorSlot>> registry;
        static std::vector<std::shared_ptr<const GpuRepSectorMirror>> keep;
        // Jul 2026: byte-aware strong cache (count cap 4 pinned ~4 x 4 GB of
        // sector arrays at N=36). Shares ED_GPU_SYM_CACHE_GIB semantics.
        static const double kKeepBudget = [] {
            double gib = 16.0;
            if (const char* v = std::getenv("ED_GPU_SYM_CACHE_GIB")) {
                const double parsed = std::atof(v);
                if (parsed > 0.0) gib = parsed;
            }
            return gib * 1073741824.0;
        }();
        const std::uint64_t key = content_key(rep, terms, spin_l);
        std::lock_guard<std::mutex> lk(mtx);
        auto& bucket = registry[key];
        for (auto it = bucket.begin(); it != bucket.end();) {
            auto locked = it->mirror.lock();
            if (!locked) { it = bucket.erase(it); continue; }   // expired
            if (fingerprint_matches(*it, rep, spin_l, terms_sig)) {
                mirror = std::move(locked);
                break;
            }
            ++it;
        }
        if (!mirror) {
            mirror = build_rep_mirror(rep, spin_l, terms);
            MirrorSlot s;
            s.n_up        = rep.n_up;
            s.spin        = spin_l;
            s.reps_sig[0] = rep.reps.size();
            s.reps_sig[1] = rep.reps.empty() ? 0 : rep.reps.front();
            s.reps_sig[2] = rep.reps.empty() ? 0
                              : rep.reps[rep.reps.size() / 2];
            s.reps_sig[3] = rep.reps.empty() ? 0 : rep.reps.back();
            s.terms_sig   = terms_sig;
            s.chi         = rep.characters;
            s.perms       = rep.perms_flat;
            s.flips       = rep.flip_masks;
            s.mirror      = mirror;
            bucket.push_back(std::move(s));
            keep.push_back(mirror);
            auto bytes_of = [](const std::shared_ptr<const GpuRepSectorMirror>& mm) {
                return static_cast<double>(
                    mm->d_reps.size() * 8 + mm->d_inv_norms.size() * 8
                    + mm->d_perm_lut.size() * 8 + mm->d_perms.size() * 4
                    + mm->d_local_of_shared.size() * 4);
            };
            double total = 0.0;
            for (const auto& mm : keep) total += bytes_of(mm);
            while (keep.size() > 1 && total > kKeepBudget) {
                total -= bytes_of(keep.front());
                keep.erase(keep.begin());
            }
        }
    }

    const double spin = spin_l;
    const std::uint64_t dim_captured = mirror->dim;

    return [mirror, spin, dim_captured](const ed::matvec::Complex* in,
                                        ed::matvec::Complex* out,
                                        std::size_t n) {
        if (n != dim_captured) {
            throw std::runtime_error(
                "ed::symmetry::make_sector_matvec_gpu_rep: size mismatch (" +
                std::to_string(n) + " vs " +
                std::to_string(dim_captured) + ")");
        }
        launch_rep_symmetry_matvec(
            *mirror,
            reinterpret_cast<const cuDoubleComplex*>(in),
            reinterpret_cast<cuDoubleComplex*>(out),
            n,
            spin);
    };
}

// ---------------------------------------------------------------------------
// Host-pointer twin: persistent device staging buffers around the resident
// mirror, one H2D + D2H per apply. Built for the little-group engine's CPU
// Lanczos (host vectors); the staging traffic is O(dim) against the kernel's
// O(dim * terms * |G|) walk.
// ---------------------------------------------------------------------------
ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep_hostptr(
    const ed::symmetry::RepSectorData& rep,
    double                             spin_l,
    const ed::matvec::TermStorage&     terms)
{
    using ed::symmetry::gpu_mirror::detail::cuda_check;

    auto dev_fn = ed::symmetry::make_sector_matvec_gpu_rep(rep, spin_l, terms);
    auto d_in   = std::make_shared<thrust::device_vector<cuDoubleComplex>>();
    auto d_out  = std::make_shared<thrust::device_vector<cuDoubleComplex>>();

    return [dev_fn, d_in, d_out](const ed::matvec::Complex* in,
                                 ed::matvec::Complex*       out,
                                 std::size_t                n) {
        if (d_in->size() != n) {
            d_in->resize(n);
            d_out->resize(n);
        }
        cuda_check(cudaMemcpy(thrust::raw_pointer_cast(d_in->data()), in,
                              n * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "hostptr rep matvec H2D");
        dev_fn(reinterpret_cast<const ed::matvec::Complex*>(
                   thrust::raw_pointer_cast(d_in->data())),
               reinterpret_cast<ed::matvec::Complex*>(
                   thrust::raw_pointer_cast(d_out->data())),
               n);
        cuda_check(cudaMemcpy(out, thrust::raw_pointer_cast(d_out->data()),
                              n * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost),
                   "hostptr rep matvec D2H");
    };
}

#endif  // WITH_CUDA
