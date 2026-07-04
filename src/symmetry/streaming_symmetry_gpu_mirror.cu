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
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::symmetry::gpu_mirror {

namespace detail {

// ---------------------------------------------------------------------------
// Phase I of the "Close CPU / GPU Gaps" plan (May 2026):
// ED_GPU_SYMMETRY_MIRROR_V2=1 (any non-zero string) enables the
// targeted small-win optimizations:
//   - dim-banded ``threads_per_block`` sweep
//   - ``cudaMemsetAsync`` on a per-mirror side stream, gated by event
//     so the kernel launch on the default stream waits on the memset
//     completion without serializing the host call.
// Off by default until the acceptance harness ships measurements;
// the pre-baked ``orbit_inv_norms`` win above is unconditional
// because it's a pure correctness-preserving simplification.
// ---------------------------------------------------------------------------
inline bool v2_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("ED_GPU_SYMMETRY_MIRROR_V2");
        if (env == nullptr || env[0] == '\0') return false;
        return env[0] != '0';
    }();
    return enabled;
}

inline int v2_threads_per_block_for(std::size_t dim) {
    // Heuristic from the Phase I plan: small sectors do better with
    // fewer threads per block (more blocks -> more chance of
    // concurrent SM occupancy and less per-block atomic contention);
    // very large sectors prefer more threads per block (amortizes
    // scheduler overhead). Boundaries match the plan -- can be
    // re-tuned once Nsight Compute traces land in docs/perf/.
    if (dim <= 2048)  return 128;
    if (dim <= 16384) return 256;
    return 512;
}

inline constexpr std::uint64_t kEmptyKey = static_cast<std::uint64_t>(-1);

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("StreamingSymmetry GPU mirror: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

inline std::uint64_t next_pow2(std::uint64_t x) noexcept {
    if (x <= 1) return 1;
    std::uint64_t v = 1;
    while (v < x) v <<= 1;
    return v;
}

inline std::uint64_t hash_capacity_for(std::uint64_t n_entries) noexcept {
    // Target ~2x occupancy so open-addressing probes stay short. The
    // ceiling at 2^31 is defensive -- at 2^30 entries we are already
    // over 64 GiB of hash table memory, which any sane workflow has
    // partitioned across multiple GPUs by now.
    const std::uint64_t target =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(16),
                                 n_entries * 2);
    const std::uint64_t cap = next_pow2(target);
    return std::min<std::uint64_t>(cap, static_cast<std::uint64_t>(1) << 31);
}

}  // namespace detail

// =============================================================================
// GpuSectorMirror -- owning device snapshot for one symmetry sector.
//
// Every device buffer is held by a thrust::device_vector so the dtor
// runs cudaFree on each in turn. The basis_view() / terms_view() helpers
// rebuild the POD ABI structs the kernel expects from the current raw
// pointers (cheap; called once per matvec from the captured lambda).
// =============================================================================
struct GpuSectorMirror {
    std::size_t sector_idx = static_cast<std::size_t>(-1);

    // -------- orbit CSR (length sum |orbit_i| except offsets length dim+1) --
    thrust::device_vector<std::uint64_t>             d_orbit_elements;
    thrust::device_vector<cuDoubleComplex>           d_orbit_coefficients;
    thrust::device_vector<std::uint32_t>             d_orbit_offsets;
    // Phase I (Close CPU/GPU Gaps, May 2026): stores pre-baked
    // ``1.0 / norm_j`` so the inner-loop scaling in
    // ``apply_terms_gpu_scatter`` is a multiply rather than a
    // divide. Host-side raw norms come from the SymmetrySector and
    // are inverted exactly once per mirror upload.
    thrust::device_vector<double>                    d_orbit_inv_norms;

    // -------- hash table for state -> (basis_idx, projection) ----------------
    // Legacy reverse-lookup path -- still populated for sym-only
    // sectors (n_up_ < 0) where the dense rank table below would not
    // fit. For sym+Sz sectors the hash stays empty and the rank-table
    // path takes over (Phase E.1).
    thrust::device_vector<ed::matvec::basis::DeviceSymmetryHashEntry>
        d_hash_table;
    std::uint32_t hash_mask = 0;

    // -------- Phase E.1: dense rank-table reverse-lookup ---------------------
    // Two device arrays indexed by ``rank_combination(state, n_sites,
    // n_up) in [0, C(n_sites, n_up))``:
    //   * ``d_sz_to_sec[r]`` -- the canonical sector index for state
    //     ``unrank(r)``, or ``-1`` if that state is not in this irrep.
    //   * ``d_sz_to_proj[r]`` -- the pre-baked projection factor
    //     ``conj(alpha_{s,k}) * group_norm / norm_k``; unspecified
    //     when ``d_sz_to_sec[r] < 0``.
    // Empty for sym-only sectors; populated for sym+Sz sectors with
    // n_up_ >= 0 unless ``ED_GPU_USE_HASH=1`` re-enables the hash path
    // wholesale (build_mirror gates on the env var).
    thrust::device_vector<std::int32_t>    d_sz_to_sec;
    thrust::device_vector<cuDoubleComplex> d_sz_to_proj;
    int           n_sites_   = 0;
    int           n_up_      = -1;       // -1 -> sym-only, no rank table

    double        group_norm = 1.0;
    std::uint64_t dim = 0;

    // -------- term SoA bins (per-mirror copy of the parent's terms_) --------
    thrust::device_vector<ed::matvec::DiagOneBody>     d_diag_one_body;
    thrust::device_vector<ed::matvec::OffDiagOneBody>  d_offdiag_one_body;
    thrust::device_vector<ed::matvec::DiagTwoBody>     d_diag_two_body;
    thrust::device_vector<ed::matvec::MixedTwoBody>    d_mixed_two_body;
    thrust::device_vector<ed::matvec::OffDiagTwoBody>  d_offdiag_two_body;
    thrust::device_vector<ed::matvec::ThreeBodyTerm>   d_three_body;

    double spin_l = 0.5;

    // Phase I (Close CPU/GPU Gaps, May 2026): per-mirror side stream
    // + event used to overlap ``cudaMemsetAsync(d_out)`` with the
    // prior matvec's tail. Created lazily on first matvec when
    // ``ED_GPU_SYMMETRY_MIRROR_V2`` is enabled; otherwise the stream
    // stays at ``nullptr`` and the legacy single-stream path runs.
    // ``cudaStreamDestroy`` / ``cudaEventDestroy`` are safe on
    // ``nullptr`` so the destructor stays trivial in the V2-off
    // case.
    mutable cudaStream_t memset_stream = nullptr;
    mutable cudaEvent_t  memset_done   = nullptr;
    mutable bool         v2_resources_inited = false;

    ~GpuSectorMirror() {
        if (memset_done   != nullptr) cudaEventDestroy(memset_done);
        if (memset_stream != nullptr) cudaStreamDestroy(memset_stream);
    }

    ed::matvec::basis::DeviceSymmetryBasisPolicy basis_view() const noexcept {
        ed::matvec::basis::DeviceSymmetryBasisPolicy v;
        v.orbit_elements     = thrust::raw_pointer_cast(d_orbit_elements.data());
        v.orbit_coefficients = thrust::raw_pointer_cast(d_orbit_coefficients.data());
        v.orbit_offsets      = thrust::raw_pointer_cast(d_orbit_offsets.data());
        v.orbit_inv_norms    = thrust::raw_pointer_cast(d_orbit_inv_norms.data());
        v.dim_               = dim;
        v.group_norm         = group_norm;
        // Phase E.1: thread the rank-table fields into the device view.
        // ``sz_to_sec == nullptr`` is the signal "use the hash"; the
        // device-side lookup branches on it.
        const bool have_rank_table = !d_sz_to_sec.empty();
        v.hash_table  = have_rank_table
                          ? nullptr
                          : thrust::raw_pointer_cast(d_hash_table.data());
        v.hash_mask   = have_rank_table ? 0u : hash_mask;
        v.sz_to_sec   = have_rank_table
                          ? thrust::raw_pointer_cast(d_sz_to_sec.data())
                          : nullptr;
        v.sz_to_proj  = have_rank_table
                          ? thrust::raw_pointer_cast(d_sz_to_proj.data())
                          : nullptr;
        v.n_sites     = n_sites_;
        v.n_up        = n_up_;
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

// Build a GpuSectorMirror from a host SymmetrySector + parent's term
// storage. Templated so it works for both the full-Hilbert and fixed-Sz
// streaming-symmetry operators (the SymmetrySector layout is shared).
//
// The terms are uploaded byte-for-byte from the host SoA. Each term
// struct is trivially copyable (POD with primitive fields + std::complex,
// which is layout-compatible with two consecutive doubles).
template <class SectorRef>
std::shared_ptr<GpuSectorMirror>
build_mirror(const SectorRef& sector,
             double group_size,
             double spin_l,
             const ed::matvec::TermStorage& terms,
             std::size_t sector_idx,
             int n_sites = 0,
             int n_up    = -1)
{
    auto mirror = std::make_shared<GpuSectorMirror>();
    mirror->sector_idx = sector_idx;
    mirror->spin_l     = spin_l;
    mirror->group_norm = (group_size > 0.0) ? (1.0 / group_size) : 0.0;
    mirror->n_sites_   = n_sites;
    mirror->n_up_      = n_up;

    const std::size_t sector_dim = sector.basis_states.size();
    mirror->dim = static_cast<std::uint64_t>(sector_dim);

    // -----------------------------------------------------------------
    // Pack orbit CSR on the host.
    // -----------------------------------------------------------------
    std::vector<std::uint32_t> h_offsets(sector_dim + 1, 0);
    std::size_t total_elements = 0;
    for (std::size_t i = 0; i < sector_dim; ++i) {
        total_elements += sector.basis_states[i].orbit_elements.size();
        h_offsets[i + 1] = static_cast<std::uint32_t>(total_elements);
    }
    std::vector<std::uint64_t>    h_elements(total_elements);
    std::vector<cuDoubleComplex>  h_coefs(total_elements);
    // Phase I: pre-bake ``1.0 / norm_i`` so the kernel inner loop is
    // multiply-only. Treats norm == 0 defensively as the host
    // representative storage already guarantees positive norms for
    // every populated orbit; if a degenerate sector ever produced
    // norm == 0 the kernel would zero-weight that orbit walk, which
    // is the correct behavior.
    std::vector<double>           h_inv_norms(sector_dim);
    for (std::size_t i = 0; i < sector_dim; ++i) {
        const auto& bs = sector.basis_states[i];
        h_inv_norms[i] = (bs.norm > 0.0) ? (1.0 / bs.norm) : 0.0;
        const std::uint32_t off = h_offsets[i];
        for (std::size_t k = 0; k < bs.orbit_elements.size(); ++k) {
            h_elements[off + k] = bs.orbit_elements[k];
            const std::complex<double>& c = bs.orbit_coefficients[k];
            h_coefs[off + k] = make_cuDoubleComplex(c.real(), c.imag());
        }
    }

    // -----------------------------------------------------------------
    // Phase E.1 of the "Kill the GPU State-Lookup Hash" plan (May 2026):
    // decide between the dense rank table (sym + Sz, the user's actual
    // 32-site workload) and the legacy open-addressing hash (sym-only,
    // where C(N, n_up) is not defined and the hash cap stays bounded
    // by the orbit total). The choice flips on:
    //
    //   * ``n_up >= 0``   (this build is for a Sz-conserving sector), AND
    //   * ``ED_GPU_USE_HASH != 1``  (rollback handle stays the legacy
    //                                 path end-to-end for diagnostics).
    //
    // We also guard against degenerate cases where C(N, n_up) is large
    // enough that the dense table would be worse than the hash.
    // For n_up == 0 or n_up == n_sites the sector has dim 1, and we
    // still build the rank table (1-entry array, costs nothing).
    // -----------------------------------------------------------------
    static const bool kLegacyHashRequested = []() {
        const char* s = std::getenv("ED_GPU_USE_HASH");
        return (s != nullptr && s[0] == '1');
    }();
    const bool can_use_rank = (n_sites > 0) && (n_up >= 0)
                              && (n_up <= n_sites) && (n_sites <= 64);
    // Compute C(n_sites, n_up) host-side; cap at INT32_MAX so the
    // sz_to_sec[r] = int32 cast stays defined. Beyond that we'd need a
    // 64-bit rank type, which is out of scope for this phase.
    std::uint64_t dim_full_sz = 0;
    if (can_use_rank) {
        // Standard host binomial; n_sites <= 64 so 128-bit math is not
        // required for the intermediate.
        int k = (n_up < n_sites - n_up) ? n_up : (n_sites - n_up);
        long double dv = 1.0L;
        for (int i = 0; i < k; ++i) {
            dv *= static_cast<long double>(n_sites - i);
            dv /= static_cast<long double>(i + 1);
        }
        if (dv <= static_cast<long double>(std::numeric_limits<std::int32_t>::max())) {
            dim_full_sz = static_cast<std::uint64_t>(dv + 0.5L);
        }
    }
    const bool use_rank_table = can_use_rank
                                && !kLegacyHashRequested
                                && dim_full_sz > 0;

    if (use_rank_table) {
        // ---- Dense rank-table path -----------------------------------
        // Allocate host-side sz_to_sec[] sized C(n_sites, n_up). The
        // cudaMemcpyToSymbol-friendly Pascal upload happens on first
        // device-side rank() call; we do an explicit upload here so
        // every TU using the rank table sees the constant memory
        // already populated.
        ed::gpu::combinadic::detail::upload_pascal();

        std::vector<std::int32_t>    h_sz_to_sec(dim_full_sz, -1);
        std::vector<cuDoubleComplex> h_sz_to_proj(dim_full_sz,
                                                  make_cuDoubleComplex(0.0, 0.0));

        // Host-side Pascal table; matches the device ``d_pascal_shared``
        // upload byte-for-byte. Computed once per build_mirror call;
        // 65x65 = 4225 entries, trivial cost relative to the orbit
        // walk that follows.
        std::vector<std::vector<std::uint64_t>> pascal(65, std::vector<std::uint64_t>(65, 0));
        for (int nn = 0; nn <= 64; ++nn) {
            pascal[nn][0] = 1ULL;
            for (int kk = 1; kk <= nn; ++kk) {
                pascal[nn][kk] = pascal[nn - 1][kk - 1]
                                 + (kk <  nn ? pascal[nn - 1][kk] : 0ULL);
            }
        }
        auto binom_host = [&pascal](int nn, int kk) -> std::uint64_t {
            if (kk < 0 || kk > nn || nn < 0 || nn > 64) return 0ULL;
            return pascal[nn][kk];
        };

        // Same orbit-walk loop as the hash build, but writing into the
        // dense table instead of inserting into the open-addressing
        // hash. ``rank_combination_host`` is the inverse of the device
        // ``rank_state`` -- we replicate the colex convention here so
        // host and device land on the same r.
        //
        // Previously this had an inline binomial that returned 1 for
        // C(0, 1) (any state with bit 0 set), shifting every host rank
        // by 1 relative to the device. The Pascal-table form above
        // exactly matches the device ``binomial()`` from
        // ``combinadic.cuh`` (both return 0 when ``k > n``), so host
        // and device produce identical r for every valid fixed-Sz
        // state.
        auto rank_combination_host = [&binom_host, n_sites](
            std::uint64_t state, int kk) -> std::uint64_t {
            std::uint64_t r = 0;
            int seen = 0;
            for (int bit = 0; bit < n_sites && seen < kk; ++bit) {
                if ((state >> bit) & 1ULL) {
                    ++seen;
                    r += binom_host(bit, seen);
                }
            }
            return r;
        };

        for (std::size_t k = 0; k < sector_dim; ++k) {
            const auto& bs = sector.basis_states[k];
            const double inv_norm_k = (bs.norm != 0.0) ? (1.0 / bs.norm) : 0.0;
            const double scale = mirror->group_norm * inv_norm_k;
            for (std::size_t j = 0; j < bs.orbit_elements.size(); ++j) {
                const std::uint64_t s_prime = bs.orbit_elements[j];
                const std::complex<double> alpha = bs.orbit_coefficients[j];
                const double proj_re =  alpha.real() * scale;
                const double proj_im = -alpha.imag() * scale;
                const std::uint64_t r = rank_combination_host(s_prime, n_up);
                if (r >= dim_full_sz) continue;  // defensive: out-of-sector
                h_sz_to_sec[r]  = static_cast<std::int32_t>(k);
                h_sz_to_proj[r] = make_cuDoubleComplex(proj_re, proj_im);
            }
        }

        // Hash table stays empty for this sector; the basis_view()
        // helper switches paths on ``d_sz_to_sec.empty()``.
        mirror->d_orbit_elements     = h_elements;
        mirror->d_orbit_coefficients = h_coefs;
        mirror->d_orbit_offsets      = h_offsets;
        mirror->d_orbit_inv_norms    = h_inv_norms;
        mirror->d_sz_to_sec          = h_sz_to_sec;
        mirror->d_sz_to_proj         = h_sz_to_proj;
        mirror->hash_mask            = 0;
    } else {
        // ---- Legacy open-addressing hash path -----------------------
        // Build the (state -> idx, projection) hash table on the host.
        //
        // For each basis state k and each orbit element s' with
        // coefficient alpha_{s',k}, insert key=s', value=k, projection
        // = conj(alpha) * group_norm / norm_k. Open-addressing linear
        // probing on a power-of-two table; the same Fibonacci constant
        // the device kernel uses (so insertions and lookups follow
        // identical probe sequences).
        //
        // Note on correctness: orbits in a single sector are disjoint,
        // so every computational state s' appears in at most one
        // orbit. The "key already present" branch below is therefore
        // unreachable for valid sector data; we keep it defensively in
        // case the host construction ever produces duplicate orbit
        // entries.
        const std::uint64_t hash_cap = hash_capacity_for(total_elements);
        if (hash_cap == 0 || (hash_cap & (hash_cap - 1)) != 0) {
            throw std::runtime_error(
                "StreamingSymmetry GPU mirror: hash capacity is not a power of two");
        }
        mirror->hash_mask = static_cast<std::uint32_t>(hash_cap - 1);

        std::vector<ed::matvec::basis::DeviceSymmetryHashEntry> h_hash(hash_cap);
        for (auto& e : h_hash) {
            e.key = kEmptyKey;
            e.value = 0;
            e._pad = 0;
            e.projection = make_cuDoubleComplex(0.0, 0.0);
        }

        for (std::size_t k = 0; k < sector_dim; ++k) {
            const auto& bs = sector.basis_states[k];
            const double inv_norm_k = (bs.norm != 0.0) ? (1.0 / bs.norm) : 0.0;
            const double scale = mirror->group_norm * inv_norm_k;
            for (std::size_t j = 0; j < bs.orbit_elements.size(); ++j) {
                const std::uint64_t s_prime = bs.orbit_elements[j];
                const std::complex<double> alpha = bs.orbit_coefficients[j];
                const double proj_re =  alpha.real() * scale;
                const double proj_im = -alpha.imag() * scale;

                std::uint64_t h = (s_prime * 11400714819323198485ULL) & mirror->hash_mask;
                for (;;) {
                    if (h_hash[h].key == kEmptyKey) {
                        h_hash[h].key        = s_prime;
                        h_hash[h].value      = static_cast<std::uint32_t>(k);
                        h_hash[h].projection = make_cuDoubleComplex(proj_re, proj_im);
                        break;
                    }
                    if (h_hash[h].key == s_prime) break;
                    h = (h + 1) & mirror->hash_mask;
                }
            }
        }

        mirror->d_orbit_elements     = h_elements;
        mirror->d_orbit_coefficients = h_coefs;
        mirror->d_orbit_offsets      = h_offsets;
        mirror->d_orbit_inv_norms    = h_inv_norms;
        mirror->d_hash_table         = h_hash;
    }

    // Term SoA: trivially copyable POD records -> direct H2D.
    mirror->d_diag_one_body      = terms.diag_one_body;
    mirror->d_offdiag_one_body   = terms.offdiag_one_body;
    mirror->d_diag_two_body      = terms.diag_two_body;
    mirror->d_mixed_two_body     = terms.mixed_two_body;
    mirror->d_offdiag_two_body   = terms.offdiag_two_body;
    mirror->d_three_body         = terms.three_body;

    cuda_check(cudaDeviceSynchronize(),
               "synchronize after GPU mirror upload");

    return mirror;
}

}  // namespace detail

// =============================================================================
// Public dispatch -- launches the unified GPU kernel against the
// uploaded mirror. Called from the captured lambdas the
// bind_cuda_for_sector() helpers return below.
// =============================================================================
void launch_symmetry_matvec(const GpuSectorMirror& mirror,
                            const cuDoubleComplex* d_in,
                            cuDoubleComplex* d_out,
                            std::size_t dim,
                            double spin_l)
{
    using detail::cuda_check;
    if (dim == 0) return;
    const bool v2 = detail::v2_enabled();

    if (v2) {
        // Phase I V2 path: route the output-zeroing through a
        // per-mirror side stream and gate the kernel launch via an
        // event. The host call returns as soon as the kernel is
        // enqueued; the scheduler is free to overlap the side-stream
        // memset with the matvec invocation that produced the input
        // (typical Lanczos pattern: previous matvec -> compute on
        // host alpha/beta -> next matvec). Default-stream semantics
        // are preserved by waiting on the event before launching.
        if (!mirror.v2_resources_inited) {
            cuda_check(cudaStreamCreateWithFlags(
                           const_cast<cudaStream_t*>(&mirror.memset_stream),
                           cudaStreamNonBlocking),
                       "create memset side stream");
            cuda_check(cudaEventCreateWithFlags(
                           const_cast<cudaEvent_t*>(&mirror.memset_done),
                           cudaEventDisableTiming),
                       "create memset done event");
            mirror.v2_resources_inited = true;
        }
        cuda_check(cudaMemsetAsync(d_out, 0,
                                   dim * sizeof(cuDoubleComplex),
                                   mirror.memset_stream),
                   "zero output (side stream)");
        cuda_check(cudaEventRecord(mirror.memset_done, mirror.memset_stream),
                   "record memset event");
        cuda_check(cudaStreamWaitEvent(/*stream=*/0, mirror.memset_done, 0),
                   "wait for memset event");
    } else {
        cuda_check(cudaMemsetAsync(d_out, 0,
                                   dim * sizeof(cuDoubleComplex)),
                   "zero output before kernel");
    }

    const auto basis = mirror.basis_view();
    const auto terms = mirror.terms_view();
    const int threads_per_block =
        v2 ? detail::v2_threads_per_block_for(static_cast<std::size_t>(dim))
           : 256;
    const cudaError_t err =
        ed::matvec::kernel::gpu::launch_apply_terms_gpu<
            ed::matvec::basis::DeviceSymmetryBasisPolicy,
            cuDoubleComplex>(basis, spin_l, terms, d_in, d_out,
                             /*stream=*/0, threads_per_block);
    cuda_check(err, "apply_terms_gpu_scatter kernel launch");
}

// =============================================================================
// GpuRepSectorMirror -- on-the-fly representative SpMV device snapshot.
//
// "On-the-fly representative SpMV for streaming symmetry" plan (Jun 2026).
//
// Unlike GpuSectorMirror this holds NO orbit CSR and NO O(full-Sz-dim)
// projection table. The resident footprint is:
//   * reps (dim x 8 B) + inv_norms (dim x 8 B)
//   * the |G| site permutations (group_size * n_sites ints)
//   * the per-sector character array (group_size complex)
//   * rep_index_of_rank: C(n_sites, n_up) int32 reverse lookup keyed by the
//     combinadic rank of an orbit REPRESENTATIVE (built from ``reps`` alone,
//     no orbit walk).
// The group action + projection are regenerated arithmetically inside the
// kernel; per-SpMV traffic is just the in/out vectors -> the genuine /|G|.
// =============================================================================
struct GpuRepSectorMirror {
    thrust::device_vector<std::uint64_t>   d_reps;
    thrust::device_vector<double>          d_inv_norms;
    thrust::device_vector<int>             d_perms;
    thrust::device_vector<cuDoubleComplex> d_characters;
    thrust::device_vector<std::uint64_t>   d_flips;   // Stage 8b: flip masks
    thrust::device_vector<std::int32_t>    d_rep_index_of_rank;

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
        v.rep_index_of_rank = thrust::raw_pointer_cast(d_rep_index_of_rank.data());
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
    if (dv > static_cast<long double>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(
            "build_rep_mirror: C(n_sites, n_up) exceeds INT32_MAX; the rank "
            "table value type would overflow");
    }
    const std::uint64_t dim_full_sz = static_cast<std::uint64_t>(dv + 0.5L);

    // Device combinadic rank() reads a Pascal triangle from constant memory.
    ed::gpu::combinadic::detail::upload_pascal();

    // Host Pascal table matching the device ``binomial`` (0 when k > n), so
    // host-built ranks line up with the device ``rank_state``.
    std::vector<std::vector<std::uint64_t>> pascal(
        65, std::vector<std::uint64_t>(65, 0));
    for (int nn = 0; nn <= 64; ++nn) {
        pascal[nn][0] = 1ULL;
        for (int kk = 1; kk <= nn; ++kk) {
            pascal[nn][kk] = pascal[nn - 1][kk - 1]
                             + (kk < nn ? pascal[nn - 1][kk] : 0ULL);
        }
    }
    auto binom_host = [&pascal](int nn, int kk) -> std::uint64_t {
        if (kk < 0 || kk > nn || nn < 0 || nn > 64) return 0ULL;
        return pascal[nn][kk];
    };
    auto rank_combination_host = [&binom_host, n_sites](
        std::uint64_t state, int kk) -> std::uint64_t {
        std::uint64_t r = 0;
        int seen = 0;
        for (int bit = 0; bit < n_sites && seen < kk; ++bit) {
            if ((state >> bit) & 1ULL) {
                ++seen;
                r += binom_host(bit, seen);
            }
        }
        return r;
    };

    // Reverse table: rank(representative) -> orbit index. Built from reps
    // ONLY -- no orbit images materialised.
    std::vector<std::int32_t> h_rep_index_of_rank(dim_full_sz, -1);
    for (std::size_t i = 0; i < data.reps.size(); ++i) {
        const std::uint64_t r = (n_up >= 0)
            ? rank_combination_host(data.reps[i], n_up)
            : data.reps[i];                    // full space: identity rank
        if (r < dim_full_sz) {
            h_rep_index_of_rank[r] = static_cast<std::int32_t>(i);
        }
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
    mirror->d_rep_index_of_rank = h_rep_index_of_rank;

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
// make_sector_matvec_gpu -- standalone per-sector GPU matvec entry.
//
// The collapse-target twin of bind_cuda_for_sector: where the legacy
// operators own an LRU-1 cache keyed by sector index (because one
// monolithic operator holds every sector), an ``ed::symmetry::SectorOperator``
// IS a single sector, so we build the mirror ONCE here and capture it by
// value (shared_ptr) in the returned callable. A Lanczos / FTLM sweep
// reuses the same mirror across all of its matvecs; the mirror is freed
// when the callable is dropped.
//
// Declared in include/ed/symmetry/sector_gpu_mirror.h (CUDA-free header);
// the non-CUDA stub lives in streaming_symmetry_gpu_mirror.cpp.
// =============================================================================

ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu(const ::SymmetrySector&        sector,
                                     double                         group_size,
                                     double                         spin_l,
                                     const ed::matvec::TermStorage& terms,
                                     int                            n_sites,
                                     int                            n_up)
{
    using ed::symmetry::gpu_mirror::GpuSectorMirror;
    using ed::symmetry::gpu_mirror::detail::build_mirror;
    using ed::symmetry::gpu_mirror::launch_symmetry_matvec;

    // One-shot device mirror for this single sector (no LRU; the
    // SectorOperator owns exactly one sector). sector_idx is irrelevant
    // here (no cache to key), so pass 0.
    std::shared_ptr<const GpuSectorMirror> mirror = build_mirror(
        sector,
        group_size,
        spin_l,
        terms,
        /*sector_idx=*/static_cast<std::size_t>(0),
        n_sites,
        n_up);

    const double spin = spin_l;
    const std::uint64_t dim_captured = mirror->dim;

    return [mirror, spin, dim_captured](const ed::matvec::Complex* in,
                                        ed::matvec::Complex* out,
                                        std::size_t n) {
        if (n != dim_captured) {
            throw std::runtime_error(
                "ed::symmetry::make_sector_matvec_gpu: size mismatch (" +
                std::to_string(n) + " vs " +
                std::to_string(dim_captured) + ")");
        }
        launch_symmetry_matvec(
            *mirror,
            reinterpret_cast<const cuDoubleComplex*>(in),
            reinterpret_cast<cuDoubleComplex*>(out),
            n,
            spin);
    };
}

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

    std::shared_ptr<const GpuRepSectorMirror> mirror =
        build_rep_mirror(rep, spin_l, terms);

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

#endif  // WITH_CUDA
