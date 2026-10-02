// =============================================================================
// src/gpu/rep_matvec.cu
//
// On-the-fly representative GPU sector matvecs (the definitions behind ed/gpu/rep_matvec.h;
// rep_matvec_stub.cpp holds the throwing stubs of a build without WITH_CUDA).
//
// A sector is uploaded once (GpuSectorMirror: representatives, inverse norms, group
// permutations, characters, the reverse rank lookup) and shared by every operator bound on
// it; an operator's row program (ed::ops::MaskedProgram) is uploaded per bind. The returned
// functions take DEVICE pointers (the bind_cuda() contract of <ed/matvec/linear_operator.h>)
// and launch the row walk kernel (walk_gather) with a row and a column sector.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/core/config.h>
#include <ed/core/log.h>
#include <ed/gpu/device_basis_policy.cuh>
#include <ed/gpu/rep_matvec.h>
#include <ed/ops/program.h>
#include <ed/ops/row_walk.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <cmath>
#include <utility>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
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

// =============================================================================
// GpuSectorMirror -- the only device representation of a symmetry sector. It holds NO orbit
// CSR and NO O(full-Sz-dim) projection table, only:
//   * reps (dim x 8 B) + inv_norms (dim x 8 B)
//   * the |G| site permutations (group_size * n_sites ints) and their byte LUT
//   * the per-sector character array (group_size complex)
//   * the reverse lookup: the shared rank table below plus a per-sector remap, or a
//     binary search over the sorted reps.
// The group action + projection are regenerated arithmetically inside the kernel; per-SpMV
// traffic is just the in/out vectors -> the genuine /|G|.
// =============================================================================
// ONE rank -> shared-rep-index table per (N, n_up) subspace, co-owned by
// every irrep sector's mirror through a content-keyed weak registry. Avoids
// a per-sector C(N, n_up) x int32 copy (2.4 GiB EACH at N=32 half filling).
struct GpuSharedRankTable {
    thrust::device_vector<std::int32_t> d_shared_of_rank;
};

// The byte budget of a strong device cache: ED_GPU_SYM_CACHE_GIB when set, else `share` of the
// device's memory and at most `cap_gib` (a fixed 16-24 GiB would fill a 10 GB MIG slice).
[[nodiscard]] inline double device_cache_budget(double share, double cap_gib) {
    constexpr double GiB = 1073741824.0;
    const double set = ed::env::real("ED_GPU_SYM_CACHE_GIB", -1.0);   // -1: not set
    if (set >= 0.0) return set * GiB;
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { cudaGetLastError(); return cap_gib * GiB; }
    return std::min(cap_gib * GiB, share * static_cast<double>(total_b));
}

// Build `make()`; when the device is out of memory, release the strong cache `keep` (what no
// live operator holds is freed with it) and build once more.
template <class Keep, class Make>
auto build_or_evict(Keep& keep, Make&& make) {
    try {
        return make();
    } catch (const std::bad_alloc&) {                  // thrust::system::detail::bad_alloc
        keep.clear();
        cudaGetLastError();
        return make();
    }
}

[[nodiscard]] inline std::shared_ptr<GpuSharedRankTable>
acquire_gpu_shared_rank(
    const std::shared_ptr<const ed::symmetry::SharedRankLookup>& srl)
{
    static std::mutex mtx;
    static std::map<std::uint64_t, std::weak_ptr<GpuSharedRankTable>> registry;
    // Keep-alive FIFO: per-sector GPU mirrors are transient (rebuilt per
    // solve), so a pure weak registry would re-upload the table between
    // consecutive sector solves. A run touches at most a couple of
    // (N, n_up) subspaces, so a tiny strong cache pins the recent tables.
    static std::vector<std::pair<std::uint64_t,
                                 std::shared_ptr<GpuSharedRankTable>>> keep;
    // BYTE-aware eviction: a count cap would pin up to 4 x 36 GB at N >= 34
    // half filling -- device OOM the moment a job touches two subspaces.
    // ED_GPU_SYM_CACHE_GIB, else a quarter of the device's memory (at most 24 GiB), bounds
    // the strong cache; the weak registry still dedups concurrent co-owners.
    static const double kBudgetBytes = device_cache_budget(0.25, 24.0);

    std::lock_guard<std::mutex> lk(mtx);
    for (auto it = registry.begin(); it != registry.end();)   // drop tables nobody holds
        it = it->second.expired() ? registry.erase(it) : std::next(it);
    auto& slot = registry[srl->uid];   // by table identity: a freed table's address can be reused
    if (auto sp = slot.lock()) return sp;
    auto sp = std::make_shared<GpuSharedRankTable>();
    build_or_evict(keep, [&] { sp->d_shared_of_rank = srl->shared_of_rank; return 0; });   // one H2D per (N, n_up)
    if (ed::env::flag("ED_SYM_PROFILE", false)) {
        ED_LOG(Info,
                     "[sym_profile] GPU shared rank table uploaded: "
                     "%zu entries (N=%d, n_up=%d), co-owned by mirrors",
                     srl->shared_of_rank.size(), srl->n_sites, srl->n_up);
    }
    slot = sp;
    keep.emplace_back(srl->uid, sp);
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

// The device snapshot of one sector (no operator). Its policy (basis_view) is the row basis
// of an operator acting into the sector and the column basis of one acting out of it.
struct GpuSectorMirror {
    thrust::device_vector<std::uint64_t>   d_reps;
    thrust::device_vector<double>          d_inv_norms;
    thrust::device_vector<int>             d_perms;
    thrust::device_vector<cuDoubleComplex> d_characters;
    thrust::device_vector<std::uint64_t>   d_flips;   // flip masks
    thrust::device_vector<std::uint64_t>   d_perm_lut; // byte-LUT fast path
    int                                     perm_lut_bpw = 0;
    // Two-level lookup: shared table (co-owned) + per-sector remap.
    std::shared_ptr<GpuSharedRankTable>    shared_rank_tab;
    thrust::device_vector<std::int32_t>    d_local_of_shared;

    int           group_size = 1;
    int           n_sites    = 0;
    int           n_up       = -1;
    std::uint64_t dim        = 0;

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
};

// An operator's row program on the device.
struct GpuProgram {
    thrust::device_vector<std::uint64_t>           d_group_flip, d_vsub_val, d_term_sign;
    thrust::device_vector<int>                     d_group_setbits;
    thrust::device_vector<std::uint32_t>           d_group_vbegin, d_vsub_tbegin;
    thrust::device_vector<thrust::complex<double>> d_term_coeff;

    explicit GpuProgram(const ed::ops::MaskedProgram& rows)
        : d_group_flip(rows.group_flip), d_vsub_val(rows.vsub_val), d_term_sign(rows.term_sign),
          d_group_setbits(rows.group_setbits), d_group_vbegin(rows.group_vbegin),
          d_vsub_tbegin(rows.vsub_tbegin) {
        // std::complex<double> and thrust::complex<double> share their layout
        const auto* c = reinterpret_cast<const thrust::complex<double>*>(rows.term_coeff.data());
        d_term_coeff.assign(c, c + rows.term_coeff.size());
    }

    ed::ops::ProgramView<thrust::complex<double>> view() const noexcept {
        return {static_cast<std::uint32_t>(d_group_flip.size()),
                thrust::raw_pointer_cast(d_group_flip.data()),
                thrust::raw_pointer_cast(d_group_setbits.data()),
                thrust::raw_pointer_cast(d_group_vbegin.data()),
                thrust::raw_pointer_cast(d_vsub_val.data()),
                thrust::raw_pointer_cast(d_vsub_tbegin.data()),
                thrust::raw_pointer_cast(d_term_sign.data()),
                thrust::raw_pointer_cast(d_term_coeff.data())};
    }
};

// O from a column sector to a row sector (the same mirror for an operator on one sector).
struct GpuRows {
    std::shared_ptr<const GpuSectorMirror> row, col;
    bool                                   same;   // one sector: the diagonal needs no lookup
    GpuProgram                             program;
};

namespace detail {

// Build a GpuSectorMirror from a CSR-free RepSectorData (no orbit walk).
inline std::shared_ptr<GpuSectorMirror>
build_sector_mirror(const ed::symmetry::RepSectorData& data)
{
    if (!data.usable()) {
        throw std::runtime_error(
            "build_sector_mirror: RepSectorData is not usable (need n_up >= -1, "
            "non-empty reps, and matching characters / perms sizes)");
    }
    const int n_sites = data.n_sites;
    const int n_up    = data.n_up;
    if (n_sites <= 0 || n_sites > 64 || n_up < -1 || n_up > n_sites) {
        throw std::runtime_error("build_sector_mirror: invalid n_sites / n_up");
    }
    // Sz-parity and full-space sectors (n_up < 0) look states up like every other
    // sector (below): nothing is indexed by the state itself, so N is not capped.

    auto mirror = std::make_shared<GpuSectorMirror>();
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
        dv = std::ldexp(1.0L, n_sites);         // 2^N; a shift would overflow at N = 64
    }
    // Ranks are 64-bit (C(36,18) ~ 9.1e9); only per-sector INDEX values
    // must fit int32 (they index the sector basis, capped below).
    const std::uint64_t dim_full_sz =                       // logged only; 2^64 does not fit
        dv >= 18446744073709551615.0L ? std::numeric_limits<std::uint64_t>::max()
                                      : static_cast<std::uint64_t>(dv + 0.5L);
    if (data.reps.size() > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(
            "build_sector_mirror: sector has more than INT32_MAX representatives; "
            "the reverse-lookup value type would overflow");
    }

    // Device combinadic rank() reads a Pascal triangle from constant memory.
    ed::gpu::combinadic::upload_pascal_shared();

    // Reverse lookup: when the host sector carries the two-level lookup,
    // upload the small per-sector remap and co-own ONE shared rank table per
    // (N, n_up); otherwise the device BINARY SEARCH over the resident sorted
    // ``reps``. No dense per-sector rank table is built (it would cost
    // 2.4 GiB per sector at N=32 and 36 GiB at N=36).
    if (data.has_two_level()) {
        mirror->shared_rank_tab = acquire_gpu_shared_rank(data.shared_rank);
        mirror->d_local_of_shared = data.local_of_shared;
    } else if (ed::env::flag("ED_SYM_PROFILE", false)) {
        ED_LOG(Info,
                     "[sym_profile] GPU rep mirror: binary-search lookup over "
                     "%zu reps (rank space %llu)",
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
    if (data.has_flips()) {   // flip-extended sector
        mirror->d_flips         = data.flip_masks;
    }
    // Byte-LUT permutation fast path: reuse the host-built table
    // when the caller carries one, else build it here from perms_flat --
    // the ~740 KB (N=36, |G|=72) upload avoids a serial n_sites-loop
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

    cuda_check(cudaDeviceSynchronize(), "synchronize after sector mirror upload");
    return mirror;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// An operator's rows on the device: one thread per row of the row sector walks the row
// program (row_walk.h) from its representative, as the host sector_rows.h does -- every
// group's target looked up once in the column sector and conjugated, and on one sector
// (Same) the diagonal without a lookup:
//     out[r] = sum conj(h) in[r] + inv_norm[r] conj(h proj) in[j].
// One walk serves NV vectors, each accumulated in the same order, so a multi-vector apply
// equals NV single applies bit for bit.
// -----------------------------------------------------------------------------
namespace {

using DC = thrust::complex<double>;

template <int NV>
struct WalkPointers {
    const DC* in[NV];
    DC*       out[NV];
};

template <int NV, bool Same>
__global__ void walk_gather(ed::matvec::basis::DeviceRepSymmetryBasisPolicy row,
                            ed::matvec::basis::DeviceRepSymmetryBasisPolicy col,
                            ed::ops::ProgramView<DC> P, WalkPointers<NV> p) {
    const std::uint64_t r = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= row.dim()) return;
    const std::uint64_t s = row.state_of(r);
    const double w = row.inv_norms[r];
    DC acc[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) acc[v] = DC(0.0, 0.0);
    ed::ops::for_each_connection(P, s, [&](std::uint64_t t, const DC& h) {
        DC c;
        std::uint64_t j;
        if (Same && t == s) {
            c = thrust::conj(h);
            j = r;
        } else {
            cuDoubleComplex proj;
            j = (Same ? row : col).index_and_projection(t, proj);
            if (j == ed::matvec::basis::kDeviceNotFound) return;
            c = w * thrust::conj(h * DC(cuCreal(proj), cuCimag(proj)));
        }
#pragma unroll
        for (int v = 0; v < NV; ++v) acc[v] += c * p.in[v][j];
    });
#pragma unroll
    for (int v = 0; v < NV; ++v) p.out[v][r] = acc[v];
}

}  // namespace

// out[i] = O in[i] for i < k (device pointers), in launches of up to 8 vectors.
void launch_walk(const GpuRows& op, const DC* const* ins, DC* const* outs, std::size_t k)
{
    using detail::cuda_check;
    const std::uint64_t dim = op.row->dim;
    if (dim == 0 || k == 0) return;
    const auto row  = op.row->basis_view();
    const auto col  = op.col->basis_view();
    const auto prog = op.program.view();
    constexpr unsigned kThreads = 256;
    const auto blocks = static_cast<unsigned>((dim + kThreads - 1) / kThreads);
    auto launch = [&](auto nv_tag, std::size_t off) {
        constexpr int NV = decltype(nv_tag)::value;
        WalkPointers<NV> p;
        for (int v = 0; v < NV; ++v) { p.in[v] = ins[off + v]; p.out[v] = outs[off + v]; }
        if (op.same) walk_gather<NV, true><<<blocks, kThreads>>>(row, col, prog, p);
        else         walk_gather<NV, false><<<blocks, kThreads>>>(row, col, prog, p);
    };
    std::size_t off = 0;
    while (off < k) {
        const std::size_t left = k - off;
        if (left >= 8)      { launch(std::integral_constant<int, 8>{}, off); off += 8; }
        else if (left >= 4) { launch(std::integral_constant<int, 4>{}, off); off += 4; }
        else if (left >= 2) { launch(std::integral_constant<int, 2>{}, off); off += 2; }
        else                { launch(std::integral_constant<int, 1>{}, off); off += 1; }
        cuda_check(cudaGetLastError(), "walk kernel launch");
    }
}

}  // namespace ed::symmetry::gpu_mirror

// =============================================================================
// The factories: the resident sector mirrors (shared, content-keyed) plus the operator's
// program, uploaded per bind (a few KiB).
// =============================================================================
namespace ed::symmetry::gpu_mirror::detail {

// The resident device mirror of a sector, shared by every operator bound on it.
std::shared_ptr<const GpuSectorMirror> acquire_sector_mirror(const ed::symmetry::RepSectorData& rep)
{
    // Sector operators are transient, so a destroyed sector's address is reused by the next
    // one: the cache is keyed on the sector's own content -- the per-sector characters (unique
    // per irrep), the rep-list signature (n_up / size / samples), the group action -- and
    // each entry carries the full fingerprint, so reuse never depends on hash quality.
    // Doubles enter by their exact bit pattern.
    auto bits = [](double x) {
        std::uint64_t u;
        std::memcpy(&u, &x, sizeof u);
        return u;
    };
    auto content_key = [&bits](const ed::symmetry::RepSectorData& r) {
        std::uint64_t h = 1469598103934665603ULL;
        // Avalanche every word (splitmix64 finalizer) BEFORE the FNV fold: on a Z8 ring,
        // chi_{k+4}(g) = (-1)^g chi_k(g) would otherwise hash like chi_k.
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
        for (const auto& c : r.characters) {
            mix(bits(c.real()));
            mix(bits(c.imag()));
        }
        if (!r.flip_masks.empty()) mix(r.flip_masks.front());
        return h;
    };
    // The reps list is determined by (n_up window, group action, characters) and
    // cross-checked by its (size, front, mid, back) signature.
    struct MirrorSlot {
        int                                     n_up;
        std::uint64_t                           reps_sig[4];
        std::vector<std::complex<double>>       chi;
        std::vector<int>                        perms;
        std::vector<std::uint64_t>              flips;
        std::weak_ptr<const GpuSectorMirror>    mirror;
    };
    auto signature = [](const ed::symmetry::RepSectorData& r, std::uint64_t (&sig)[4]) {
        sig[0] = r.reps.size();
        sig[1] = r.reps.empty() ? 0 : r.reps.front();
        sig[2] = r.reps.empty() ? 0 : r.reps[r.reps.size() / 2];
        sig[3] = r.reps.empty() ? 0 : r.reps.back();
    };
    std::uint64_t sig[4];
    signature(rep, sig);
    auto matches = [&](const MirrorSlot& s) {
        return s.n_up == rep.n_up && std::equal(sig, sig + 4, s.reps_sig) && s.chi == rep.characters
            && s.perms == rep.perms_flat && s.flips == rep.flip_masks;
    };

    static std::mutex mtx;
    static std::map<std::uint64_t, std::vector<MirrorSlot>> registry;
    static std::vector<std::shared_ptr<const GpuSectorMirror>> keep;
    // Byte-aware strong cache (a count cap would pin ~4 x 4 GB of sector arrays at N=36):
    // ED_GPU_SYM_CACHE_GIB, else 15% of the device's memory (at most 16 GiB).
    static const double kKeepBudget = device_cache_budget(0.15, 16.0);
    std::lock_guard<std::mutex> lk(mtx);
    auto& bucket = registry[content_key(rep)];
    for (auto it = bucket.begin(); it != bucket.end();) {
        auto locked = it->mirror.lock();
        if (!locked) { it = bucket.erase(it); continue; }   // expired
        if (matches(*it)) return locked;
        ++it;
    }
    std::shared_ptr<const GpuSectorMirror> mirror = build_or_evict(keep, [&] { return build_sector_mirror(rep); });
    MirrorSlot s;
    s.n_up   = rep.n_up;
    std::copy(sig, sig + 4, s.reps_sig);
    s.chi    = rep.characters;
    s.perms  = rep.perms_flat;
    s.flips  = rep.flip_masks;
    s.mirror = mirror;
    bucket.push_back(std::move(s));
    keep.push_back(mirror);
    auto bytes_of = [](const std::shared_ptr<const GpuSectorMirror>& mm) {
        return static_cast<double>(mm->d_reps.size() * 8 + mm->d_inv_norms.size() * 8 + mm->d_perm_lut.size() * 8
                                   + mm->d_perms.size() * 4 + mm->d_local_of_shared.size() * 4);
    };
    double total = 0.0;
    for (const auto& mm : keep) total += bytes_of(mm);
    while (keep.size() > 1 && total > kKeepBudget) {
        total -= bytes_of(keep.front());
        keep.erase(keep.begin());
    }
    return mirror;
}

// O from the column sector to the row sector on the device.
std::shared_ptr<const GpuRows> make_rows(const ed::symmetry::RepSectorData& row,
                                         const ed::symmetry::RepSectorData& col,
                                         const ed::ops::MaskedProgram& rows) {
    const bool same = &row == &col;
    auto r_mirror = acquire_sector_mirror(row);
    auto c_mirror = same ? r_mirror : acquire_sector_mirror(col);
    auto out = std::make_shared<const GpuRows>(GpuRows{std::move(r_mirror), std::move(c_mirror), same, GpuProgram(rows)});
    cuda_check(cudaDeviceSynchronize(), "synchronize after program upload");
    return out;
}

ed::LinearOperator::MatvecFn single(std::shared_ptr<const GpuRows> op, const char* who) {
    using DC = thrust::complex<double>;
    return [op, who](const ed::matvec::Complex* in, ed::matvec::Complex* out, std::size_t n) {
        if (n != op->row->dim)
            throw std::runtime_error(std::string(who) + ": output length " + std::to_string(n) + " != rows "
                                     + std::to_string(op->row->dim));
        const DC* ins[1] = {reinterpret_cast<const DC*>(in)};
        DC* outs[1] = {reinterpret_cast<DC*>(out)};
        launch_walk(*op, ins, outs, 1);
    };
}

}  // namespace ed::symmetry::gpu_mirror::detail

ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep(const ed::symmetry::RepSectorData& rep,
                                         const ed::ops::MaskedProgram&      rows)
{
    namespace gd = ed::symmetry::gpu_mirror::detail;
    return gd::single(gd::make_rows(rep, rep, rows), "make_sector_matvec_gpu_rep");
}

ed::LinearOperator::MatvecFn
ed::symmetry::make_cross_matvec_gpu_rep(const ed::symmetry::RepSectorData& src,
                                        const ed::symmetry::RepSectorData& tgt,
                                        const ed::ops::MaskedProgram&      rows)
{
    namespace gd = ed::symmetry::gpu_mirror::detail;
    return gd::single(gd::make_rows(tgt, src, rows), "make_cross_matvec_gpu_rep");
}

// k vectors per call through the multi-vector gather (one row walk serves up to 8 of them).
ed::LinearOperator::MultiMatvecFn
ed::symmetry::make_sector_matvec_gpu_rep_multi(const ed::symmetry::RepSectorData& rep,
                                               const ed::ops::MaskedProgram&      rows)
{
    using DC = thrust::complex<double>;
    const auto op = ed::symmetry::gpu_mirror::detail::make_rows(rep, rep, rows);
    return [op](const ed::matvec::Complex* const* ins, ed::matvec::Complex* const* outs,
                std::size_t n, std::size_t k) {
        if (n != op->row->dim)
            throw std::runtime_error("ed::symmetry::make_sector_matvec_gpu_rep_multi: size mismatch (" +
                                     std::to_string(n) + " vs " + std::to_string(op->row->dim) + ")");
        ed::symmetry::gpu_mirror::launch_walk(*op, reinterpret_cast<const DC* const*>(ins),
                                              reinterpret_cast<DC* const*>(outs), k);
    };
}
// ---------------------------------------------------------------------------
// Host-pointer twin:// ---------------------------------------------------------------------------
// Host-pointer twin: persistent device staging buffers around the resident
// mirror, one H2D + D2H per apply. Built for the little-group engine's CPU
// Lanczos (host vectors); the staging traffic is O(dim) against the kernel's
// O(dim * terms * |G|) walk.
// ---------------------------------------------------------------------------
// Per-apply staging accounting for the host-pointer twin, under
// ED_SYM_PROFILE only. Both the clock reads AND the extra device sync (without
// it the blocking D2H absorbs the kernel and the split is a lie) sit behind
// the gate, so an unprofiled run is untouched. ONE summary when the last copy
// of the returned lambda dies -- a frontier star runs thousands of applies and
// a per-apply line would bury every other signal in the log.
namespace {

struct HostPtrStagingProfile {
    bool          on    = false;
    std::uint64_t calls = 0;
    std::size_t   dim   = 0;
    double t_h2d = 0, t_kernel = 0, t_d2h = 0;

    ~HostPtrStagingProfile() {
        if (!on || calls == 0) return;
        const double pcie = t_h2d + t_d2h;
        const double tot  = pcie + t_kernel;
        ED_LOG(Info,
            "[sym_profile] hostptr rep matvec dim=%zu applies=%llu: "
            "H2D=%.3fs kernel=%.3fs D2H=%.3fs (staging %.1f%% of %.3fs)",
            dim, static_cast<unsigned long long>(calls),
            t_h2d, t_kernel, t_d2h,
            tot > 0.0 ? 100.0 * pcie / tot : 0.0, tot);
    }
};

}  // namespace

ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep_hostptr(
    const ed::symmetry::RepSectorData& rep,
    const ed::ops::MaskedProgram&      rows)
{
    using ed::symmetry::gpu_mirror::detail::cuda_check;

    auto dev_fn = ed::symmetry::make_sector_matvec_gpu_rep(rep, rows);
    auto d_in   = std::make_shared<thrust::device_vector<cuDoubleComplex>>();
    auto d_out  = std::make_shared<thrust::device_vector<cuDoubleComplex>>();
    auto prof   = std::make_shared<HostPtrStagingProfile>();
    prof->on = ed::env::flag("ED_SYM_PROFILE", false);

    return [dev_fn, d_in, d_out, prof](const ed::matvec::Complex* in,
                                       ed::matvec::Complex*       out,
                                       std::size_t                n) {
        using Clock = std::chrono::steady_clock;
        const bool p = prof->on;
        auto stamp = [p] { return p ? Clock::now() : Clock::time_point{}; };
        auto secs  = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        if (d_in->size() != n) {
            d_in->resize(n);
            d_out->resize(n);
        }
        const auto t0 = stamp();
        cuda_check(cudaMemcpy(thrust::raw_pointer_cast(d_in->data()), in,
                              n * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice),
                   "hostptr rep matvec H2D");
        const auto t1 = stamp();
        dev_fn(reinterpret_cast<const ed::matvec::Complex*>(
                   thrust::raw_pointer_cast(d_in->data())),
               reinterpret_cast<ed::matvec::Complex*>(
                   thrust::raw_pointer_cast(d_out->data())),
               n);
        if (p)
            cuda_check(cudaDeviceSynchronize(),
                       "hostptr rep matvec profile sync");
        const auto t2 = stamp();
        cuda_check(cudaMemcpy(out, thrust::raw_pointer_cast(d_out->data()),
                              n * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost),
                   "hostptr rep matvec D2H");
        const auto t3 = stamp();
        if (p) {
            ++prof->calls;
            prof->dim       = n;
            prof->t_h2d    += secs(t0, t1);
            prof->t_kernel += secs(t1, t2);
            prof->t_d2h    += secs(t2, t3);
        }
    };
}

#endif  // WITH_CUDA
