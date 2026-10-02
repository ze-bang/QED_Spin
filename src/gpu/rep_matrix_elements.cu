// =============================================================================
// src/gpu/rep_matrix_elements.cu -- device sweep behind
// rep_matrix_elements(..., {.use_gpu = true}).
//
// Same arithmetic as the CPU sweep (src/ops/program.cpp), one thread per KET
// representative (grid-stride). Per-(pair, observable) accumulators live in shared
// memory; each block writes its partial sums, and the host adds the block partials in
// block order. Target lookups use DeviceRepSymmetryBasisPolicy::index_and_projection,
// the same device routine the Hamiltonian gather kernels use, with the binary search
// over the resident sorted reps (the production path at N = 36).
//
// Memory: both sectors (reps + inv_norms, 16 B per rep each; one copy when src == tgt),
// the program, and the distinct ket/bra vectors of a pair batch (16 B per rep each).
// Pairs are batched so the vectors of one batch fit in the free device memory.
// =============================================================================
#include <ed/ops/program.h>

#include <ed/gpu/device_basis_policy.cuh>
#include <ed/basis/rep_sector.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace ed::ops {

namespace {

using Complex = std::complex<double>;
using DevPolicy = ed::matvec::basis::DeviceRepSymmetryBasisPolicy;

constexpr int kMaxPairs = 16;       // pairs per launch (per-thread registers)
constexpr int kThreads = 256;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("rep_matrix_elements_gpu: ") + what + ": "
                                 + cudaGetErrorString(e));
}

struct DevSector {
    thrust::device_vector<std::uint64_t> reps;
    thrust::device_vector<double> inv_norms;
    thrust::device_vector<int> perms;
    thrust::device_vector<cuDoubleComplex> chars;
    thrust::device_vector<std::uint64_t> flips;
    thrust::device_vector<std::uint64_t> lut;
    int bpw = 0;
    DevPolicy pol{};

    explicit DevSector(const ed::symmetry::RepSectorData& rd) {
        reps = rd.reps;
        inv_norms = rd.inv_norms;
        perms = rd.perms_flat;
        std::vector<cuDoubleComplex> h(rd.characters.size());
        for (std::size_t g = 0; g < h.size(); ++g)
            h[g] = make_cuDoubleComplex(rd.characters[g].real(), rd.characters[g].imag());
        chars = h;
        if (rd.has_flips()) flips = rd.flip_masks;
        if (!rd.perm_lut_data.empty()) {
            lut = rd.perm_lut_data;
            bpw = rd.perm_lut_bpw;
        } else {
            ed::symmetry::RepSectorData tmp;
            tmp.n_sites = rd.n_sites;
            tmp.group_size = rd.group_size;
            tmp.perms_flat = rd.perms_flat;
            tmp.build_perm_lut();
            lut = tmp.perm_lut_data;
            bpw = tmp.perm_lut_bpw;
        }
        pol.reps = thrust::raw_pointer_cast(reps.data());
        pol.inv_norms = thrust::raw_pointer_cast(inv_norms.data());
        pol.perms = thrust::raw_pointer_cast(perms.data());
        pol.characters = thrust::raw_pointer_cast(chars.data());
        pol.flips = flips.empty() ? nullptr : thrust::raw_pointer_cast(flips.data());
        pol.perm_lut = lut.empty() ? nullptr : thrust::raw_pointer_cast(lut.data());
        pol.perm_lut_bpw = bpw;
        pol.dim_ = rd.dim();
        pol.group_size = rd.group_size;
        pol.n_sites = rd.n_sites;
        pol.n_up = rd.n_up;
        // rep_index_of_rank / shared_rank_of stay null: binary search over reps
    }
};

struct ProgView {
    const std::uint64_t* group_flip;
    const int* group_setbits;
    const std::uint32_t* group_vbegin;
    const std::uint64_t* vsub_val;
    const std::uint32_t* vsub_tbegin;
    const std::uint64_t* term_sign;
    const cuDoubleComplex* term_coeff;
    const std::uint32_t* term_obs;
    int n_groups;
};

struct DevProgram {
    thrust::device_vector<std::uint64_t> group_flip, vsub_val, term_sign;
    thrust::device_vector<int> group_setbits;
    thrust::device_vector<std::uint32_t> group_vbegin, vsub_tbegin, term_obs;
    thrust::device_vector<cuDoubleComplex> term_coeff;

    explicit DevProgram(const MaskedProgram& P) {
        group_flip = P.group_flip;
        group_setbits = P.group_setbits;
        group_vbegin = P.group_vbegin;
        vsub_val = P.vsub_val;
        vsub_tbegin = P.vsub_tbegin;
        term_sign = P.term_sign;
        term_obs = P.term_obs;
        std::vector<cuDoubleComplex> h(P.term_coeff.size());
        for (std::size_t k = 0; k < h.size(); ++k)
            h[k] = make_cuDoubleComplex(P.term_coeff[k].real(), P.term_coeff[k].imag());
        term_coeff = h;
    }
    ProgView view() const {
        return ProgView{thrust::raw_pointer_cast(group_flip.data()),
                        thrust::raw_pointer_cast(group_setbits.data()),
                        thrust::raw_pointer_cast(group_vbegin.data()),
                        thrust::raw_pointer_cast(vsub_val.data()),
                        thrust::raw_pointer_cast(vsub_tbegin.data()),
                        thrust::raw_pointer_cast(term_sign.data()),
                        thrust::raw_pointer_cast(term_coeff.data()),
                        thrust::raw_pointer_cast(term_obs.data()),
                        static_cast<int>(group_flip.size())};
    }
};

// Constraint projector masks (RepMEOptions::balanced_masks), passed by value.
constexpr int kMaxBalanced = 64;
struct Balanced {
    std::uint64_t m[kMaxBalanced];
    int n;
};
__device__ inline bool balanced(std::uint64_t s, const Balanced& b) {
    for (int i = 0; i < b.n; ++i)
        if (2 * __popcll(s & b.m[i]) != __popcll(b.m[i])) return false;
    return true;
}

struct PairPtrs {
    const cuDoubleComplex* ket[kMaxPairs];
    const cuDoubleComplex* bra[kMaxPairs];
};

__device__ inline cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

// shared layout: acc[(p * n_loc + (o - obs_lo)) * 2 + {0: re, 1: im}]
__global__ void me_kernel(DevPolicy src, DevPolicy tgt, bool same_sector, ProgView prog,
                          PairPtrs pp, Balanced bal, int n_pairs, int obs_lo, int obs_hi,
                          double* __restrict__ block_out) {
    extern __shared__ double acc[];
    const int n_loc = obs_hi - obs_lo;
    const int n_acc = 2 * n_pairs * n_loc;
    for (int i = threadIdx.x; i < n_acc; i += blockDim.x) acc[i] = 0.0;
    __syncthreads();

    const std::uint64_t dim = src.dim_;
    for (std::uint64_t r = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; r < dim;
         r += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        cuDoubleComplex bk[kMaxPairs];
        bool any = false;
        for (int p = 0; p < n_pairs; ++p) {
            bk[p] = pp.ket[p][r];
            any = any || bk[p].x != 0.0 || bk[p].y != 0.0;
        }
        if (!any) continue;
        const std::uint64_t s = src.reps[r];
        if (!balanced(s, bal)) continue;                 // P on the ket
        const double w = src.inv_norms[r];
        for (int gi = 0; gi < prog.n_groups; ++gi) {
            const std::uint64_t F = prog.group_flip[gi];
            const std::uint64_t v = s & F;
            if (__popcll(v) != prog.group_setbits[gi]) continue;
            std::uint32_t lo = prog.group_vbegin[gi], hi = prog.group_vbegin[gi + 1];
            while (lo < hi) {
                const std::uint32_t mid = lo + ((hi - lo) >> 1);
                if (prog.vsub_val[mid] < v) lo = mid + 1; else hi = mid;
            }
            if (lo == prog.group_vbegin[gi + 1] || prog.vsub_val[lo] != v) continue;
            const std::uint32_t vi = lo;

            cuDoubleComplex proj;
            std::uint64_t j;
            if (same_sector && F == 0) {
                j = r;
                proj = make_cuDoubleComplex(1.0 / w, 0.0);
            } else {
                if (!balanced(s ^ F, bal)) continue;         // P on the bra
                j = tgt.index_and_projection(s ^ F, proj);
                if (j == ed::matvec::basis::kDeviceNotFound) continue;
            }
            const cuDoubleComplex base = make_cuDoubleComplex(proj.x * w, proj.y * w);
            cuDoubleComplex y[kMaxPairs];
            for (int p = 0; p < n_pairs; ++p)
                y[p] = cmul(cmul(base, bk[p]), cuConj(pp.bra[p][j]));
            for (std::uint32_t k = prog.vsub_tbegin[vi]; k < prog.vsub_tbegin[vi + 1]; ++k) {
                const int o = static_cast<int>(prog.term_obs[k]);
                if (o < obs_lo || o >= obs_hi) continue;
                cuDoubleComplex c = prog.term_coeff[k];
                if (__popcll(s & prog.term_sign[k]) & 1) { c.x = -c.x; c.y = -c.y; }
                for (int p = 0; p < n_pairs; ++p) {
                    const cuDoubleComplex t = cmul(c, y[p]);
                    const int a = 2 * (p * n_loc + (o - obs_lo));
                    atomicAdd(&acc[a], t.x);
                    atomicAdd(&acc[a + 1], t.y);
                }
            }
        }
    }
    __syncthreads();
    double* out = block_out + static_cast<std::size_t>(blockIdx.x) * n_acc;
    for (int i = threadIdx.x; i < n_acc; i += blockDim.x) out[i] = acc[i];
}

}  // namespace

std::vector<Complex>
rep_matrix_elements_gpu(const ed::symmetry::RepSectorData& src,
                        const ed::symmetry::RepSectorData& tgt,
                        const MaskedProgram& prog,
                        const std::vector<RepVectorView>& kets,
                        const std::vector<RepVectorView>& bras,
                        const std::vector<std::pair<int, int>>& pairs,
                        const RepMEOptions& opt) {
    const std::size_t n_obs = static_cast<std::size_t>(prog.n_obs);
    std::vector<Complex> out(pairs.size() * n_obs, Complex(0.0, 0.0));

    const bool same_sector = (&src == &tgt);
    const DevSector dsrc(src);
    std::unique_ptr<DevSector> dtgt_own;
    if (!same_sector) dtgt_own = std::make_unique<DevSector>(tgt);
    const DevPolicy tpol = same_sector ? dsrc.pol : dtgt_own->pol;
    const DevProgram dprog(prog);
    if (opt.balanced_masks.size() > static_cast<std::size_t>(kMaxBalanced))
        throw std::invalid_argument("rep_matrix_elements_gpu: more than 64 balanced masks");
    Balanced bal{};
    bal.n = static_cast<int>(opt.balanced_masks.size());
    for (int i = 0; i < bal.n; ++i) bal.m[i] = opt.balanced_masks[static_cast<std::size_t>(i)];

    int dev = 0, n_sm = 0;
    ck(cudaGetDevice(&dev), "cudaGetDevice");
    ck(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev), "SM count");
    const int max_shared = static_cast<int>(opt.gpu_shared_bytes);
    ck(cudaFuncSetAttribute(me_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared),
       "shared-memory attribute");
    const std::uint64_t dim = src.dim();
    const int n_blocks = static_cast<int>(std::max<std::uint64_t>(
        1, std::min<std::uint64_t>((dim + kThreads - 1) / kThreads,
                                   static_cast<std::uint64_t>(n_sm) * 8)));

    // Pair batches: at most kMaxPairs pairs, and distinct vectors that fit in free memory.
    std::size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const std::size_t vec_bytes = std::max(src.dim(), tgt.dim()) * sizeof(cuDoubleComplex);
    const std::size_t budget = static_cast<std::size_t>(0.85 * static_cast<double>(free_b));
    const std::size_t max_vecs = vec_bytes ? budget / vec_bytes : 2 * kMaxPairs;
    if (max_vecs < 2)
        throw std::runtime_error("rep_matrix_elements_gpu: one ket + one bra do not fit on the device");

    std::size_t p0 = 0;
    while (p0 < pairs.size()) {
        // grow the batch
        std::map<const void*, int> slot;   // host pointer -> device slot
        std::vector<const Complex*> host_vecs;
        std::vector<std::size_t> host_len;
        std::size_t p1 = p0;
        while (p1 < pairs.size() && p1 - p0 < static_cast<std::size_t>(kMaxPairs)) {
            const auto& bv = bras[static_cast<std::size_t>(pairs[p1].first)];
            const auto& kv = kets[static_cast<std::size_t>(pairs[p1].second)];
            const std::size_t extra = (slot.count(bv.data) ? 0 : 1)
                                    + ((slot.count(kv.data) || kv.data == bv.data) ? 0 : 1);
            if (host_vecs.size() + extra > max_vecs) break;
            for (const RepVectorView* v : {&bv, &kv})
                if (!slot.count(v->data)) {
                    slot[v->data] = static_cast<int>(host_vecs.size());
                    host_vecs.push_back(v->data);
                    host_len.push_back(v->size);
                }
            ++p1;
        }
        std::vector<thrust::device_vector<cuDoubleComplex>> dvec(host_vecs.size());
        for (std::size_t i = 0; i < host_vecs.size(); ++i) {
            dvec[i].resize(host_len[i]);
            ck(cudaMemcpy(thrust::raw_pointer_cast(dvec[i].data()), host_vecs[i],
                          host_len[i] * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice),
               "upload vector");
        }
        PairPtrs pp{};
        const int n_pairs = static_cast<int>(p1 - p0);
        for (int p = 0; p < n_pairs; ++p) {
            const auto& pr = pairs[p0 + static_cast<std::size_t>(p)];
            pp.bra[p] = thrust::raw_pointer_cast(
                dvec[static_cast<std::size_t>(slot[bras[static_cast<std::size_t>(pr.first)].data])].data());
            pp.ket[p] = thrust::raw_pointer_cast(
                dvec[static_cast<std::size_t>(slot[kets[static_cast<std::size_t>(pr.second)].data])].data());
        }

        // observable chunks that fit the shared-memory budget
        const int per_obs = 2 * n_pairs * static_cast<int>(sizeof(double));
        const int chunk = std::max(1, max_shared / per_obs);
        for (int o0 = 0; o0 < prog.n_obs; o0 += chunk) {
            const int o1 = std::min(prog.n_obs, o0 + chunk);
            const int n_acc = 2 * n_pairs * (o1 - o0);
            thrust::device_vector<double> block_out(static_cast<std::size_t>(n_blocks) * n_acc);
            me_kernel<<<n_blocks, kThreads, static_cast<std::size_t>(n_acc) * sizeof(double)>>>(
                dsrc.pol, tpol, same_sector, dprog.view(), pp, bal, n_pairs, o0, o1,
                thrust::raw_pointer_cast(block_out.data()));
            ck(cudaGetLastError(), "kernel launch");
            ck(cudaDeviceSynchronize(), "kernel");
            std::vector<double> h(block_out.size());
            ck(cudaMemcpy(h.data(), thrust::raw_pointer_cast(block_out.data()),
                          h.size() * sizeof(double), cudaMemcpyDeviceToHost), "download partials");
            for (int b = 0; b < n_blocks; ++b)
                for (int p = 0; p < n_pairs; ++p)
                    for (int o = o0; o < o1; ++o) {
                        const std::size_t a = static_cast<std::size_t>(b) * n_acc
                                            + 2 * static_cast<std::size_t>(p * (o1 - o0) + (o - o0));
                        out[(p0 + static_cast<std::size_t>(p)) * n_obs + static_cast<std::size_t>(o)]
                            += Complex(h[a], h[a + 1]);
                    }
        }
        p0 = p1;
    }
    return out;
}

bool rep_matrix_elements_gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

}  // namespace ed::ops
