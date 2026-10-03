#pragma once
// =============================================================================
// src/engine/walk.h -- the block walk shared by the sector-resolved
// verbs (eigs.cpp, thermal.cpp, dynamics.cpp, expect.cpp): engine options for one subspace, and the star-by-star walk.
// Private to the little-group engine.
// =============================================================================

#include "internal.h"

#include <ed/core/footprint.h>
#include <ed/core/memory.h>
#include <ed/sectors/sectors.h>
#include <ed/basis/su2_dims.h>
#include <ed/core/interrupt.h>
#include <ed/ops/casimir.h>
#include <ed/ops/casimir_projector.h>

#include <algorithm>
#include <array>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

namespace ed::sectors::detail {

inline std::vector<Perm> abelian_or_identity(const Spec& s, int n_sites) {
    if (!s.abelian.empty()) return s.abelian;
    Perm id(static_cast<std::size_t>(n_sites));
    std::iota(id.begin(), id.end(), 0);
    return {id};
}

/// Every supplied site permutation (the abelian generators and the residues) commutes with H and
/// every residue normalises the abelian group, else InvalidRequest (eigs.cpp). Each verb runs it
/// on the Spec as given, before any folding drops part of it.
void require_symmetries(const ::Operator& H, const Spec& s);

/// The walk options of one subspace.
inline ed::solvers::LittleGroupOptions engine_options(const Spec& s, const Subspace& sub) {
    ed::solvers::LittleGroupOptions o;
    o.n_up = sub.n_up;
    o.sz_parity = sub.sz_parity;
    // subspaces() already enforced 'require' against H. Inside a subspace the engine engages
    // the flip where the subspace is its own image (n_up = N/2, a parity half with N even, the
    // full space) and not elsewhere: an Sz = S tower, an explicit n_up != N/2 or a mirror pair
    // (which gets the symmetry through the fold) -- 'auto' there, never a refusal.
    o.spin_flip = s.spin_flip == 1 ? -1 : s.spin_flip;
    o.time_reversal = s.time_reversal;
    o.only_k0 = s.only_k0;
    o.only_irrep = s.only_irrep;
    o.only_irrep_chars = s.only_irrep_chars;
    return o;
}

/// The operator one block is solved with: its H, and with a total-spin restriction the block's spin-S
/// `tower` (internal.h, Tower: the lanes run the bare H from starts inside it and certify what they
/// return) with the Lowdin `projector` onto it for the sampled lanes' Gaussian starts.
/// `multiplicity` includes the 2S + 1 members of each multiplet. A null `op` means the block holds no
/// state of the requested spin.
struct BlockOp {
    std::shared_ptr<const ed::LinearOperator> op;
    std::shared_ptr<const ed::solvers::lg_detail::Tower> tower;   ///< the spin-S tower (SU(2) only)
    std::uint64_t multiplicity = 1;
    std::shared_ptr<const ed::symmetry::LowdinS2Projector> projector;   ///< onto the tower (SU(2) only)
};

/// The block budget of a block's reduced CSRs (csr_policy.h): block_csr_budget_bytes less the
/// `working_set` its solver will hold, sized now -- before the solver allocates anything.
inline std::shared_ptr<ed::planner::CsrBudget> block_budget(std::uint64_t working_set) {
    return std::make_shared<ed::planner::CsrBudget>(ed::planner::block_csr_budget_bytes(working_set));
}

inline BlockOp block_operator(const Spec& s, int n_sites, const Subspace& sub,
                              const ed::solvers::lg_detail::StarBuild& sb,
                              const std::shared_ptr<ed::solvers::BlockData>& bi,
                              const std::shared_ptr<::Operator>& s2_carrier, Device device = Device::Cpu,
                              const std::shared_ptr<ed::planner::CsrBudget>& budget = nullptr) {
    using namespace ed::solvers::lg_detail;
    BlockOp b;
    b.op = std::shared_ptr<const ed::LinearOperator>(bi, &block_mv(*bi));
    b.multiplicity = bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
    // H's reduced CSR first, then S^2's, from the block's budget.
    RepSectorMatVec* rep = bi->gop ? bi->gop.get() : sb.hk.get();
    if (budget) rep->set_csr_budget(budget);
    const bool dev = device != Device::Cpu;
    if (s.two_S < 0) {
        if (dev) rep->enable_device(true);   // its device kernel: op->has_device_kernel()
        return b;
    }
    const auto towers = ed::symmetry::allowed_two_S_in_block(n_sites, sub.n_up, sub.sz_parity, bi->tag.flip_parity);
    if (std::find(towers.begin(), towers.end(), s.two_S) == towers.end()) {
        b.op.reset();                   // this flip-parity block holds no spin-S state
        return b;
    }
    // The block's spin-S states, by Burnside.
    const std::shared_ptr<const ed::symmetry::RepSectorData> sec = bi->gop ? bi->gsec : sb.hk->rep_data_ptr();
    auto t = std::make_shared<Tower>();
    t->sector = sec;
    t->two_S = s.two_S;
    t->towers = towers;
    t->dim = tower_dimension(*sec, s.two_S);
    if (t->dim == 0) {
        b.op.reset();
        return b;
    }
    // S^2 on the block. On the host S- S+ + Sz(Sz + 1) through the sector one up spin higher (P6.5):
    // ~N entries a row against ~N^2/4 for the S^2 carrier; a spin-flip sector (n_up + 1 is not its own
    // flip image) through the sector without the flip (FlipLadderS2). The carrier stays for the device
    // lane (its kernel: the penalty of the fallback solve runs there) and for a flip sector of an irrep
    // of dimension > 1; certifying a level or two walks it, a second apply builds its CSR.
    if (!dev && sub.n_up >= 0 && !sec->has_flips()) {
        t->s2 = std::make_shared<LadderS2>(sec);
    } else if (!dev && FlipLadderS2::fits(*sec)) {
        t->s2 = std::make_shared<FlipLadderS2>(sec);
    } else {
        auto s2rep = std::make_shared<RepSectorMatVec>(*s2_carrier, sec);
        s2rep->set_csr_budget(budget);
        s2rep->defer_csr(1);
        if (dev) s2rep->enable_device(true);
        t->s2 = s2rep;
    }
    if (dev) rep->enable_device(true);
    b.multiplicity *= static_cast<std::uint64_t>(sub.members);
    // The sampled lanes start from P_S of a Gaussian (isotropic in the tower: an unbiased trace).
    b.projector = std::make_shared<ed::symmetry::LowdinS2Projector>(t->s2, s.two_S, towers);
    b.tower = std::move(t);
    return b;
}

/// The antiunitary pairing of a block's levels: its star fold's map (`star_tr`, the walk context's
/// tr: K or Theta), or Theta when the subspace's mirror is the time-reversal image.
inline Antiunitary fold_of(Antiunitary star_tr, const Subspace& sub, const ed::solvers::LittleGroupBlockTag& tag) {
    if (tag.tr_folded) return star_tr;
    return sub.mirror == 2 && sub.theta ? Antiunitary::Theta : Antiunitary::None;
}

/// The antiunitary map that folded any of `levels` (fold_of: a folded star, or a subspace Theta
/// mirrors), None when no level was folded -- also when the context engaged K or Theta but no
/// star or subspace it serves made it into the levels.
inline Antiunitary folded_map(const std::vector<Level>& levels) {
    Antiunitary seen = Antiunitary::None;
    for (const Level& L : levels)
        if (L.fold != Antiunitary::None) seen = L.fold;
    return seen;
}

/// Whether a level's mirror is its Theta image (else the spin flip's, when it has one).
inline bool theta_mirror(const Level& L) { return L.mirror == 2 && L.fold == Antiunitary::Theta && !L.tag.tr_folded; }

/// The S^z members a level stands for: 2S + 1 for a whole SU(2) multiplet solved at its Sz = S
/// member, else 1 (H in a uniform field: every member is a level of its own).
inline std::uint64_t members(const Level& L) {
    const std::uint64_t base = L.tag.multiplicity * static_cast<std::uint64_t>(L.mirror);
    return base > 0 ? L.multiplicity / base : 1;
}

/// The states a total-spin restriction over `subs` holds: the members of every multiplet.
inline std::uint64_t tower_states(const std::vector<Subspace>& subs, int n_sites, int two_S) {
    std::uint64_t members = 0;
    for (const Subspace& sub : subs) members += static_cast<std::uint64_t>(sub.members);
    return members * ed::symmetry::multiplet_count(n_sites, two_S);
}

/// Dense spectra of many blocks, materialised as the walk visits them (the walk streams stars, so
/// the matrices are the only thing that outlives a star). place(Task::DenseBatch) chooses each
/// entry's lane (under 'auto' the host below kDeviceDenseMinDim). On a device lane they are solved
/// in batched cuSOLVER calls, a real block in real arithmetic, a sector operator's block written
/// in place from its CSR: a batch is packed on the host and uploaded at once, so it is solved
/// before it outgrows a quarter of the free device memory or of the RAM the job may still allocate
/// (cuSOLVER's workspace and the eigenvalues come on top), and never holds more than 2 GiB of
/// matrices; a block larger than that is solved by itself on the device when it fits there
/// (fits_device_alone). Under device='auto' a block that does not fit, and a batch whose device
/// solve fails, are solved on the host; under 'gpu' the block raises ResourceLimit and the failed
/// batch is retried in halves on the device (solve_device_split). The last batch and the host
/// queue are solved at once. On the host every block is queued (up to
/// host_budget()) and the queue solved concurrently, one serial LAPACK call per thread, largest
/// first; only a block larger than the budget is solved alone. Concurrency across blocks is the
/// parallelism: this platform's threaded zheevd does not scale (n = 4200: 19.2 s on one thread,
/// 19.4 s on 32; dsyevd 3.1 / 2.4 s; dev/p67/lapack_scale.py), so a threaded solve of one block
/// after another would leave all but one core idle (tri20 exact thermal: 3256 s, probe 62624309).
// DenseBatch's packing of a matrix make() forms into the device batch: real when real_block says so.
template <class Make> auto pack_formed(const Make& make) {
    return [&make](ed::solvers::LgBlocksPacked& p) {
        const Eigen::MatrixXcd Hb = make();
        const auto n = static_cast<std::int64_t>(Hb.rows());
        if (ed::solvers::lg_detail::real_block(Hb)) {   // column-major, real arithmetic on the device
            const Eigen::MatrixXd R = Hb.real();
            p.add_real(R.data(), n);
        } else {
            p.add_complex(Hb.data(), n);
        }
    };
}

class DenseBatch {
public:
    DenseBatch(Device device, const char* verb) : device_(device), verb_(verb) {}

    /// Queue (or, on the host, solve now) the spectrum of `mv`; returns the entry index.
    std::size_t add(const ed::LinearOperator& mv) {
        using namespace ed::solvers::lg_detail;
        const auto* hk = dynamic_cast<const RepSectorMatVec*>(&mv);
        if (!hk) {
            const auto make = [&mv] { return materialize(mv); };
            return add_lazy(mv.dim(), std::nullopt, make, pack_formed(make));
        }
        // A sector operator: its CSR first -- its stored values say whether the block is real, so
        // the device sizes it exactly -- then either lane writes the matrix from it in parallel,
        // the device batch in place (no complex matrix, no n^2 scan, no copy).
        const auto csr = hk->reduced_csr();
        const bool real = csr_is_real(csr);
        const auto n = static_cast<Eigen::Index>(csr.dim);
        const auto make = [&csr, n] {
            Eigen::MatrixXcd H(n, n);
            csr_to_dense(csr, H.data());
            return H;
        };
        return add_lazy(mv.dim(), real, make, [&csr, real, n](ed::solvers::LgBlocksPacked& p) {
            if (real)
                csr_to_dense(csr, p.add_block(n, true));
            else
                csr_to_dense(csr, reinterpret_cast<Complex*>(p.add_block(n, false)));
        });
    }
    /// The same for a block given as its matrix (a spin tower's Q^dag H Q).
    std::size_t add(Eigen::MatrixXcd M) {
        const auto n = static_cast<std::uint64_t>(M.rows());
        const auto make = [&M] { return std::move(M); };
        return add_lazy(n, std::nullopt, make, pack_formed(make));
    }

    /// Solve everything queued; afterwards spectrum(id) is valid for every entry. The device batch
    /// and the host queue run at once (the device's host thread mostly waits on its streams).
    void solve() {
        if (queued_.empty() || host_.empty()) {
            solve_device();
            solve_host();
            return;
        }
        auto device = std::async(std::launch::async, [this] { solve_device(); });
        try {
            solve_host();
        } catch (...) {
            device.wait();
            throw;
        }
        device.get();
    }

    [[nodiscard]] const std::vector<double>& spectrum(std::size_t id) const { return spectra_[id]; }
    [[nodiscard]] ed::Lane lane(std::size_t id) const { return lanes_[id]; }
    [[nodiscard]] std::size_t device_blocks() const noexcept { return device_blocks_; }

private:
    // add(): the block of dimension `dim` (`real`: whether it is real, when known before it is
    // formed); make() forms its matrix for the host, pack(packed_) puts it in the device batch.
    template <class Make, class Pack>
    std::size_t add_lazy(std::uint64_t dim, std::optional<bool> real, Make&& make, Pack&& pack) {
        using namespace ed::solvers::lg_detail;
        const std::size_t id = spectra_.size();
        spectra_.emplace_back();
        ed::BlockRequest req;
        req.task = ed::Task::DenseBatch;
        req.dim = dim;
        req.verb = verb_;
        lanes_.push_back(ed::place(device_, req));
        const std::uint64_t bytes = 16 * dim * dim;   // as a complex matrix (the host's form)
        bool alone = false;                           // larger than a batch, solved by itself
        if (ed::on_device(lanes_.back())) {
            if (budget_ == 0) budget_ = batch_budget();
            // Device bytes: half for a real block; an unknown one counts as complex.
            const std::uint64_t dev_bytes = real.value_or(false) ? bytes / 2 : bytes;
            if (dev_bytes > budget_) {
                alone = fits_device_alone(dim, real.value_or(false));
                if (!alone) {   // too large for the device
                    // device='gpu' never moves a block to the host: a full spectrum has no Krylov
                    // stand-in, so the block cannot run on the device at all.
                    if (device_ == Device::Gpu)
                        throw ed::ResourceLimit(std::string(verb_ ? verb_ : "dense spectra") + ": a dense block of "
                                                + std::to_string(dim)
                                                + " states does not fit the device (matrix and cuSOLVER workspace "
                                                  "need more than half the free device memory); device='auto' "
                                                  "solves it on the host");
                    lanes_.back() = ed::Lane::HostDense;
                }
            } else if (packed_.bytes() + dev_bytes > budget_) {
                solve_device();
            }
        }
        if (!ed::on_device(lanes_.back())) {
            ed::core::guard_working_set(ed::core::footprint(ed::core::Path::DenseValues, {dim}).host, "dense spectrum");
            if (host_budget_ == 0) host_budget_ = host_budget();
            if (bytes > host_budget_) {          // too large for the pool: alone, on the threaded LAPACK
                Eigen::MatrixXcd M = make();
                spectra_.back() = dense_eigenvalues_inplace(M);
                return id;
            }
            if (host_bytes_ + bytes > host_budget_) solve_host();
            const auto t0 = Clock::now();
            host_.push_back({id, make()});
            prof_.host_form_s += seconds(t0, Clock::now());
            host_bytes_ += bytes;
            return id;
        }
        ed::core::guard_working_set(ed::core::footprint(ed::core::Path::DenseValues, {dim}).host, "dense spectrum");
        if (alone) solve_device();   // the batch so far first: this block goes up by itself
        if (packed_.data.empty()) packed_.data.reserve(budget_ / sizeof(double));   // a batch grows without copying
        const auto t0 = Clock::now();
        pack(packed_);
        prof_.form_s += seconds(t0, Clock::now());
        const double nb = static_cast<double>(packed_.block_dim.back());
        prof_.dev_n3 += nb * nb * nb;
        queued_.push_back(id);
        if (alone) solve_device();
        return id;
    }

    // A block larger than a batch is solved by itself on the device when its matrix and cuSOLVER's
    // workspace for it fit in half the free device memory (`real`: a real block; one whose
    // realness is not known before it is formed counts as complex); else on the host under
    // 'auto', and add_lazy raises ResourceLimit under 'gpu'. Not checked under ED_MEM_GUARD_OFF.
    static bool fits_device_alone(std::uint64_t dim, bool real) {
#ifdef WITH_CUDA
        if (ed::core::mem_guard_off()) return true;
        const auto free = ed::core::available_device_bytes(/*fresh=*/true);
        if (!free) return false;
        try {
            const std::uint64_t need =
                (real ? 8 : 16) * dim * dim
                + ed::solvers::lg_block_workspace_bytes_gpu(static_cast<std::int64_t>(dim), real);
            return need <= *free / 2;
        } catch (const std::exception& e) {
            ED_LOG(Warn,
                   "dense spectra: cuSOLVER's workspace query for a block of %llu states failed (%s); "
                   "the block counts as too large for the device",
                   static_cast<unsigned long long>(dim), e.what());
            return false;
        }
#else
        (void)dim;
        (void)real;
        return false;
#endif
    }

#ifdef WITH_CUDA
    // Blocks [lo, hi) of the packed batch solved on the device, halving the batch on a failure;
    // a single block that still fails raises (device='gpu' does not fall back to the host).
    std::vector<double> solve_device_split(std::size_t lo, std::size_t hi) {
        ed::solvers::LgBlocksPacked part;
        for (std::size_t q = lo; q < hi; ++q) {
            const double* d = packed_.data.data() + packed_.offset[q];
            if (packed_.real[q])
                part.add_real(d, packed_.block_dim[q]);
            else
                part.add_complex(reinterpret_cast<const std::complex<double>*>(d), packed_.block_dim[q]);
        }
        try {
            return ed::solvers::lg_blocks_batched_eigenvalues_gpu(part);
        } catch (const std::exception& e) {
            if (hi - lo == 1)
                throw ed::ResourceLimit(std::string(verb_ ? verb_ : "dense spectra")
                                        + ": the device solve of a dense "
                                          "block of "
                                        + std::to_string(packed_.block_dim[lo]) + " states failed (" + e.what()
                                        + "); device='auto' solves it on the host");
            ED_LOG(Info, "dense spectra: a device batch of %zu blocks failed (%s); retrying it in halves", hi - lo,
                   e.what());
            const std::size_t mid = lo + (hi - lo) / 2;
            std::vector<double> ev = solve_device_split(lo, mid);
            const std::vector<double> rest = solve_device_split(mid, hi);
            ev.insert(ev.end(), rest.begin(), rest.end());
            return ev;
        }
    }
#endif

    // The device batch. Under device='auto' a failed device solve falls back to the host; under
    // 'gpu' it is retried in halves on the device (solve_device_split), and raises if a block alone fails.
    void solve_device() {
        if (queued_.empty()) return;
        const Timed timed(prof_.dev_s);
        prof_.dev_blocks += queued_.size();
        std::vector<double> ev;
        bool on_device = false;
#ifdef WITH_CUDA
        try {
            ev = ed::solvers::lg_blocks_batched_eigenvalues_gpu(packed_);
            on_device = true;
        } catch (const std::exception& e) {
            if (device_ == Device::Gpu) {
                ED_LOG(Info, "dense spectra: the batched device solve of %zu blocks failed (%s); retrying in halves",
                       queued_.size(), e.what());
                const std::size_t mid = queued_.size() / 2;
                ev = queued_.size() > 1 ? solve_device_split(0, mid) : std::vector<double>{};
                const std::vector<double> rest = solve_device_split(queued_.size() > 1 ? mid : 0, queued_.size());
                ev.insert(ev.end(), rest.begin(), rest.end());
                on_device = true;
            } else {
                ED_LOG(Warn,
                       "dense spectra: the batched device solve of %zu blocks failed (%s); solving them on the host",
                       queued_.size(), e.what());
            }
        }
#endif
        std::size_t off = 0;
        for (std::size_t q = 0; q < queued_.size(); ++q) {
            const std::size_t nb = static_cast<std::size_t>(packed_.block_dim[q]);
            if (on_device) {
                spectra_[queued_[q]].assign(ev.begin() + static_cast<long>(off),
                                            ev.begin() + static_cast<long>(off + nb));
                off += nb;
            } else {
                const auto n = static_cast<Eigen::Index>(nb);
                const double* d = packed_.data.data() + packed_.offset[q];
                Eigen::MatrixXcd Hb =
                    packed_.real[q]
                        ? Eigen::MatrixXcd(Eigen::Map<const Eigen::MatrixXd>(d, n, n).cast<std::complex<double>>())
                        : Eigen::MatrixXcd(
                            Eigen::Map<const Eigen::MatrixXcd>(reinterpret_cast<const std::complex<double>*>(d), n, n));
                spectra_[queued_[q]] = ed::solvers::lg_detail::dense_eigenvalues_inplace(Hb);
                lanes_[queued_[q]] = ed::Lane::HostDense;
            }
        }
        if (on_device) device_blocks_ += queued_.size();
        queued_.clear();
        packed_ = {};
        budget_ = 0;   // measured afresh for the next batch
    }

    // The queued host blocks, concurrently: one serial LAPACK solve per thread, largest first
    // (LPT); a lone block keeps the threaded solve.
    void solve_host() {
        if (host_.empty()) return;
        const Timed timed(prof_.host_s);
        prof_.host_blocks += host_.size();
#ifdef _OPENMP
        const int team = omp_get_max_threads();
#else
        const int team = 1;
#endif
        if (host_.size() < 2 || team < 2) {
            for (auto& [id, M] : host_) spectra_[id] = ed::solvers::lg_detail::dense_eigenvalues_inplace(M);
        } else {
            std::vector<std::size_t> order(host_.size());
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::stable_sort(order.begin(), order.end(), [this](std::size_t a, std::size_t b) {
                return host_[a].second.rows() > host_[b].second.rows();
            });
            const ed::parallel::ThreadBudgetScope blas_serial(team, 1);
            std::exception_ptr err;
#pragma omp parallel for schedule(dynamic, 1)
            for (long long q = 0; q < static_cast<long long>(order.size()); ++q) {
                auto& [id, M] = host_[order[static_cast<std::size_t>(q)]];
                try {
                    spectra_[id] = ed::solvers::lg_detail::dense_eigenvalues_inplace(M);
                } catch (...) {
#pragma omp critical(qed_dense_batch_error)
                    if (!err) err = std::current_exception();
                }
                M.resize(0, 0);
            }
            if (err) {
                host_.clear();
                host_bytes_ = 0;
                std::rethrow_exception(err);
            }
        }
        host_.clear();
        host_bytes_ = 0;
        host_budget_ = 0;   // measured afresh for the next queue
    }

    // The host queue holds at most 16 GiB of matrices, or a quarter of the RAM the job may still
    // allocate when that is smaller (not checked under ED_MEM_GUARD_OFF): a team's worth of large
    // blocks at once (32 blocks of 4600 states are 11 GB).
    static std::uint64_t host_budget() {
        std::uint64_t b = std::uint64_t{16} << 30;
        if (ed::core::mem_guard_off()) return b;
        if (const std::uint64_t ram = ed::core::available_ram_bytes()) b = std::min<std::uint64_t>(b, ram / 4);
        return std::max<std::uint64_t>(b, 1);
    }

    // 2 GiB of matrices: a cluster's blocks share one batch, so its largest block runs beside the
    // others on the stream pool instead of alone after them (NLCE 16-site clusters, P7.4); less
    // when a quarter of the free device memory or of the job's RAM is smaller (those two are not
    // checked under ED_MEM_GUARD_OFF, or where they cannot be measured).
    // ED_GPU_DENSE_BATCH_GIB replaces the 2 GiB.
    static std::uint64_t batch_budget() {
        const double gib = ed::env::real("ED_GPU_DENSE_BATCH_GIB", -1.0);   // -1: not set
        std::uint64_t b = gib >= 0.0 ? static_cast<std::uint64_t>(gib * 1073741824.0) : std::uint64_t{2} << 30;
        if (ed::core::mem_guard_off()) return b;
        if (const auto dev = ed::core::available_device_bytes(/*fresh=*/true)) b = std::min<std::uint64_t>(b, *dev / 4);
        if (const std::uint64_t ram = ed::core::available_ram_bytes()) b = std::min<std::uint64_t>(b, ram / 4);
        return std::max<std::uint64_t>(b, 1);
    }

    Device device_;
    const char* verb_;
    std::vector<ed::Lane> lanes_;
    ed::solvers::LgBlocksPacked packed_;
    std::vector<std::size_t> queued_;
    std::vector<std::vector<double>> spectra_;
    std::size_t device_blocks_ = 0;
    std::uint64_t budget_ = 0;   // bytes of the current batch's matrices at most
    std::vector<std::pair<std::size_t, Eigen::MatrixXcd>> host_;   // (entry, matrix) for solve_host
    std::uint64_t host_bytes_ = 0;
    std::uint64_t host_budget_ = 0;

    // Phase seconds under ED_SYM_PROFILE, one line when the batch dies: forming and packing the
    // device blocks, the device solves (wall, host fallback included), forming and solving the host
    // blocks. The two solve times may overlap (solve() runs the last two queues at once).
    using Clock = std::chrono::steady_clock;
    static double seconds(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    }
    struct Timed {   // adds the scope's seconds to `into`
        double& into;
        Clock::time_point t = Clock::now();
        explicit Timed(double& s) : into(s) {}
        ~Timed() { into += seconds(t, Clock::now()); }
    };
    struct Profile {
        double form_s = 0, dev_s = 0, host_form_s = 0, host_s = 0, dev_n3 = 0;
        std::size_t dev_blocks = 0, host_blocks = 0;
    } prof_;

public:
    ~DenseBatch() {
        if (prof_.dev_blocks + prof_.host_blocks == 0 || !ed::env::flag("ED_SYM_PROFILE", false)) return;
        ED_LOG(Info,
               "[dense] %s: device %zu blocks (sum n^3 %.3g): form %.3f s, solve %.3f s | host %zu "
               "blocks: form %.3f s, solve %.3f s",
               verb_, prof_.dev_blocks, prof_.dev_n3, prof_.form_s, prof_.dev_s, prof_.host_blocks, prof_.host_form_s,
               prof_.host_s);
    }
};

/// The S^2 operator a total-spin restriction needs (null without one).
inline std::shared_ptr<::Operator> s2_carrier_for(const Spec& s, int n_sites) {
    return s.two_S >= 0 ? ed::ops::make_S2_carrier(static_cast<std::uint64_t>(n_sites)) : nullptr;
}

// Every element of the group the permutations generate.
inline std::vector<Perm> close_group(const std::vector<Perm>& gens, int n) {
    Perm id(static_cast<std::size_t>(n));
    std::iota(id.begin(), id.end(), 0);
    std::set<Perm> seen{id};
    std::vector<Perm> out{id};
    for (std::size_t head = 0; head < out.size(); ++head)
        for (const Perm& g : gens) {
            Perm y(id.size());
            for (std::size_t i = 0; i < y.size(); ++i) y[i] = g[static_cast<std::size_t>(out[head][i])];
            if (seen.insert(y).second) out.push_back(std::move(y));
            if (out.size() > 1'000'000)
                throw ed::ResourceLimit(
                    "symmetry group too large to average an operator over (more than 10^6 elements)");
        }
    return out;
}

using Keep = ed::ops::SzKeep;

/// O averaged over the symmetry group of a Spec (and the spin flip), without the terms whose
/// S^z change `keep` excludes, mapped by the antiunitary `image` (K or Theta: A O A^-1, for a
/// partner A|v>, <A v|O|A v> = conj <v|A^-1 O A|v>); built once per
/// (operator, flip, keep, image), as canonical terms (average) or as the row program a sector
/// matvec walks (program). An operator averaged over the symmetries a block uses is block
/// diagonal and has the same trace against any function of H over an ensemble those
/// symmetries preserve.
class Averager {
public:
    // Where a level stands for a whole multiplet (`multiplets`: total spin with an SU(2)-symmetric
    // H) an O that is not SU(2) invariant enters through its SU(2)-scalar part, whose expectation
    // is the multiplet average (and whose thermal trace with that H is O's). In a uniform field
    // every member is a level of its own and O enters as it is.
    Averager(const Spec& s, int n_sites, bool multiplets) : su2_(multiplets) {
        std::vector<Perm> gens = abelian_or_identity(s, n_sites);
        gens.insert(gens.end(), s.residues.begin(), s.residues.end());
        G_ = close_group(gens, n_sites);
    }
    const ed::ops::MaskedOperator& average(const ::Operator& O, bool flip, Keep keep,
                                           Antiunitary image = Antiunitary::None) {
        auto& slot = averages_[{&O, flip, static_cast<int>(keep), static_cast<int>(image)}];
        if (!slot) {
            const ed::ops::MaskedOperator src = su2_ ? ed::ops::su2_scalar_part(O.canonical()) : O.canonical();
            ed::ops::MaskedOperator a = ed::ops::group_average(ed::ops::keep_sz_changes(src, keep), G_, flip);
            using Map = ed::ops::MaskedOperator::Map;
            if (image != Antiunitary::None) a = a.image(image == Antiunitary::Theta ? Map::Theta : Map::K);
            slot = std::make_shared<const ed::ops::MaskedOperator>(std::move(a));
        }
        return *slot;
    }
    std::shared_ptr<const ed::ops::MaskedProgram> program(const ::Operator& O, bool flip, Keep keep,
                                                          Antiunitary image = Antiunitary::None) {
        auto& slot = programs_[{&O, flip, static_cast<int>(keep), static_cast<int>(image)}];
        if (!slot)
            slot = std::make_shared<const ed::ops::MaskedProgram>(
                ed::ops::compile_operator(average(O, flip, keep, image).dagger()));
        return slot;
    }

private:
    using Key = std::tuple<const ::Operator*, bool, int, int>;
    bool su2_ = false;
    std::vector<Perm> G_;
    std::map<Key, std::shared_ptr<const ed::ops::MaskedOperator>> averages_;
    std::map<Key, std::shared_ptr<const ed::ops::MaskedProgram>> programs_;
};

/// A block-diagonal (averaged) operator, given by its row program, restricted to block `bi` of
/// star `sb`, in the basis the block's H acts on; with `device` it may bind to a CUDA backend. Its
/// reduced CSR takes from the block's `budget` after H's.
inline std::shared_ptr<const ed::LinearOperator>
block_observable(const std::shared_ptr<const ed::ops::MaskedProgram>& A, const ed::solvers::lg_detail::StarBuild& sb,
                 const std::shared_ptr<ed::solvers::BlockData>& bi, bool device,
                 const std::shared_ptr<ed::planner::CsrBudget>& budget = nullptr) {
    using namespace ed::solvers::lg_detail;
    auto rep = std::make_shared<RepSectorMatVec>(A, bi->gop ? bi->gsec : sb.hk->rep_data_ptr());
    rep->set_csr_budget(budget);
    if (device) rep->enable_device(true);
    return rep;
}

/// The co-group character table of block `irrep` of a star, as (elements, characters): a group
/// sector's row; for the plain block of a trivial co-group the trivial irrep (the identity, character
/// 1); none (false) for an irrep index the star does not have.
inline bool irrep_table(const ed::solvers::LittleGroupStarInfo& info, int irrep, const std::vector<int>*& elems,
                        const std::vector<Complex>*& chars) {
    using namespace ed::solvers::lg_detail;
    if (irrep >= 0) {
        if (static_cast<std::size_t>(irrep) >= info.little_characters.size()) return false;
        elems = &info.little_elems;
        chars = &info.little_characters[static_cast<std::size_t>(irrep)];
        return true;
    }
    elems = &trivial_elems();
    chars = &trivial_chars();
    return true;
}

/// chi_sigma(residue i) of block `irrep` of a star (-1: the identity); nullopt when
/// i is not in its group or the block has no character table.
inline std::optional<Complex> irrep_char(const ed::solvers::LittleGroupStarInfo& info, int irrep, int i) {
    const std::vector<int>* elems = nullptr;
    const std::vector<Complex>* chars = nullptr;
    if (!irrep_table(info, irrep, elems, chars)) return std::nullopt;
    return ed::solvers::lg_detail::co_group_char(*elems, *chars, i);
}

/// The physical labels (momentum, co-group irrep characters) of a level of star `sb`: every
/// co-group element.
inline void label(Level& L, const ed::solvers::lg_detail::StarBuild& sb) {
    const auto& info = sb.info;
    L.momentum = info.momentum;
    L.irrep_characters.clear();
    const std::vector<int>* elems = nullptr;
    const std::vector<Complex>* chars = nullptr;
    if (!irrep_table(info, L.tag.irrep, elems, chars)) return;
    for (std::size_t e = 0; e < elems->size(); ++e) L.irrep_characters.emplace_back((*elems)[e], (*chars)[e]);
}

inline bool has_selection(const Spec& s) {
    return !s.only_k0.empty() || !s.only_irrep.empty() || !s.only_momentum.empty() || !s.only_irrep_chars.empty();
}

/// A verb whose selection (momentum, star, irrep, irrep character) matched no block raises
/// EmptySelection instead of answering for an empty space.
inline void require_some_block(const Spec& s, std::size_t n_blocks, const char* verb) {
    if (has_selection(s) && n_blocks == 0)
        throw ed::EmptySelection(std::string(verb)
                                 + ": the selection matches no block: no star of the "
                                   "requested Sz sectors has that momentum, star index or little-group irrep");
}

/// The same after the spin-tower filter: under total_spin a selected block can hold no state of
/// the requested spin (every state of it belongs to a higher multiplet).
inline void require_some_level(const Spec& s, bool none, const char* verb) {
    if (has_selection(s) && none && s.two_S >= 0)
        throw ed::EmptySelection(std::string(verb) + ": the selected blocks hold no state of total spin S = "
                                 + (s.two_S % 2 ? std::to_string(s.two_S) + "/2" : std::to_string(s.two_S / 2)));
}

/// device='gpu' needs a usable device before anything runs.
inline void require_device(Device d, const char* verb) {
    if (d == Device::Gpu && !ed::have_cuda())
        throw ed::DeviceUnavailable(std::string(verb) + ": device='gpu', but no usable CUDA device is visible");
}

/// "the block of star K, irrep I, n_up N (dim D)": how a device refusal names a block.
inline std::string block_name(const ed::solvers::LittleGroupBlockTag& tag) {
    return "the block of star " + std::to_string(tag.k0) + ", irrep " + std::to_string(tag.irrep) + ", n_up "
           + std::to_string(tag.n_up) + " (dim " + std::to_string(tag.dim) + ")";
}

/// Why a block has no device kernel, for place()'s refusal.
inline const char* no_kernel_reason(int irrep_dim) {
    if (irrep_dim > 1)
        return "is a sector of an irrep of dimension > 1, whose device kernel -- its reduced CSR, built on the host "
               "and "
               "uploaded -- does not fit the block's CSR budget or the device CSR budget";
    return "has no device kernel";
}

/// A thermal average under a spec that restricts the sectors (a total spin, one Sz sector or
/// parity, a momentum or irrep selection) describes that restricted ensemble, not the canonical
/// one: a ("restricted_ensemble", ...) diagnostic says which. Nothing when nothing is restricted.
inline void note_restricted_ensemble(const Spec& s, Diagnostics& out, const char* verb) {
    std::vector<std::string> parts;
    if (s.two_S >= 0)
        parts.push_back("total spin S = "
                        + (s.two_S % 2 ? std::to_string(s.two_S) + "/2" : std::to_string(s.two_S / 2)));
    if (s.n_up >= 0) parts.push_back("one Sz sector");
    if (s.sz_parity >= 0) parts.push_back("one Sz parity");
    if (!s.only_k0.empty() || !s.only_momentum.empty()) parts.push_back("the selected momenta");
    if (!s.only_irrep.empty() || !s.only_irrep_chars.empty()) parts.push_back("the selected irreps");
    if (parts.empty()) return;
    std::string what = parts.front();
    for (std::size_t i = 1; i < parts.size(); ++i) what += ", " + parts[i];
    out.emplace_back("restricted_ensemble", std::string(verb) + ": the averages run over " + what
                                                + " only: a restricted ensemble, not the canonical one");
}

/// fn(cx, tr_on, star) for every star of one subspace, one star resident at a time. Returns the
/// number of blocks handed to fn, after the selection.
template <class Fn>
std::size_t walk(const ::Operator& H, int n_sites, const Spec& s, const ed::solvers::LittleGroupOptions& opt, Fn&& fn) {
    using namespace ed::solvers::lg_detail;
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(H, abelian_or_identity(s, n_sites), s.residues, n_sites, opt, cx, tr_on);
    // A co-group character names sigma alone; time reversal would fold sigma* into its block. Theta
    // also changes Sz: under any selection the selected sectors are not its image of themselves.
    if (!s.only_irrep_chars.empty() || (cx.tr == ed::solvers::Antiunitary::Theta && has_selection(s))) {
        tr_on = false;
        cx.tr = ed::solvers::Antiunitary::None;
    }
    // The stars to walk: engine_options() copies the Spec's k0 selection; eigs narrows it to one
    // survivor's star without changing what the caller selected.
    const std::set<int> only(opt.only_k0.begin(), opt.only_k0.end());
    auto momentum_of = [&](int k_ext) -> const std::vector<Complex>& {
        return cx.giA.irreps[static_cast<std::size_t>(k_ext % cx.n_irr_raw)].character;
    };
    // A star that time reversal closes -- k with -k where no residue relates them -- counts the
    // conjugate members in its multiplicity, so its blocks are folded like a sigma <-> sigma* pair:
    // multiplet adds the conjugate, expect and the thermal observables average <O> with it.
    std::map<int, std::size_t> residue_star;   // momentum -> the size of its star under the residues
    if (tr_on)
        for (const auto& [root, mem] : star_partition(cx, false))
            for (int k : mem) residue_star[k] = mem.size();
    std::size_t n_blocks = 0;
    for (const auto& [k0, members] : star_partition(cx, tr_on)) {
        if (!only.empty() && only.count(k0) == 0) continue;
        const bool hit = std::any_of(members.begin(), members.end(), [&](int m) {
            const auto& chi = momentum_of(m);
            return meets(s.only_momentum, [&](int i) -> std::optional<Complex> {
                if (i >= 0 && static_cast<std::size_t>(i) < chi.size()) return chi[static_cast<std::size_t>(i)];
                return std::nullopt;
            });
        });
        if (!hit) continue;
        ed::core::poll_interrupt();
        const auto t_build = std::chrono::steady_clock::now();
        StarBuild sb = build_star_blocks(H, cx, tr_on, k0, members, opt);
        if (tr_on && members.size() > residue_star.at(k0))
            for (auto& bi : sb.blocks) bi->tag.tr_folded = true;
        sb.t_build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count();
        sb.info.momentum = momentum_of(k0);
        // build_star_blocks solved only the wanted irreps already; the filter states the contract.
        if (!s.only_irrep_chars.empty())
            sb.blocks.erase(std::remove_if(sb.blocks.begin(), sb.blocks.end(),
                                           [&](const auto& bi) {
                                               return !meets(s.only_irrep_chars, [&](int i) {
                                                   return irrep_char(sb.info, bi->tag.irrep, i);
                                               });
                                           }),
                            sb.blocks.end());
        n_blocks += sb.blocks.size();
        fn(cx, tr_on, sb);
    }
    return n_blocks;
}

// ---- A level's multiplet in momentum sectors (members.cpp) --------------------------

/// The momentum sectors of one subspace under the abelian group A, each built on its first request
/// from the subspace's one orbit table.
class MomentumSectors {
public:
    MomentumSectors(const ::Operator& H, const std::vector<Perm>& A, int n_sites, const Subspace& sub);
    /// The sector of momentum chi (characters over A), or null when no momentum has them.
    [[nodiscard]] std::shared_ptr<const ed::symmetry::RepSectorData> of(const std::vector<Complex>& chi);

private:
    ed::solvers::lg_detail::EngineContext cx_;
    std::map<int, std::shared_ptr<const ed::symmetry::RepSectorData>> built_;
};

/// One member of a level's multiplet: a unit vector of a momentum sector of A, in its subspace.
struct Member {
    Subspace sub;
    BlockVector v;
};

/// The momentum sectors a call's members land in, per subspace (n_up, sz_parity).
struct MemberSectors {
    const ::Operator& H;
    std::vector<Perm> A;
    int n_sites = 0;
    std::map<std::pair<int, int>, std::unique_ptr<MomentumSectors>> by_sub;
};

/// The degenerate multiplet of level L (of a walk under Spec s) from its vector v, as `count`
/// orthonormal unit vectors of momentum sectors of A: the closure of v under the residues, L's
/// antiunitary fold and its flip / Theta mirror (multiplet()'s operations). Each member is gathered
/// at its sector's representatives from v's amplitudes, so no vector of the whole Sz sector is
/// formed. Throws std::logic_error when the closure does not hold `count` states.
[[nodiscard]] std::vector<Member> members_of(const Level& L, const BlockVector& v, std::uint64_t count, const Spec& s,
                                             MemberSectors& ms);

}  // namespace ed::sectors::detail
