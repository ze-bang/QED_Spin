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

inline ed::solvers::LittleGroupOptions
engine_options(const Spec& s, const Subspace& sub) {
    ed::solvers::LittleGroupOptions o;
    o.n_up          = sub.n_up;
    o.sz_parity     = sub.sz_parity;
    // subspaces() already enforced 'require' against H. Inside a subspace the engine engages
    // the flip where the subspace is its own image (n_up = N/2, a parity half with N even, the
    // full space) and not elsewhere: an Sz = S tower, an explicit n_up != N/2 or a mirror pair
    // (which gets the symmetry through the fold) -- 'auto' there, never a refusal.
    o.spin_flip     = s.spin_flip == 1 ? -1 : s.spin_flip;
    o.time_reversal = s.time_reversal;
    o.only_k0       = s.only_k0;
    o.only_irrep    = s.only_irrep;
    o.only_irrep_chars = s.only_irrep_chars;
    return o;
}

/// The operator one block is solved with. With a total-spin restriction it is the block's
/// H wrapped in the Lowdin projector onto the spin-S tower (S^2 built on the same basis):
/// H on the tower and `ghost` on the rest, which callers drop. `multiplicity` includes the
/// 2S + 1 members of each multiplet. A null `op` means the block holds no
/// state of the requested spin.
struct BlockOp {
    std::shared_ptr<const ed::LinearOperator> op;
    double        ghost        = std::numeric_limits<double>::infinity();
    std::uint64_t multiplicity = 1;
    std::shared_ptr<const ed::symmetry::LowdinS2Projector> projector;   ///< onto the tower (SU(2) only)
    [[nodiscard]] bool is_ghost(double e) const {
        return std::isfinite(ghost) && e > ghost - 1e-6 * std::max(1.0, std::abs(ghost));
    }
};

/// The block budget of a block's reduced CSRs (csr_policy.h): block_csr_budget_bytes less the
/// `working_set` its solver will hold, sized now -- before the solver allocates anything.
inline std::shared_ptr<ed::planner::CsrBudget> block_budget(std::uint64_t working_set) {
    return std::make_shared<ed::planner::CsrBudget>(ed::planner::block_csr_budget_bytes(working_set));
}

inline BlockOp block_operator(const Spec& s, int n_sites, const Subspace& sub,
                              const ed::solvers::lg_detail::StarBuild& sb,
                              const std::shared_ptr<ed::solvers::BlockData>& bi,
                              const std::shared_ptr<::Operator>& s2_carrier,
                              Device device = Device::Cpu,
                              const std::shared_ptr<ed::planner::CsrBudget>& budget = nullptr) {
    using namespace ed::solvers::lg_detail;
    BlockOp b;
    b.op = std::shared_ptr<const ed::LinearOperator>(bi, &block_mv(*bi));
    b.multiplicity = bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
    // H's reduced CSR first, then S^2's, from the block's budget (a W block's H is the star's
    // k-sector operator).
    if (budget) {
        if (bi->gop) bi->gop->set_csr_budget(budget);
        else if (sb.hk) sb.hk->set_csr_budget(budget);
    }
    // Group and momentum sectors have a device kernel; the isotypic sandwich does not.
    RepSectorMatVec* rep = bi->gop ? bi->gop.get() : (bi->W ? nullptr : sb.hk.get());
    const bool dev = rep && device != Device::Cpu;
    if (s.two_S < 0) {
        if (dev) rep->enable_device(true);   // its device kernel: op->has_device_kernel()
        return b;
    }
    std::shared_ptr<const ed::LinearOperator> s2;
    std::shared_ptr<RepSectorMatVec> s2rep;
    if (bi->gop) {
        s2 = s2rep = std::make_shared<RepSectorMatVec>(*s2_carrier, bi->gsec);
        s2rep->set_csr_budget(budget);
    } else {
        auto s2k = std::make_shared<RepSectorMatVec>(*s2_carrier, sb.hk->rep_data_ptr());
        s2k->set_csr_budget(budget);
        if (bi->W) s2 = std::make_shared<ProjectedBlockOp>(s2k, bi->W);
        else       s2 = s2rep = s2k;
    }
    const auto towers = ed::symmetry::allowed_two_S_in_block(n_sites, sub.n_up, sub.sz_parity,
                                                             bi->tag.flip_parity);
    if (std::find(towers.begin(), towers.end(), s.two_S) == towers.end()) {
        b.op.reset();                   // this flip-parity block holds no spin-S state
        return b;
    }
    auto proj = std::make_shared<ed::symmetry::LowdinS2Projector>(s2, s.two_S, towers);
    if (dev && s2rep) {                 // H and S^2 on the device: the projected apply runs there
        rep->enable_device(true);
        s2rep->enable_device(true);
    }
    auto wrapped = std::make_shared<ed::symmetry::CasimirProjectedOperator>(b.op, proj, 1);
    b.ghost = wrapped->ghost_shift();
    b.projector = proj;
    b.op = wrapped;
    b.multiplicity *= static_cast<std::uint64_t>(sub.members);
    return b;
}

/// The antiunitary pairing of a block's levels: its star fold's map (`star_tr`, the walk context's
/// tr: K or Theta), or Theta when the subspace's mirror is the time-reversal image.
inline Antiunitary fold_of(Antiunitary star_tr, const Subspace& sub, const ed::solvers::LittleGroupBlockTag& tag) {
    if (tag.tr_folded) return star_tr;
    return sub.mirror == 2 && sub.theta ? Antiunitary::Theta : Antiunitary::None;
}

/// Records in a result which antiunitary map folded anything: the stars' (cx.tr), or Theta for a
/// subspace mirrored by it.
inline void note_time_reversal(Antiunitary& seen, const ed::solvers::lg_detail::EngineContext& cx,
                               const Subspace& sub) {
    if (cx.tr != Antiunitary::None) seen = cx.tr;
    else if (sub.mirror == 2 && sub.theta) seen = Antiunitary::Theta;
}

/// Whether a level's mirror is its Theta image (else the spin flip's, when it has one).
inline bool theta_mirror(const Level& L) {
    return L.mirror == 2 && L.fold == Antiunitary::Theta && !L.tag.tr_folded;
}

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

/// Dense spectra of many blocks: on the host one block at a time, or -- on a device lane --
/// materialised as the walk visits them and solved in batched cuSOLVER calls (the walk streams
/// stars, so the matrices are the only thing that outlives a star). place(Task::DenseBatch)
/// chooses each entry's lane. A batch is packed on the host and uploaded at once, so it is solved
/// before it outgrows a quarter of the free device memory or of the RAM the job may still
/// allocate (cuSOLVER's workspace and the eigenvalues come on top), and never holds more than
/// 256 MiB of matrices; a block larger than that is solved on the host, and so is a batch whose
/// device solve fails.
class DenseBatch {
public:
    DenseBatch(Device device, const char* verb) : device_(device), verb_(verb) {}

    /// Queue (or, on the host, solve now) the spectrum of `mv`; returns the entry index.
    std::size_t add(const ed::LinearOperator& mv) {
        using namespace ed::solvers::lg_detail;
        const std::size_t id = spectra_.size();
        spectra_.emplace_back();
        ed::BlockRequest req;
        req.task = ed::Task::DenseBatch;
        req.dim  = mv.dim();
        req.verb = verb_;
        lanes_.push_back(ed::place(device_, req));
        const std::uint64_t bytes = 16 * mv.dim() * mv.dim();
        if (ed::on_device(lanes_.back())) {
            if (budget_ == 0) budget_ = batch_budget();
            if (bytes > budget_) lanes_.back() = ed::Lane::HostDense;   // too large for any batch
            else if (16 * packed_.data.size() + bytes > budget_) solve();
        }
        if (!ed::on_device(lanes_.back())) {
            ed::core::guard_working_set(ed::core::footprint(ed::core::Path::DenseValues, {mv.dim()}).host,
                                        "dense spectrum");
            spectra_.back() = solve_block_full(mv);
            return id;
        }
        const Eigen::MatrixXcd Hb = materialize(mv);
        const std::size_t nb = static_cast<std::size_t>(Hb.rows());
        packed_.offset.push_back(packed_.data.size());
        packed_.block_dim.push_back(ed::core::checked_narrow<int>(nb, "dense batch block"));
        packed_.block_irrep_dim.push_back(1);
        packed_.data.insert(packed_.data.end(), Hb.data(), Hb.data() + nb * nb);   // column-major
        queued_.push_back(id);
        return id;
    }

    /// Solve everything queued; afterwards spectrum(id) is valid for every entry.
    void solve() {
        if (queued_.empty()) return;
        std::vector<double> ev;
        bool on_device = false;
#ifdef WITH_CUDA
        try {
            ev = ed::solvers::lg_blocks_batched_eigenvalues_gpu(packed_);
            on_device = true;
        } catch (const std::exception& e) {
            ED_LOG(Warn, "dense spectra: the batched device solve of %zu blocks failed (%s); solving them on the host",
                   queued_.size(), e.what());
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
                Eigen::MatrixXcd Hb = Eigen::Map<const Eigen::MatrixXcd>(
                    packed_.data.data() + packed_.offset[q], static_cast<Eigen::Index>(nb),
                    static_cast<Eigen::Index>(nb));
                spectra_[queued_[q]] = ed::solvers::lg_detail::dense_eigenvalues_inplace(Hb);
                lanes_[queued_[q]] = ed::Lane::HostDense;
            }
        }
        if (on_device) device_blocks_ += queued_.size();
        queued_.clear();
        packed_ = {};
        budget_ = 0;   // measured afresh for the next batch
    }

    [[nodiscard]] const std::vector<double>& spectrum(std::size_t id) const { return spectra_[id]; }
    [[nodiscard]] ed::Lane lane(std::size_t id) const { return lanes_[id]; }
    [[nodiscard]] std::size_t device_blocks() const noexcept { return device_blocks_; }

private:
    // 256 MiB of matrices already amortise the launch (many small blocks, or a few large ones);
    // less when a quarter of the free device memory or of the job's RAM is smaller (those two
    // are not checked under ED_MEM_GUARD_OFF, or where they cannot be measured).
    static std::uint64_t batch_budget() {
        std::uint64_t b = std::uint64_t{256} << 20;
        if (ed::core::mem_guard_off()) return b;
        if (const auto dev = ed::core::available_device_bytes(/*fresh=*/true)) b = std::min<std::uint64_t>(b, *dev / 4);
        if (const std::uint64_t ram = ed::core::available_ram_bytes()) b = std::min<std::uint64_t>(b, ram / 4);
        return std::max<std::uint64_t>(b, 1);
    }

    Device device_;
    const char* verb_;
    std::vector<ed::Lane>             lanes_;
    ed::solvers::LgBlocksPacked       packed_;
    std::vector<std::size_t>          queued_;
    std::vector<std::vector<double>>  spectra_;
    std::size_t                       device_blocks_ = 0;
    std::uint64_t                     budget_ = 0;   // bytes of the current batch's matrices at most
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
            if (out.size() > 1'000'000) throw std::runtime_error("symmetry group too large to average an operator over");
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
    if (bi->W) return std::make_shared<ProjectedBlockOp>(rep, bi->W);
    if (device) rep->enable_device(true);
    return rep;
}

/// The co-group character table of block `irrep` of a star, as (elements, characters): a projected
/// block's row; for a plain block of a trivial co-group the trivial irrep (the identity, character 1);
/// for a declined non-trivial co-group, whose plain block mixes irreps, none (false).
inline bool irrep_table(const ed::solvers::LittleGroupStarInfo& info, int irrep,
                        const std::vector<int>*& elems, const std::vector<Complex>*& chars) {
    using namespace ed::solvers::lg_detail;
    if (irrep >= 0) {
        if (static_cast<std::size_t>(irrep) >= info.little_characters.size()) return false;
        elems = &info.little_elems;
        chars = &info.little_characters[static_cast<std::size_t>(irrep)];
        return true;
    }
    if (!info.declined.empty()) return false;
    elems = &trivial_elems();
    chars = &trivial_chars();
    return true;
}

/// chi_sigma(residue i) of block `irrep` of a star (-1: the identity), aliases included; nullopt when
/// i is not in its group or the block has no character table.
inline std::optional<Complex> irrep_char(const ed::solvers::LittleGroupStarInfo& info, int irrep, int i) {
    const std::vector<int>* elems = nullptr;
    const std::vector<Complex>* chars = nullptr;
    if (!irrep_table(info, irrep, elems, chars)) return std::nullopt;
    return ed::solvers::lg_detail::co_group_char(*elems, *chars, info.little_aliases, i);
}

/// The physical labels (momentum, co-group irrep characters) of a level of star `sb`: every listed
/// co-group element, then the aliased residues.
inline void label(Level& L, const ed::solvers::lg_detail::StarBuild& sb) {
    const auto& info = sb.info;
    L.momentum = info.momentum;
    L.irrep_characters.clear();
    const std::vector<int>* elems = nullptr;
    const std::vector<Complex>* chars = nullptr;
    if (!irrep_table(info, L.tag.irrep, elems, chars)) return;
    for (std::size_t e = 0; e < elems->size(); ++e) L.irrep_characters.emplace_back((*elems)[e], (*chars)[e]);
    for (const auto& [r, e, c] : info.little_aliases)
        L.irrep_characters.emplace_back(r, c * (*chars)[static_cast<std::size_t>(e)]);
}

inline bool has_selection(const Spec& s) {
    return !s.only_k0.empty() || !s.only_irrep.empty() || !s.only_momentum.empty() || !s.only_irrep_chars.empty();
}

/// A verb whose selection (momentum, star, irrep, irrep character) matched no block raises
/// EmptySelection instead of answering for an empty space.
inline void require_some_block(const Spec& s, std::size_t n_blocks, const char* verb) {
    if (has_selection(s) && n_blocks == 0)
        throw ed::EmptySelection(std::string(verb) + ": the selection matches no block: no star of the "
                                 "requested Sz sectors has that momentum, star index or little-group irrep");
}

/// The same after the spin-tower filter: under total_spin a selected block can hold no state of
/// the requested spin (every state of it belongs to a higher multiplet).
inline void require_some_level(const Spec& s, bool none, const char* verb) {
    if (has_selection(s) && none)
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
inline const char* no_kernel_reason(bool w_block) {
    return w_block ? "is an isotypic (W) block, which has no device kernel" : "has no device kernel";
}

/// A thermal average under a spec that restricts the sectors (a total spin, one Sz sector or
/// parity, a momentum or irrep selection) describes that restricted ensemble, not the canonical
/// one: a ("restricted_ensemble", ...) diagnostic says which. Nothing when nothing is restricted.
inline void note_restricted_ensemble(const Spec& s, Diagnostics& out, const char* verb) {
    std::vector<std::string> parts;
    if (s.two_S >= 0)
        parts.push_back("total spin S = " + (s.two_S % 2 ? std::to_string(s.two_S) + "/2"
                                                          : std::to_string(s.two_S / 2)));
    if (s.n_up >= 0) parts.push_back("one Sz sector");
    if (s.sz_parity >= 0) parts.push_back("one Sz parity");
    if (!s.only_k0.empty() || !s.only_momentum.empty()) parts.push_back("the selected momenta");
    if (!s.only_irrep.empty() || !s.only_irrep_chars.empty()) parts.push_back("the selected irreps");
    if (parts.empty()) return;
    std::string what = parts.front();
    for (std::size_t i = 1; i < parts.size(); ++i) what += ", " + parts[i];
    out.emplace_back("restricted_ensemble", std::string(verb) + ": the averages run over " + what +
                                                " only: a restricted ensemble, not the canonical one");
}

/// fn(cx, tr_on, star) for every star of one subspace, one star resident at a time. Returns the
/// number of blocks handed to fn, after the selection.
template <class Fn>
std::size_t walk(const ::Operator& H, int n_sites, const Spec& s, const ed::solvers::LittleGroupOptions& opt,
                 Fn&& fn) {
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
        StarBuild sb = build_star_blocks(H, cx, tr_on, k0, members, opt, false,
                                         nullptr, nullptr, nullptr);
        if (tr_on && members.size() > residue_star.at(k0))
            for (auto& bi : sb.blocks) bi->tag.tr_folded = true;
        sb.t_build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count();
        sb.info.momentum = momentum_of(k0);
        // build_star_blocks solved only the wanted irreps already; the filter states the contract.
        if (!s.only_irrep_chars.empty())
            sb.blocks.erase(std::remove_if(sb.blocks.begin(), sb.blocks.end(), [&](const auto& bi) {
                return !meets(s.only_irrep_chars, [&](int i) { return irrep_char(sb.info, bi->tag.irrep, i); });
            }), sb.blocks.end());
        n_blocks += sb.blocks.size();
        fn(cx, tr_on, sb);
    }
    return n_blocks;
}

}  // namespace ed::sectors::detail
