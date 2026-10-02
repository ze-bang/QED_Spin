#pragma once
// =============================================================================
// src/engine/walk.h -- the block walk shared by the sector-resolved
// verbs (eigs.cpp, thermal.cpp, dynamics.cpp, expect.cpp): engine options for one subspace, and the star-by-star walk.
// Private to the little-group engine.
// =============================================================================

#include "internal.h"

#include <ed/sectors/sectors.h>
#include <ed/ops/casimir.h>
#include <ed/ops/casimir_projector.h>
#include <ed/ops/su2.h>

#include <algorithm>
#include <array>
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
    // subspaces() already enforced 'require' against H. A subspace the flip maps onto a
    // different one gets the symmetry through the mirror fold, so inside it the engine
    // may only engage the flip where the subspace is its own image.
    o.spin_flip     = (s.spin_flip == 1 && sub.mirror == 2) ? -1 : s.spin_flip;
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

inline BlockOp block_operator(const Spec& s, int n_sites, const Subspace& sub,
                              const ed::solvers::lg_detail::StarBuild& sb,
                              const std::shared_ptr<ed::solvers::BlockData>& bi,
                              const std::shared_ptr<::Operator>& s2_carrier,
                              Device device = Device::Cpu) {
    using namespace ed::solvers::lg_detail;
    BlockOp b;
    b.op = std::shared_ptr<const ed::LinearOperator>(bi, &block_mv(*bi));
    b.multiplicity = bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
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
    } else {
        auto s2k = std::make_shared<RepSectorMatVec>(*s2_carrier, sb.hk->rep_data_ptr());
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
    b.multiplicity *= static_cast<std::uint64_t>(s.two_S + 1);
    return b;
}

/// Dense spectra of many blocks: on the host one block at a time, or -- on a device lane --
/// materialised as the walk visits them and solved in one batched cuSOLVER call at the end
/// (the walk streams stars, so the matrices are the only thing that outlives a star).
/// place(Task::DenseBatch) chooses each entry's lane.
class DenseBatch {
public:
    explicit DenseBatch(Device device) : device_(device) {}

    /// Queue (or, on the host, solve now) the spectrum of `mv`; returns the entry index.
    std::size_t add(const ed::LinearOperator& mv) {
        using namespace ed::solvers::lg_detail;
        const std::size_t id = spectra_.size();
        spectra_.emplace_back();
        lanes_.push_back(ed::place(device_, {ed::Task::DenseBatch, mv.dim()}));
        if (!ed::on_device(lanes_.back())) {
            spectra_.back() = solve_block_full(mv);
            return id;
        }
        const Eigen::MatrixXcd Hb = materialize(mv);
        const std::size_t nb = static_cast<std::size_t>(Hb.rows());
        packed_.offset.push_back(packed_.data.size());
        packed_.block_dim.push_back(static_cast<int>(nb));
        packed_.block_irrep_dim.push_back(1);
        packed_.data.insert(packed_.data.end(), Hb.data(), Hb.data() + nb * nb);   // column-major
        queued_.push_back(id);
        return id;
    }

    /// Solve everything queued; afterwards spectrum(id) is valid for every entry.
    void solve() {
        if (queued_.empty()) return;
#ifdef WITH_CUDA
        const std::vector<double> ev = ed::solvers::lg_blocks_batched_eigenvalues_gpu(packed_);
#else
        const std::vector<double> ev;   // unreachable: nothing is queued without a device
#endif
        std::size_t off = 0;
        for (std::size_t q = 0; q < queued_.size(); ++q) {
            const std::size_t nb = static_cast<std::size_t>(packed_.block_dim[q]);
            spectra_[queued_[q]].assign(ev.begin() + static_cast<long>(off),
                                        ev.begin() + static_cast<long>(off + nb));
            off += nb;
        }
        device_blocks_ += queued_.size();
        queued_.clear();
        packed_ = {};
    }

    [[nodiscard]] const std::vector<double>& spectrum(std::size_t id) const { return spectra_[id]; }
    [[nodiscard]] ed::Lane lane(std::size_t id) const { return lanes_[id]; }
    [[nodiscard]] std::size_t device_blocks() const noexcept { return device_blocks_; }

private:
    Device device_;
    std::vector<ed::Lane>             lanes_;
    ed::solvers::LgBlocksPacked       packed_;
    std::vector<std::size_t>          queued_;
    std::vector<std::vector<double>>  spectra_;
    std::size_t                       device_blocks_ = 0;
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

inline int sz_shift(int op) { return op == 0 ? -1 : (op == 1 ? 1 : 0); }

// One product of single-site operators; the factors of distinct sites commute, so they are
// kept in site order and equal products merge.
struct AvgTerm {
    std::array<std::uint8_t, 3>  op{};
    std::array<std::uint64_t, 3> site{};
    int n = 0;
};

enum class Keep { All, Sz, Parity };

// O averaged over the site permutations `G` (and the Sz flip), without the terms that change
// what `keep` conserves. Equal terms are merged, so the result is no larger than it must be.
inline ::Operator averaged(const ::Operator& O, const std::vector<Perm>& G, bool flip, Keep keep) {
    std::vector<std::pair<AvgTerm, Complex>> terms;
    for (const auto& t : O.transform_data_) {
        AvgTerm x;
        x.n = t.is_two_body ? 2 : 1;
        x.op   = {t.op_type, t.op_type_2, 0};
        x.site = {t.site_index, t.site_index_2, 0};
        terms.push_back({x, t.coefficient});
    }
    for (const auto& t : O.three_body_data_) {
        AvgTerm x;
        x.n = 3;
        x.op   = {t.op_type_1, t.op_type_2, t.op_type_3};
        x.site = {t.site_index_1, t.site_index_2, t.site_index_3};
        terms.push_back({x, t.coefficient});
    }
    const double w = 1.0 / static_cast<double>(G.size() * (flip ? 2 : 1));
    std::map<std::array<std::uint64_t, 7>, Complex> acc;
    for (const auto& [t, c] : terms) {
        int shift = 0;
        for (int f = 0; f < t.n; ++f) shift += sz_shift(t.op[static_cast<std::size_t>(f)]);
        if ((keep == Keep::Sz && shift != 0) || (keep == Keep::Parity && shift % 2 != 0)) continue;
        for (int fl = 0; fl < (flip ? 2 : 1); ++fl)
            for (const Perm& g : G) {
                std::array<std::pair<std::uint64_t, std::uint8_t>, 3> f{};
                Complex coef = c * w;
                for (int k = 0; k < t.n; ++k) {
                    std::uint8_t op = t.op[static_cast<std::size_t>(k)];
                    if (fl) {                       // S+ <-> S-, Sz -> -Sz
                        if (op == 2) coef = -coef;
                        else op = static_cast<std::uint8_t>(1 - op);
                    }
                    f[static_cast<std::size_t>(k)] = {static_cast<std::uint64_t>(g[t.site[static_cast<std::size_t>(k)]]), op};
                }
                bool distinct = true;
                for (int a = 0; a < t.n; ++a)
                    for (int b = a + 1; b < t.n; ++b)
                        if (f[static_cast<std::size_t>(a)].first == f[static_cast<std::size_t>(b)].first) distinct = false;
                if (distinct) std::sort(f.begin(), f.begin() + t.n);
                std::array<std::uint64_t, 7> key{static_cast<std::uint64_t>(t.n)};
                for (int k = 0; k < t.n; ++k) {
                    key[1 + 2 * static_cast<std::size_t>(k)] = f[static_cast<std::size_t>(k)].first;
                    key[2 + 2 * static_cast<std::size_t>(k)] = f[static_cast<std::size_t>(k)].second;
                }
                acc[key] += coef;
            }
    }
    ::Operator out(O.getNumBits(), O.getSpin());
    for (const auto& [k, c] : acc) {
        if (std::abs(c) < 1e-15) continue;
        const auto op = [&](int i) { return static_cast<std::uint8_t>(k[2 + 2 * static_cast<std::size_t>(i)]); };
        const auto st = [&](int i) { return k[1 + 2 * static_cast<std::size_t>(i)]; };
        if (k[0] == 1)      out.addOneBodyTerm(op(0), st(0), c);
        else if (k[0] == 2) out.addTwoBodyTerm(op(0), st(0), op(1), st(1), c);
        else                out.addThreeBodyTerm(op(0), st(0), op(1), st(1), op(2), st(2), c);
    }
    return out;
}

inline ::Operator conjugated(const ::Operator& O) {
    ::Operator out(O.getNumBits(), O.getSpin());
    out.copyTermsFrom(O);
    for (auto& t : out.transform_data_) t.coefficient = std::conj(t.coefficient);
    for (auto& t : out.three_body_data_) t.coefficient = std::conj(t.coefficient);
    return out;
}

/// O averaged over the symmetry group of a Spec, built once per (operator, flip, keep,
/// conjugate). An operator averaged over the symmetries a block uses is block diagonal and
/// has the same trace against any function of H over an ensemble those symmetries preserve.
class Averager {
public:
    Averager(const Spec& s, int n_sites) {
        std::vector<Perm> gens = abelian_or_identity(s, n_sites);
        gens.insert(gens.end(), s.residues.begin(), s.residues.end());
        G_ = close_group(gens, n_sites);
    }
    std::shared_ptr<const ::Operator> get(const ::Operator& O, bool flip, Keep keep, bool conj) {
        auto& slot = cache_[{&O, flip, static_cast<int>(keep), conj}];
        if (!slot) {
            ::Operator a = averaged(O, G_, flip, keep);
            slot = std::make_shared<::Operator>(conj ? conjugated(a) : std::move(a));
        }
        return slot;
    }

private:
    std::vector<Perm> G_;
    std::map<std::tuple<const ::Operator*, bool, int, bool>, std::shared_ptr<::Operator>> cache_;
};

/// A block-diagonal (averaged) operator restricted to block `bi` of star `sb`, in the
/// basis the block's H acts on; with `device` it may bind to a CUDA backend.
inline std::shared_ptr<const ed::LinearOperator>
block_observable(const ::Operator& A, const ed::solvers::lg_detail::StarBuild& sb,
                 const std::shared_ptr<ed::solvers::BlockData>& bi, bool device) {
    using namespace ed::solvers::lg_detail;
    auto rep = std::make_shared<RepSectorMatVec>(A, bi->gop ? bi->gsec : sb.hk->rep_data_ptr());
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
    // A co-group character names sigma alone; time reversal would fold sigma* into its block.
    if (!s.only_irrep_chars.empty()) tr_on = false;
    const std::set<int> only(s.only_k0.begin(), s.only_k0.end());
    auto momentum_of = [&](int k_ext) -> const std::vector<Complex>& {
        return cx.giA.irreps[static_cast<std::size_t>(k_ext % cx.n_irr_raw)].character;
    };
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
        const auto t_build = std::chrono::steady_clock::now();
        StarBuild sb = build_star_blocks(H, cx, tr_on, k0, members, opt, false,
                                         nullptr, nullptr, nullptr);
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
