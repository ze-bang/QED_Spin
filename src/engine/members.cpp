// =============================================================================
// src/engine/members.cpp -- a level's degenerate multiplet as unit vectors of momentum sectors
// of the abelian group (walk.h, members_of). Dynamics solves its ground manifold on the caller's
// folded Spec (point group, flip, time reversal) and needs every member where the probes' rows
// live: in momentum sectors (audit P5-dynamics-01).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "walk.h"

#include <ed/basis/bits.h>
#include <ed/matvec/cpu_backend.h>

#include <cmath>
#include <stdexcept>

namespace ed::sectors::detail {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

using Policy = ed::matvec::basis::RepSymmetryBasisPolicy;
constexpr int kMaxIrrepDim = 8;

// <s|v> for v in sector rd, as expand() writes it: one pass over the sector's group.
Complex amplitude(const ed::symmetry::RepSectorData& rd, const Policy& pol, const Complex* v, std::uint64_t s) {
    const int d = rd.irrep_dim;
    if (d == 1) {
        Complex pr;
        const std::int64_t i = pol.index_and_projection(s, pr);
        return i < 0 ? Complex(0, 0) : v[i] * std::conj(pr);
    }
    Complex A[kMaxIrrepDim * kMaxIrrepDim];   // sum_{g: g s = rep} D(g)^T; expand() sums its row 0
    const std::int64_t i = pol.index_and_matrix(s, A);
    if (i < 0) return Complex(0, 0);
    const auto k = static_cast<std::uint64_t>(i);
    const Complex* C = pol.C_of(k);
    const std::uint64_t o = pol.first_state_of(k);
    const int rank = pol.rank_of(k);
    Complex amp(0, 0);
    for (int j = 0; j < d; ++j) {
        Complex y(0, 0);
        for (int al = 0; al < rank; ++al) y += C[j * d + al] * v[o + static_cast<std::uint64_t>(al)];
        amp += y * A[j];
    }
    return amp;
}

// A unitary or antiunitary map of basis states, U|s> = eps(s)|map(s)> (antiunitary: the amplitude
// conjugated too): a residue, the spin flip, conjugation K or time reversal Theta.
struct StateMap {
    std::vector<int> perm, inv;       // a site permutation and its inverse (empty: none)
    std::uint64_t    mask  = 0;       // the flip (0: none)
    bool             theta = false;   // eps(s) = (-1)^{n_down(s)}
    bool             anti  = false;
    int              n_sites = 0;

    [[nodiscard]] std::uint64_t forward(std::uint64_t s) const {
        if (!perm.empty()) s = applyPermutation(s, perm);
        return s ^ mask;
    }
    // <t|U v> from v's amplitudes.
    template <class Amp>
    [[nodiscard]] Complex image(std::uint64_t t, const Amp& amp) const {
        t ^= mask;
        const std::uint64_t s = perm.empty() ? t : applyPermutation(t, inv);
        Complex a = amp(s);
        if (anti) a = std::conj(a);
        if (theta && (n_sites - __builtin_popcountll(s)) % 2 != 0) a = -a;
        return a;
    }
    // The subspace U maps `sub` to.
    [[nodiscard]] Subspace target(const Subspace& sub) const {
        if (mask == 0) return sub;
        if (sub.n_up >= 0) return {n_sites - sub.n_up, -1, 1};
        if (sub.sz_parity >= 0) return {-1, (n_sites + sub.sz_parity) % 2, 1};
        return sub;
    }
};

StateMap residue_map(const Perm& p, int n_sites) {
    StateMap U;
    U.n_sites = n_sites;
    U.perm = p;
    U.inv.assign(p.size(), 0);
    for (std::size_t i = 0; i < p.size(); ++i) U.inv[static_cast<std::size_t>(p[i])] = static_cast<int>(i);
    return U;
}

// The state of v's leading orbit where |U v| is largest: the probe that reads off U v's momentum.
std::uint64_t probe_state(const BlockVector& from, const Policy& pol, const StateMap& U) {
    const auto& rd = *from.basis;
    const auto& v = from.amplitudes;
    // The representative whose states carry the largest amplitude.
    std::size_t lead = 0;
    double top = -1.0;
    for (std::size_t a = 0; a < rd.reps.size(); ++a) {
        const std::uint64_t o = rd.irrep_dim == 1 ? a : pol.first_state_of(a);
        const int rank = rd.irrep_dim == 1 ? 1 : pol.rank_of(a);
        for (int al = 0; al < rank; ++al) {
            const double m = std::abs(v[o + static_cast<std::uint64_t>(al)]);
            if (m > top) { top = m; lead = a; }
        }
    }
    std::uint64_t best_state = 0;
    double best = 0.0;
    for (int g = 0; g < rd.group_size; ++g) {
        const std::uint64_t s = pol.apply_perm(rd.reps[lead], g);
        const double m = std::abs(amplitude(rd, pol, v.data(), s));
        if (m > best) { best = m; best_state = s; }
    }
    if (!(best > 0.0)) throw std::logic_error("members_of: a vector has no amplitude on its leading orbit");
    return U.forward(best_state);
}

// U v gathered into the momentum sector of A that holds it; `from` is any sector (a group sector of
// any irrep dimension, a flip-extended or a plain momentum sector). The momentum is read off at the
// probe state t, (U v)(a t) = conj(chi(a)) (U v)(t) for every a (expand()'s convention), and
// checked on t's orbit in the gathered vector, so a wrong convention fails loudly.
Member carry(const BlockVector& from, const Subspace& from_sub, const StateMap& U, MemberSectors& ms) {
    const auto& rd = *from.basis;
    const Policy pol = rd.make_policy();
    const Complex* v = from.amplitudes.data();
    auto amp = [&](std::uint64_t s) { return amplitude(rd, pol, v, s); };
    auto phi = [&](std::uint64_t t) { return U.image(t, amp); };
    const std::uint64_t t = probe_state(from, pol, U);
    const Complex p0 = phi(t);
    std::vector<Complex> chi(ms.A.size());
    for (std::size_t a = 0; a < ms.A.size(); ++a) chi[a] = std::conj(phi(applyPermutation(t, ms.A[a])) / p0);

    const Subspace sub = U.target(from_sub);
    auto& sectors = ms.by_sub[{sub.n_up, sub.sz_parity}];
    if (!sectors) sectors = std::make_unique<MomentumSectors>(ms.H, ms.A, ms.n_sites, sub);
    const auto to = sectors->of(chi);
    if (!to) throw std::logic_error("members_of: a member has no momentum of the abelian group");
    const Policy pt = to->make_policy();
    const std::size_t n = to->reps.size();
    std::vector<Complex> y(n);
    #pragma omp parallel for schedule(dynamic, 1024) if(n > 8192)
    for (long long jj = 0; jj < static_cast<long long>(n); ++jj) {
        const auto j = static_cast<std::size_t>(jj);
        Complex pr;
        const std::int64_t i = pt.index_and_projection(to->reps[j], pr);
        y[j] = (i < 0 || pr == Complex(0, 0)) ? Complex(0, 0) : phi(to->reps[j]) / std::conj(pr);
    }
    double n2 = 0.0;
    for (const auto& c : y) n2 += std::norm(c);
    if (!(n2 > 0.0)) throw std::logic_error("members_of: a member vanished in its momentum sector");
    // The gathered vector must reproduce U v on t's orbit (a momentum eigenstate of the right sector).
    for (std::size_t a = 0; a < ms.A.size(); ++a) {
        const std::uint64_t s = applyPermutation(t, ms.A[a]);
        // scale-free: two amplitudes of one vector against its largest one
        if (std::abs(amplitude(*to, pt, y.data(), s) - phi(s)) > 1e-8 * std::abs(p0))
            throw std::logic_error("members_of: a member is not a momentum eigenstate of its sector");
    }
    const double scale = 1.0 / std::sqrt(n2);
    for (auto& c : y) c *= scale;
    return {sub, {to, std::move(y)}};
}

}  // namespace

MomentumSectors::MomentumSectors(const ::Operator& H, const std::vector<Perm>& A, int n_sites, const Subspace& sub) {
    LittleGroupOptions o;
    o.n_up = sub.n_up;
    o.sz_parity = sub.n_up >= 0 ? -1 : sub.sz_parity;
    o.spin_flip = 0;
    o.time_reversal = 0;
    bool tr_on = false;
    make_engine_context(H, A, {}, n_sites, o, cx_, tr_on);
}

std::shared_ptr<const ed::symmetry::RepSectorData> MomentumSectors::of(const std::vector<Complex>& chi) {
    for (int k = 0; k < cx_.n_irr_raw; ++k) {
        const auto& c = cx_.giA.irreps[static_cast<std::size_t>(k)].character;
        bool same = c.size() == chi.size();
        for (std::size_t a = 0; same && a < c.size(); ++a)
            // scale-free: unit-modulus characters (group data, not energies)
            same = std::abs(c[a] - chi[a]) <= 1e-6;
        if (!same) continue;
        auto& p = built_[k];
        if (!p) p = share_rep_sector(build_k_sector(cx_, k, cx_.n_up));
        return p;
    }
    return nullptr;
}

std::vector<Member>
members_of(const Level& L, const BlockVector& v, std::uint64_t count, const Spec& s, MemberSectors& ms) {
    if (!v.basis) throw std::invalid_argument("members_of: the level has no vector");
    if (v.basis->irrep_dim > kMaxIrrepDim)
        throw std::invalid_argument("members_of: irreps of dimension above " + std::to_string(kMaxIrrepDim));
    const int N = ms.n_sites;
    const std::uint64_t full = N >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << N) - 1);
    // multiplet()'s operations: the residues, the level's antiunitary fold, its flip / Theta mirror.
    std::vector<StateMap> ops;
    for (const auto& p : s.residues) ops.push_back(residue_map(p, N));
    StateMap K;
    K.n_sites = N;
    K.anti = true;
    StateMap F;
    F.n_sites = N;
    F.mask = full;
    StateMap Th = F;
    Th.theta = true;
    Th.anti = true;
    const bool theta_fold = L.fold == Antiunitary::Theta;
    if (L.tag.tr_folded) ops.push_back(theta_fold ? Th : K);
    if (theta_fold && L.mirror == 2 && !L.tag.tr_folded) ops.push_back(Th);
    else if (L.mirror == 2 || L.tag.flip_parity >= 0) ops.push_back(F);

    StateMap I;
    I.n_sites = N;
    const Subspace sub0{v.basis->n_up, v.basis->n_up >= 0 ? -1 : L.tag.sz_parity, 1};
    std::vector<Member> out;
    out.push_back(carry(v, sub0, I, ms));
    // Breadth first; Gram-Schmidt twice within a sector (members of two sectors are orthogonal).
    const auto& be = ed::matvec::default_cpu_backend();
    auto add = [&](Member m) {
        const std::size_t dim = m.v.amplitudes.size();
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& b : out)
                if (b.v.basis == m.v.basis)
                    be.axpy(-be.dot(b.v.amplitudes.data(), m.v.amplitudes.data(), dim), b.v.amplitudes.data(),
                            m.v.amplitudes.data(), dim);
        const double nrm = be.nrm2(m.v.amplitudes.data(), dim);
        if (nrm * nrm < 1e-16) return;
        be.scale(Complex(1.0 / nrm, 0.0), m.v.amplitudes.data(), dim);
        out.push_back(std::move(m));
    };
    for (std::size_t head = 0; head < out.size() && out.size() < count; ++head)
        for (const auto& U : ops) {
            if (out.size() >= count) break;
            add(carry(out[head].v, out[head].sub, U, ms));
        }
    if (out.size() != count)
        throw std::logic_error("members_of: a level's multiplet closes on " + std::to_string(out.size())
                               + " states, its multiplicity is " + std::to_string(count));
    return out;
}

}  // namespace ed::sectors::detail
