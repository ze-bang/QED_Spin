// The spin-S tower of a fixed-Sz block (P6.5): valence-bond starts inside it, the certification of
// eigenpairs through S^2, and the penalty operator of the fallback solve. internal.h has the why.
#include "internal.h"

#include <ed/ops/casimir_projector.h>
#ifdef WITH_CUDA
#include <ed/gpu/cuda_backend.cuh>
#endif

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::solvers::lg_detail {

namespace {

double lam_of(int two_S) { return 0.25 * two_S * (two_S + 2); }

// One valence-bond state: singlets (p up, q down: +1; p down, q up: -1) on the pairs, the free sites
// in their symmetric (Dicke) state, i.e. any of their configurations with the same amplitude. A state
// s is in its support when every pair is antiparallel (the free sites then hold the remaining up
// spins); the swap permutation p <-> q of all pairs, by bytes, tests that at once:
// ((s ^ swap(s)) & pmask) == pmask, and the sign is (-1)^popcount(pmask & ~s).
struct Matching {
    std::uint64_t pmask = 0;
    std::vector<std::uint64_t> swap_lut;   // [byte][value] -> the swapped bits, 64-bit words
    int bytes = 0;
    double coef = 0.0;

    [[nodiscard]] std::uint64_t swapped(std::uint64_t s) const noexcept {
        std::uint64_t r = 0;
        for (int b = 0; b < bytes; ++b)
            r |= swap_lut[static_cast<std::size_t>(b) * 256 + ((s >> (8 * b)) & 0xFFu)];
        return r;
    }
    [[nodiscard]] double amplitude(std::uint64_t s) const noexcept {
        if (((s ^ swapped(s)) & pmask) != pmask) return 0.0;
        return (__builtin_popcountll(pmask & ~s) & 1) ? -coef : coef;
    }
};

Matching random_matching(int N, int two_S, std::mt19937_64& gen) {
    std::vector<int> sites(static_cast<std::size_t>(N));
    std::iota(sites.begin(), sites.end(), 0);
    std::shuffle(sites.begin(), sites.end(), gen);
    std::vector<int> partner(static_cast<std::size_t>(N));
    std::iota(partner.begin(), partner.end(), 0);   // the free sites stay fixed
    Matching m;
    for (int t = two_S; t + 1 < N; t += 2) {
        const int p = sites[static_cast<std::size_t>(t)], q = sites[static_cast<std::size_t>(t) + 1];
        partner[static_cast<std::size_t>(p)] = q;
        partner[static_cast<std::size_t>(q)] = p;
        m.pmask |= std::uint64_t{1} << p;
    }
    m.bytes = (N + 7) / 8;
    m.swap_lut.assign(static_cast<std::size_t>(m.bytes) * 256, 0);
    for (int b = 0; b < m.bytes; ++b)
        for (int v = 0; v < 256; ++v) {
            std::uint64_t r = 0;
            for (int i = 0; i < 8 && 8 * b + i < N; ++i)
                if ((v >> i) & 1) r |= std::uint64_t{1} << partner[static_cast<std::size_t>(8 * b + i)];
            m.swap_lut[static_cast<std::size_t>(b) * 256 + static_cast<std::size_t>(v)] = r;
        }
    std::normal_distribution<double> nd(0.0, 1.0);
    m.coef = nd(gen);
    return m;
}

}  // namespace

double Tower::gap() const {
    const double lam = lambda();
    double g = 0.0;
    for (int ts : towers)
        if (ts != two_S) {
            const double d = std::abs(lam_of(ts) - lam);
            if (g == 0.0 || d < g) g = d;
        }
    return g;
}

bool Tower::highest_weight() const { return 2 * sector->n_up - sector->n_sites == two_S; }

std::vector<Complex> Tower::seed(std::uint64_t s) const {
    const ed::symmetry::RepSectorData& rd = *sector;
    const int N = rd.n_sites;
    if (rd.n_up < 0 || two_S < 0 || two_S > N || (N - two_S) % 2 != 0)
        throw std::invalid_argument("Tower::seed: needs a fixed-Sz sector and a spin S of its sites");
    const int pairs = (N - two_S) / 2;
    const int free_up = rd.n_up - pairs;   // the up spins among the 2S free sites
    if (free_up < 0 || free_up > two_S) return {};   // spin S has no member in this Sz sector
    // A few random matchings with Gaussian weights: a generic spin-S state (one matching alone can be
    // orthogonal to an eigenstate by a symmetry of its own).
    constexpr int kMatchings = 4;
    std::mt19937_64 gen(s ^ 0x7A3E5C1D9B2F4E68ULL);
    std::vector<Matching> M;
    for (int c = 0; c < kMatchings; ++c) M.push_back(random_matching(N, two_S, gen));
    auto phi = [&M](std::uint64_t st) {
        double a = 0.0;
        for (const Matching& m : M) a += m.amplitude(st);
        return a;
    };
    // <b_r|phi>: d = 1, inv_norm sum_g chi(g) phi(g r); d > 1, sum_j conj(C_r[j][a]) sum_g D(g)_0j phi(g r)
    // (the global sqrt(d/|G|) dropped; rep_sector.h has the basis).
    const auto pol = rd.make_policy();
    const int d = rd.irrep_dim;
    const std::size_t dd = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
    const std::size_t n_reps = rd.reps.size();
    std::vector<Complex> u(rd.states());
    double raw = 0.0;   // the weight before the projection: what a vanishing projection is measured against
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 256) reduction(+ : raw)
#endif
    for (long long ii = 0; ii < static_cast<long long>(n_reps); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        Complex z[ed::matvec::kMaxIrrepDim];
        for (int j = 0; j < d; ++j) z[j] = Complex(0, 0);
        for (int g = 0; g < rd.group_size; ++g) {
            const double a = phi(pol.apply_perm(rd.reps[i], g));
            if (a == 0.0) continue;
            raw += a * a;
            if (d == 1) z[0] += rd.characters[static_cast<std::size_t>(g)] * a;
            else
                for (int j = 0; j < d; ++j)
                    z[j] += rd.irrep_D[static_cast<std::size_t>(g) * dd + static_cast<std::size_t>(j)] * a;
        }
        if (d == 1) {
            u[i] = z[0] * rd.inv_norms[i];
            continue;
        }
        const Complex* C = pol.C_of(i);
        const std::uint64_t o = pol.first_state_of(i);
        for (int al = 0; al < pol.rank_of(i); ++al) {
            Complex x(0, 0);
            for (int j = 0; j < d; ++j) x += std::conj(C[j * d + al]) * z[j];
            u[o + static_cast<std::uint64_t>(al)] = x;
        }
    }
    double n2 = 0.0;
    for (const Complex& c : u) n2 += std::norm(c);
    // The projection of a spin-S state on a block that holds none is zero up to the roundoff of its
    // character sums. scale-free: a ratio of norms of one vector
    if (!(n2 > 1e-20 * raw)) return {};
    const double inv = 1.0 / std::sqrt(n2);
    for (Complex& c : u) c *= inv;
    return u;
}

TowerLevels tower_filter(const Tower& t, const ed::LinearOperator& H, const std::vector<double>& values,
                         std::vector<std::vector<Complex>> vectors, double cluster_tol) {
    TowerLevels out;
    const std::size_t m = values.size();
    if (vectors.size() != m) throw std::invalid_argument("tower_filter: one vector per value");
    const double gap = t.gap();
    if (gap == 0.0) {   // the block holds this tower alone
        out.values = values;
        out.vectors = std::move(vectors);
        return out;
    }
    const std::size_t n = t.s2->dim();
    const double lam = t.lambda();
    // ||(S^2 - S(S+1)) psi|| >= gap ||off-tower part of psi||: below this a vector is in the tower to
    // roundoff. scale-free: a fraction of the unit vector's norm, in units of the tower gap
    const double clean = 1e-8 * gap;
    std::vector<double> leak(m);
    std::vector<Complex> r(n);
    for (std::size_t i = 0; i < m; ++i) {
        t.s2->apply(vectors[i].data(), r.data(), n);
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j) s += std::norm(r[j] - lam * vectors[i][j]);
        leak[i] = std::sqrt(s);
    }
    std::unique_ptr<ed::symmetry::LowdinS2Projector> P;   // built for the first mixed cluster
    std::vector<Complex> h(n);
    for (std::size_t a = 0; a < m;) {
        std::size_t b = a + 1;
        while (b < m && values[b] - values[b - 1] <= cluster_tol) ++b;
        bool mixed = false;
        for (std::size_t i = a; i < b; ++i) mixed = mixed || leak[i] > clean;
        if (!mixed) {
            for (std::size_t i = a; i < b; ++i) {
                out.values.push_back(values[i]);
                out.vectors.push_back(std::move(vectors[i]));
            }
            a = b;
            continue;
        }
        // The cluster's spin-S directions: the eigenvectors of G = U^dag P U at 1 (an exact cluster
        // spans tower and off-tower eigenvectors, G's eigenvalues are 1 and 0); each is P U w,
        // with its energy the Rayleigh quotient of H.
        if (!P) P = std::make_unique<ed::symmetry::LowdinS2Projector>(t.s2, t.two_S, t.towers);
        const std::size_t c = b - a;
        std::vector<std::vector<Complex>> PU(c);
        for (std::size_t i = 0; i < c; ++i) {
            PU[i] = vectors[a + i];
            P->project(PU[i].data(), n);
        }
        Eigen::MatrixXcd G(static_cast<Eigen::Index>(c), static_cast<Eigen::Index>(c));
        for (std::size_t i = 0; i < c; ++i)
            for (std::size_t j = 0; j < c; ++j) {
                Complex s(0, 0);
                for (std::size_t q = 0; q < n; ++q) s += std::conj(vectors[a + i][q]) * PU[j][q];
                G(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = s;
            }
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(0.5 * (G + G.adjoint()));
        for (Eigen::Index e = static_cast<Eigen::Index>(c) - 1; e >= 0; --e) {
            const double g = es.eigenvalues()(e);
            // scale-free: eigenvalues of a projector's Gram matrix in unit vectors (0 or 1 when exact)
            if (g < 0.1) { ++out.off; continue; }
            if (g < 0.9) out.ambiguous = true;   // mixed at O(1): neither certifiably in nor out
            std::vector<Complex> v(n, Complex(0, 0));
            for (std::size_t i = 0; i < c; ++i) {
                const Complex w = es.eigenvectors()(static_cast<Eigen::Index>(i), e);
                for (std::size_t q = 0; q < n; ++q) v[q] += w * PU[i][q];
            }
            double vn = 0.0;
            for (const Complex& x : v) vn += std::norm(x);
            const double inv = 1.0 / std::sqrt(vn);
            for (Complex& x : v) x *= inv;
            H.apply(v.data(), h.data(), n);
            Complex e_v(0, 0);
            for (std::size_t q = 0; q < n; ++q) e_v += std::conj(v[q]) * h[q];
            out.values.push_back(std::real(e_v));
            out.vectors.push_back(std::move(v));
        }
        a = b;
    }
    // Ascending, vectors aligned.
    std::vector<std::size_t> order(out.values.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t x, std::size_t y) { return out.values[x] < out.values[y]; });
    TowerLevels sorted;
    sorted.off = out.off;
    sorted.ambiguous = out.ambiguous;
    for (std::size_t i : order) {
        sorted.values.push_back(out.values[i]);
        sorted.vectors.push_back(std::move(out.vectors[i]));
    }
    return sorted;
}

namespace {

// H + mu f(S^2), f(x) = x - S(S+1) at the highest weight (S- S+ >= 2(S + 1) off the tower there) and
// (x - S(S+1))^2 elsewhere, mu such that f >= its smallest off-tower value lifts every off-tower
// state 2 s_H above where it was: above the whole band.
class TowerPenalty final : public ed::LinearOperator {
public:
    TowerPenalty(const ed::LinearOperator& H, const Tower& t)
        : h_(&H), s2_(t.s2), lam_(t.lambda()), linear_(t.highest_weight()) {
        const double gap = t.gap();
        double top = 0.0;   // the largest |S'(S'+1) - S(S+1)| in the block
        for (int ts : t.towers) top = std::max(top, std::abs(lam_of(ts) - lam_));
        const double fmin = linear_ ? gap : gap * gap;
        const double fmax = linear_ ? top : top * top;
        const double sH = ed::numerics::scale_or_one(h_->norm_bound());
        mu_ = fmin > 0.0 ? 2.0 * sH / fmin : 0.0;
        bound_ = sH + mu_ * fmax;
    }

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        h_->apply(in, out, n);
        std::vector<Complex> a(n);
        s2_->apply(in, a.data(), n);
        for (std::size_t i = 0; i < n; ++i) a[i] -= lam_ * in[i];
        if (!linear_) {
            std::vector<Complex> b(n);
            s2_->apply(a.data(), b.data(), n);
            for (std::size_t i = 0; i < n; ++i) a[i] = b[i] - lam_ * a[i];
        }
        for (std::size_t i = 0; i < n; ++i) out[i] += mu_ * a[i];
    }
    [[nodiscard]] std::size_t dim() const override { return h_->dim(); }
    [[nodiscard]] double norm_bound() const override { return bound_; }
    [[nodiscard]] std::string description() const override { return "TowerPenalty(H + mu f(S^2))"; }

#ifdef WITH_CUDA
    [[nodiscard]] bool has_device_kernel() const override {
        return h_->has_device_kernel() && s2_->has_device_kernel();
    }
    [[nodiscard]] MatvecFn bind_cuda() const override {
        if (!has_device_kernel()) return ed::LinearOperator::bind_cuda();   // throws DeviceUnsupported
        struct Scratch {
            ed::matvec::CudaBackend be;
            ed::matvec::Backend::UniqueVec a, b;
            explicit Scratch(std::size_t n) : a(be.make_zero_vector(n)), b(be.make_zero_vector(n)) {}
        };
        auto st = std::make_shared<Scratch>(dim());
        return [this, st, hd = h_->bind_cuda(), sd = s2_->bind_cuda()](const Complex* in, Complex* out,
                                                                       std::size_t n) {
            hd(in, out, n);
            sd(in, st->a.get(), n);
            st->be.axpy(Complex(-lam_), in, st->a.get(), n);
            if (!linear_) {
                sd(st->a.get(), st->b.get(), n);
                st->be.axpy(Complex(-lam_), st->a.get(), st->b.get(), n);
                st->be.axpy(Complex(mu_), st->b.get(), out, n);
            } else {
                st->be.axpy(Complex(mu_), st->a.get(), out, n);
            }
        };
    }
#endif

private:
    const ed::LinearOperator* h_;
    std::shared_ptr<const ed::LinearOperator> s2_;
    double lam_ = 0.0, mu_ = 0.0, bound_ = 0.0;
    bool linear_ = true;
};

}  // namespace

std::unique_ptr<const ed::LinearOperator> tower_penalty(const ed::LinearOperator& H, const Tower& t) {
    return std::make_unique<TowerPenalty>(H, t);
}

}  // namespace ed::solvers::lg_detail
