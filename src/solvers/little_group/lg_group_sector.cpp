// Group-sector lane (include/ed/solvers/group_sector.h): a symmetric basis under the FULL little group of a block,
// assembled from the engine's existing pieces (orbit table, closed-form norms, rep-sector matvec + reduced CSR,
// little-group block eigensolver). The lg_detail helpers are shared with the build_star_blocks fast path
// (lg_stars.cpp), which makes this lane the engine default for one-dimensional irreps.
#include "lg_internal.h"

#include <ed/solvers/group_sector.h>
#include <ed/symmetry/orbit_table.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::solvers {

namespace {
double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void check_group_args(const std::vector<std::vector<int>>& perms, int n_sites, int n_up, bool flip) {
    if (perms.empty()) throw std::invalid_argument("group sector: empty group");
    if (n_sites <= 0 || n_sites > 64) throw std::invalid_argument("group sector: need 0 < n_sites <= 64");
    for (const auto& p : perms)
        if (static_cast<int>(p.size()) != n_sites)
            throw std::invalid_argument("group sector: every permutation must act on n_sites sites");
    if (flip && 2 * n_up != n_sites)
        throw std::invalid_argument("group sector: the spin flip needs n_up = N/2");
    if ((flip ? 2 : 1) * perms.size() > 65535)
        throw std::invalid_argument("group sector: |G| too large for the uint16 stabiliser ids");
}
}  // namespace

namespace lg_detail {

ed::symmetry::OrbitTable
group_orbit_table(const std::vector<std::vector<int>>& perms, int n_sites, int n_up, bool flip) {
    check_group_args(perms, n_sites, n_up, flip);
    const ed::symmetry::CompiledGroup cg = flip
        ? ed::symmetry::make_flip_extended_group_from_perms(perms, static_cast<std::uint64_t>(n_sites))
        : ed::symmetry::CompiledGroup::from_permutations(perms, n_sites);
    return ed::symmetry::build_orbit_table_fixed_sz_streaming(static_cast<std::uint64_t>(n_sites), n_up, cg);
}

ed::symmetry::RepSectorData
group_sector_from_table(const ed::symmetry::OrbitTable& tab, const std::vector<std::vector<int>>& perms,
                        int n_sites, int n_up, bool flip, const std::vector<Complex>& characters) {
    const std::size_t G  = perms.size();
    const std::size_t Gx = flip ? 2 * G : G;
    if (characters.size() != Gx)
        throw std::invalid_argument("group sector: characters must have length |G| (2|G| with flip), got "
                                    + std::to_string(characters.size()) + " for |G| = " + std::to_string(G));
    ed::symmetry::RepSectorData rd;
    rd.n_sites    = n_sites;
    rd.n_up       = n_up;
    rd.group_size = static_cast<int>(Gx);
    rd.characters = characters;
    rd.perms_flat.reserve(Gx * static_cast<std::size_t>(n_sites));
    for (const auto& p : perms) rd.perms_flat.insert(rd.perms_flat.end(), p.begin(), p.end());
    if (flip) {
        // same element layout as make_flip_extended_group_from_perms / build_k_sector: [g_0..g_{G-1}, g_0 F, ..]
        for (const auto& p : perms) rd.perms_flat.insert(rd.perms_flat.end(), p.begin(), p.end());
        const std::uint64_t all_ones = (n_sites >= 64) ? ~0ULL : ((1ULL << n_sites) - 1ULL);
        rd.flip_masks.assign(Gx, 0ULL);
        for (std::size_t g = G; g < Gx; ++g) rd.flip_masks[g] = all_ones;
    }
    rd.reps.reserve(tab.reps.size());
    rd.inv_norms.reserve(tab.reps.size());
    for (std::size_t i = 0; i < tab.reps.size(); ++i) {
        const double nsq = ed::symmetry::projected_norm_sq_stab(tab.stabilizer_of(i), characters);
        if (nsq <= 1e-12) continue;
        rd.reps.push_back(tab.reps[i]);
        rd.inv_norms.push_back(1.0 / std::sqrt(nsq));
    }
    return rd;
}

std::vector<Complex>
lift_group_vector(const ed::symmetry::RepSectorData& g, const ed::symmetry::RepSectorData& k, const Complex* v) {
    if (g.n_sites != k.n_sites || g.n_up != k.n_up)
        throw std::invalid_argument("lift_group_vector: sectors differ in n_sites / n_up");
    const auto pg = g.make_policy();
    const auto pk = k.make_policy();
    // The engine normalises a rep state as inv_norm * sum_g chi(g)^* g|s> (no 1/sqrt|G|), so the same physical state
    // carries coefficients larger by sqrt(|G|/|H|) in the subgroup basis: rescale to keep it normalised.
    const double scale = std::sqrt(static_cast<double>(k.group_size) / static_cast<double>(g.group_size));
    const std::size_t n = k.reps.size();
    std::unique_ptr<Complex[]> buf(new Complex[n]);        // no serial zero fill: the parallel loop first-touches
    long long bad = 0;
    #pragma omp parallel for reduction(+ : bad) schedule(static)
    for (long long jj = 0; jj < static_cast<long long>(n); ++jj) {
        const std::size_t j = static_cast<std::size_t>(jj);
        const std::uint64_t st = k.reps[j];
        Complex pgp, pkp;
        const std::int64_t i  = pg.index_and_projection(st, pgp);
        const std::int64_t j2 = pk.index_and_projection(st, pkp);
        if (j2 != static_cast<std::int64_t>(j) || std::abs(pkp) == 0.0) { ++bad; buf[j] = Complex(0, 0); continue; }
        // conj convention: pinned by group_convert_test (complex characters reproduce <H> only this way)
        buf[j] = (i < 0) ? Complex(0, 0) : v[static_cast<std::size_t>(i)] * (std::conj(pgp) / std::conj(pkp)) * scale;
    }
    if (bad) throw std::runtime_error("lift_group_vector: " + std::to_string(bad)
                                      + " target reps are not their own representative (inconsistent sectors)");
    return std::vector<Complex>(buf.get(), buf.get() + n);
}

}  // namespace lg_detail

ed::symmetry::RepSectorData
build_group_sector(const std::vector<std::vector<int>>&     perms,
                   int                                      n_sites,
                   int                                      n_up,
                   bool                                     flip,
                   const std::vector<std::complex<double>>& characters)
{
    const ed::symmetry::OrbitTable tab = lg_detail::group_orbit_table(perms, n_sites, n_up, flip);
    return lg_detail::group_sector_from_table(tab, perms, n_sites, n_up, flip, characters);
}

GroupSectorSolveResult
solve_group_sector(const ::Operator&                                  op,
                   std::shared_ptr<const ed::symmetry::RepSectorData> rd,
                   const GroupSectorSolveOptions&                     opt)
{
    GroupSectorSolveResult out;
    if (!rd || rd->reps.empty()) return out;
    const auto t0 = std::chrono::steady_clock::now();
    auto data = std::make_shared<ed::symmetry::RepSectorData>(*rd);
    data->build_perm_lut();                                  // the shared-data constructor requires the LUT
    const lg_detail::RepSectorMatVec hk(op, std::shared_ptr<const ed::symmetry::RepSectorData>(data));
    out.dim     = hk.dim();
    out.t_setup = seconds_since(t0);

    const auto t1 = std::chrono::steady_clock::now();
    bool conv = false;
    auto [ev, vv] = lg_detail::solve_block_eigenpairs(hk, opt.levels, opt.dense_max_dim, opt.block_size, &conv);
    out.t_solve   = seconds_since(t1);
    out.converged = conv;
    out.energies  = ev;

    const std::size_t n = hk.dim();
    std::vector<std::complex<double>> w(n);
    for (std::size_t j = 0; j < vv.size(); ++j) {
        hk.apply(vv[j].data(), w.data(), n);
        double r2 = 0.0;
        #pragma omp parallel for reduction(+ : r2) schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const auto d = w[static_cast<std::size_t>(i)] - ev[j] * vv[j][static_cast<std::size_t>(i)];
            r2 += std::norm(d);
        }
        out.residuals.push_back(std::sqrt(r2));
    }
    if (opt.return_vectors) out.vectors = std::move(vv);
    return out;
}

std::vector<std::complex<double>>
convert_group_vector(const std::vector<std::complex<double>>& v,
                     const ed::symmetry::RepSectorData&       src,
                     const ed::symmetry::RepSectorData&       dst,
                     bool                                     conjugate)
{
    if (v.size() != src.reps.size())
        throw std::invalid_argument("convert_group_vector: vector length != source sector dimension");
    auto s = std::make_shared<ed::symmetry::RepSectorData>(src); s->build_perm_lut();
    auto d = std::make_shared<ed::symmetry::RepSectorData>(dst); d->build_perm_lut();
    if (conjugate) return lg_detail::lift_group_vector(*s, *d, v.data());
    // conjugate = false is kept only so the convention test can show it fails with complex characters
    auto w = lg_detail::lift_group_vector(*s, *d, v.data());
    const auto ps = s->make_policy(); const auto pd = d->make_policy();
    #pragma omp parallel for schedule(static)
    for (long long jj = 0; jj < static_cast<long long>(w.size()); ++jj) {
        const std::size_t j = static_cast<std::size_t>(jj);
        Complex a, b;
        const std::int64_t i = ps.index_and_projection(d->reps[j], a);
        pd.index_and_projection(d->reps[j], b);
        if (i >= 0 && std::abs(a) > 0 && std::abs(b) > 0)
            w[j] *= (a / b) / (std::conj(a) / std::conj(b));
    }
    return w;
}

}  // namespace ed::solvers
