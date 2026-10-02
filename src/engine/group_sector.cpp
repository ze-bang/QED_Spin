// Group-sector fast path: a symmetric basis under the FULL little group of a block (translations x little co-group
// x {1, flip}) in a 1-dim representation, so the block dimension is C(N, n_up) / |G| instead of the whole k-sector.
// Assembled from the engine's existing pieces (orbit table, closed-form norms, rep-sector matvec + reduced CSR);
// build_star_blocks (stars.cpp, try_group_path) makes this the engine default for one-dimensional irreps.
#include "internal.h"

#include <ed/basis/orbit_table.h>
#include <ed/basis/symmetry_cache.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::solvers {

namespace {
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

// Through the orbit-table registry: the estimate walk, each survivor's re-walk and the next call
// on the same group reuse the table.
std::shared_ptr<const ed::symmetry::OrbitTable>
group_orbit_table(const std::vector<std::vector<int>>& perms, int n_sites, int n_up, bool flip) {
    check_group_args(perms, n_sites, n_up, flip);
    const ed::symmetry::CompiledGroup cg = flip
        ? ed::symmetry::make_flip_extended_group_from_perms(perms, static_cast<std::uint64_t>(n_sites))
        : ed::symmetry::CompiledGroup::from_permutations(perms, n_sites);
    return ed::symmetry::acquire_orbit_table_fixed_sz_compiled(static_cast<std::uint64_t>(n_sites), n_up, cg);
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
    filter_reps(tab, characters, rd);
    return rd;
}

void filter_reps(const ed::symmetry::OrbitTable& tab, const std::vector<Complex>& characters,
                 ed::symmetry::RepSectorData& rd, std::vector<std::int32_t>* local) {
    const std::size_t n = tab.reps.size();
    int T = 1;
#ifdef _OPENMP
    if (n >= (std::size_t{1} << 14)) T = omp_get_max_threads();
#endif
    const std::size_t C = std::max<std::size_t>(1, std::min<std::size_t>(n, 4 * static_cast<std::size_t>(T)));
    const auto bound = [n, C](std::size_t c) { return c * (n / C) + std::min(c, n % C); };
    std::vector<double> inv(n);           // 1/norm, 0 where the rep cancels
    std::vector<std::size_t> at(C + 1, 0);
#ifdef _OPENMP
#   pragma omp parallel for schedule(static) num_threads(T)
#endif
    for (long long c = 0; c < static_cast<long long>(C); ++c) {
        std::size_t kept = 0;
        for (std::size_t i = bound(static_cast<std::size_t>(c)); i < bound(static_cast<std::size_t>(c) + 1); ++i) {
            const double nsq = ed::symmetry::projected_norm_sq_stab(tab.stabilizer_of(i), characters);
            // scale-free: squared norm of a projected unit vector
            inv[i] = nsq <= 1e-12 ? 0.0 : 1.0 / std::sqrt(nsq);
            kept += inv[i] != 0.0;
        }
        at[static_cast<std::size_t>(c) + 1] = kept;
    }
    for (std::size_t c = 0; c < C; ++c) at[c + 1] += at[c];
    rd.reps.resize(at[C]);
    rd.inv_norms.resize(at[C]);
    if (local) local->resize(n);
#ifdef _OPENMP
#   pragma omp parallel for schedule(static) num_threads(T)
#endif
    for (long long c = 0; c < static_cast<long long>(C); ++c) {
        std::size_t pos = at[static_cast<std::size_t>(c)];
        for (std::size_t i = bound(static_cast<std::size_t>(c)); i < bound(static_cast<std::size_t>(c) + 1); ++i) {
            if (inv[i] == 0.0) {
                if (local) (*local)[i] = -1;
                continue;
            }
            rd.reps[pos] = tab.reps[i];
            rd.inv_norms[pos] = inv[i];
            // narrow-ok: a rank table (the only user of `local`) exists only for at most INT32_MAX reps
            if (local) (*local)[i] = static_cast<std::int32_t>(pos);
            ++pos;
        }
    }
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

}  // namespace ed::solvers
