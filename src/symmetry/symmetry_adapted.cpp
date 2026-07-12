// =============================================================================
// src/symmetry/symmetry_adapted.cpp  --  see symmetry_adapted.h
// =============================================================================

#include <ed/symmetry/symmetry_adapted.h>

#include <mutex>

#include <ed/core/basis_utils.h>   // applyPermutation
#include <ed/symmetry/gosper.h>    // next_bit_permutation (fixed-popcount enumeration)

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ed::symmetry {

using Complex = std::complex<double>;

std::vector<SABVector>
build_sab_partition0(const GroupIrreps&                   gi,
                     const std::vector<std::vector<int>>& max_clique,
                     int                                  irrep_index,
                     int                                  n_sites,
                     int                                  n_up,
                     int                                  partner,
                     int                                  sz_parity)
{
    const IrrepData& ir = gi.irreps[static_cast<std::size_t>(irrep_index)];
    const int        G  = gi.order;
    const int        d  = ir.dim;

    // ----- Scale guard --------------------------------------------------------
    // The symmetry-adapted-basis engine is a MODERATE-N reference: it ENUMERATES
    // the sector (2^n_sites, or C(n_sites,n_up) for fixed-Sz) and STORES the SAB
    // amplitudes + a state->index map. It does NOT scale like the matrix-free
    // abelian rep walk (RepSymmetryBasisPolicy), which holds only the orbit-rep
    // list and regenerates the projection arithmetically. Refuse to silently
    // OOM / hang on large problems and point the caller at the scalable path.
    {
        long double enum_sz;
        if (n_up < 0 && sz_parity >= 0) {
            enum_sz = std::pow(2.0L, static_cast<long double>(n_sites - 1));
        } else if (n_up < 0) {
            enum_sz = std::ldexp(1.0L, n_sites);                 // 2^n_sites
        } else {
            enum_sz = 1.0L;                                      // C(n_sites,n_up)
            const int kk = std::min(n_up, n_sites - n_up);
            for (int i = 0; i < kk; ++i)
                enum_sz = enum_sz * (n_sites - i) / (i + 1);
        }
        std::uint64_t cap = std::uint64_t{1} << 22;              // ~4.2M default
        if (const char* e = std::getenv("ED_SYM_SAB_MAX_DIM")) {
            char* end = nullptr;
            const unsigned long long v = std::strtoull(e, &end, 10);
            if (end != e && v > 0) cap = static_cast<std::uint64_t>(v);
        }
        if (enum_sz > static_cast<long double>(cap)) {
            throw std::runtime_error(
                "build_sab_partition0: symmetry-adapted-basis enumeration size (~"
                + std::to_string(static_cast<unsigned long long>(enum_sz))
                + ") exceeds the moderate-N cap (" + std::to_string(cap)
                + "). The non-abelian SAB engine stores the reduced basis and does "
                  "not scale to this size. Use the matrix-free ABELIAN rep path "
                  "(qed.solve / qed.thermal / qed.full_spectrum with an abelian "
                  "generator set), reduce n_sites/restrict n_up, or raise "
                  "ED_SYM_SAB_MAX_DIM if the machine can hold the basis. A "
                  "matrix-free NON-abelian engine is not yet implemented.");
        }
    }

    // Partner-`partner` diagonal element D^Γ_{pp}(g) for every group element.
    // Summing eigenstates over ALL partners p = 0..d-1 spans the full isotypic
    // space (needed for complete matrix-element sums, e.g. DSSF); a single
    // partner suffices for eigenvalues (GS / thermodynamics).
    std::vector<Complex> D00(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g)
        D00[static_cast<std::size_t>(g)] =
            ir.matrices[static_cast<std::size_t>(g)]
                       [static_cast<std::size_t>(partner) * d + partner];

    std::vector<SABVector> out;

    // Per-orbit-representative SAB construction (independent of which subspace
    // the rep was enumerated from — orbits of fixed-popcount states stay in the
    // Sz sector because permutations preserve popcount).
    auto process_rep = [&](std::uint64_t s) {
        std::uint64_t rep = s;
        for (int g = 0; g < G; ++g)
            rep = std::min(rep, applyPermutation(s, max_clique[static_cast<std::size_t>(g)]));
        if (rep != s) return;  // only orbit minima

        std::map<std::uint64_t, int> coord;
        std::vector<std::uint64_t>   orbit;
        for (int g = 0; g < G; ++g) {
            const std::uint64_t t = applyPermutation(s, max_clique[static_cast<std::size_t>(g)]);
            if (coord.emplace(t, static_cast<int>(orbit.size())).second) orbit.push_back(t);
        }
        const int M = static_cast<int>(orbit.size());

        // Column k = P^Γ_{00}|orbit[k]> = Σ_g conj(D00(g)) |perm_g(orbit[k])>
        // (constant d_Γ/|G| dropped; SVD column basis is gauge/scale invariant).
        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(M, M);
        for (int k = 0; k < M; ++k)
            for (int g = 0; g < G; ++g) {
                const std::uint64_t t =
                    applyPermutation(orbit[static_cast<std::size_t>(k)],
                                     max_clique[static_cast<std::size_t>(g)]);
                A(coord[t], k) += std::conj(D00[static_cast<std::size_t>(g)]);
            }

        // Orthonormal column space = the m_Γ(O) partner-0 SAB vectors. Genuine
        // singular values are |G|/d_Γ = O(1); an ABSENT irrep gives A≈0 (~1e-14),
        // so an ABSOLUTE threshold contributes zero spurious vectors.
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A, Eigen::ComputeThinU);
        const auto& sv = svd.singularValues();
        const double tol = 1e-8 * static_cast<double>(G);
        for (int i = 0; i < sv.size(); ++i) {
            if (sv(i) <= tol) continue;
            SABVector v;
            const Eigen::VectorXcd u = svd.matrixU().col(i);  // unit-norm
            for (int c = 0; c < M; ++c)
                if (std::abs(u(c)) > 1e-12) {
                    v.states.push_back(orbit[static_cast<std::size_t>(c)]);
                    v.coeffs.push_back(u(c));
                }
            out.push_back(std::move(v));
        }
    };

    if (n_up < 0 && sz_parity >= 0) {
        // Sz-parity half: popcount(s) mod 2 == sz_parity.
        const std::uint64_t full = (1ULL << n_sites);
        for (std::uint64_t s2 = 0; s2 < full; ++s2) {
            if ((static_cast<int>(__builtin_popcountll(s2)) & 1) == sz_parity)
                process_rep(s2);
        }
    } else if (n_up < 0) {
        const std::uint64_t full = std::uint64_t{1} << n_sites;
        for (std::uint64_t s = 0; s < full; ++s) process_rep(s);
    } else if (n_up <= n_sites) {
        // Fixed-Sz: enumerate states of popcount n_up via Gosper's hack.
        if (n_up == 0) {
            process_rep(0);
        } else {
            const std::uint64_t lim =
                (n_sites < 64) ? (std::uint64_t{1} << n_sites)
                               : std::numeric_limits<std::uint64_t>::max();
            std::uint64_t s = (std::uint64_t{1} << n_up) - 1;
            while (s < lim) {
                process_rep(s);
                const std::uint64_t nxt = next_bit_permutation(s);
                if (nxt <= s) break;  // wrap-around / end of sequence
                s = nxt;
            }
        }
    }
    return out;
}

::SymmetrySector build_symmetry_adapted_sector(
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  irrep_index,
    int                                  n_sites,
    int                                  n_up,
    int                                  partner,
    int                                  sz_parity)
{
    // Structural consolidation (Jul 2026): the SAB basis is a pure
    // function of (group, irrep, subspace) -- NOT of the operator's
    // terms -- so cache it content-keyed like the abelian orbit
    // tables. Kills the per-call reconstruction that dominated the
    // non-abelian lane's fixed costs (solve/thermal/DSSF on the same
    // model re-derived identical bases every call).
    // Reuse must NEVER depend on hash quality (the GPU-mirror memo's
    // word-XOR FNV collided on structured inputs -- same recipe here:
    // avalanche every mixed word AND verify a FULL identity fingerprint
    // on every hit; a silent wrong-basis hit is wrong physics).
    struct SabCacheEntry {
        std::uint64_t                 key;
        std::vector<std::vector<int>> clique;
        int  irrep_index, n_sites, n_up, partner, sz_parity;
        ::SymmetrySector              sector;
    };
    static std::mutex sab_cache_mtx;
    static std::vector<SabCacheEntry> sab_cache;  // small FIFO
    static std::size_t sab_cache_bytes = 0;       // C11: running footprint
    std::uint64_t key = 0xcbf29ce484222325ULL;
    auto mix = [&key](std::uint64_t v) {
        v += 0x9E3779B97F4A7C15ULL;                       // splitmix64 avalanche
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
        v ^= v >> 31;
        key ^= v; key *= 0x100000001b3ULL;
    };
    for (const auto& p : max_clique)
        for (int site : p) mix(static_cast<std::uint64_t>(site) + 0x9E37ULL);
    mix(static_cast<std::uint64_t>(irrep_index) + 1);
    mix(static_cast<std::uint64_t>(n_sites) + 0x51ULL);
    mix(static_cast<std::uint64_t>(n_up + 2));
    mix(static_cast<std::uint64_t>(partner + 2));
    mix(static_cast<std::uint64_t>(sz_parity + 2));
    {
        std::lock_guard<std::mutex> lk(sab_cache_mtx);
        for (const auto& e : sab_cache)
            if (e.key == key && e.irrep_index == irrep_index
                && e.n_sites == n_sites && e.n_up == n_up
                && e.partner == partner && e.sz_parity == sz_parity
                && e.clique == max_clique)
                return e.sector;
    }

    ::SymmetrySector sec;
    sec.sector_id        = static_cast<std::uint64_t>(irrep_index);
    sec.quantum_numbers  = {irrep_index};
    const auto sab = build_sab_partition0(gi, max_clique, irrep_index, n_sites, n_up, partner, sz_parity);
    sec.basis_states.reserve(sab.size());
    for (const auto& v : sab) {
        ::SymBasisState bs;
        bs.orbit_elements     = v.states;
        bs.orbit_coefficients = v.coeffs;
        bs.quantum_numbers    = {irrep_index};
        // SAB vectors are orthonormal: norm = 1, and the matvec uses group_norm = 1
        // (no 1/|G|), so SymmetryBasisPolicy's coeff_modifier reduces to conj(c_{s'}).
        bs.norm     = 1.0;
        bs.inv_norm = 1.0;
        bs.orbit_rep = v.states.empty()
            ? 0ULL
            : *std::min_element(v.states.begin(), v.states.end());
        bs.sortOrbit();   // sort orbit_elements (+ parallel coeffs) for findCoeff; refreshes inv_norm
        sec.basis_states.push_back(std::move(bs));
    }
    // C11: FIFO with BOTH a count cap and a byte budget. Full-space SAB
    // enumerations can make a single sector many MB (orbit elements +
    // coefficients per basis vector), so a pure 64-entry FIFO could pin GBs.
    // Evict oldest until under both limits. Default 512 MiB
    // (ED_SYM_SAB_CACHE_BYTES overrides); a sector larger than the budget is
    // returned but not cached.
    auto sector_bytes = [](const ::SymmetrySector& s) -> std::size_t {
        std::size_t b = 0;
        for (const auto& bs : s.basis_states)
            b += bs.orbit_elements.size() * sizeof(std::uint64_t)
               + bs.orbit_coefficients.size() * sizeof(std::complex<double>);
        return b + sizeof(::SymmetrySector);
    };
    {
        std::lock_guard<std::mutex> lk(sab_cache_mtx);
        constexpr std::size_t kCap = 64;          // ~one model's irreps
        std::size_t budget = std::size_t{512} * 1024 * 1024;
        if (const char* e = std::getenv("ED_SYM_SAB_CACHE_BYTES")) {
            char* end = nullptr;
            const unsigned long long v = std::strtoull(e, &end, 10);
            if (end != e && v > 0) budget = static_cast<std::size_t>(v);
        }
        const std::size_t this_bytes = sector_bytes(sec);
        // Evict oldest until this entry fits under both the count cap and the
        // byte budget (never evict below one slot; a too-big sector is served
        // uncached).
        while (!sab_cache.empty()
               && (sab_cache.size() >= kCap
                   || sab_cache_bytes + this_bytes > budget)) {
            sab_cache_bytes -= sector_bytes(sab_cache.front().sector);
            sab_cache.erase(sab_cache.begin());
        }
        if (this_bytes <= budget) {
            sab_cache.push_back(SabCacheEntry{
                key, max_clique, irrep_index, n_sites, n_up, partner,
                sz_parity, sec});
            sab_cache_bytes += this_bytes;
        }
    }
    return sec;
}


// Exact canonical thermodynamics from a full eigenvalue list (multiplicities
// already folded in). Z(β)=Σ e^{-βE}; reference-shifted by E0 for stability.
// Public so the GPU consumer reuses the identical reduction.
ThermodynamicData
canonical_thermo_from_eigs(const std::vector<double>& eigs,
                           const std::vector<double>& T) {
    ThermodynamicData td;
    td.temperatures = T;
    if (eigs.empty()) return td;
    const double E0 = *std::min_element(eigs.begin(), eigs.end());
    td.e_min = E0;
    for (double t : T) {
        const double beta = 1.0 / t;
        double Z = 0.0, E = 0.0, E2 = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-beta * (e - E0));
            Z += w; E += w * e; E2 += w * e * e;
        }
        const double Eavg  = E / Z;
        const double E2avg = E2 / Z;
        const double Cv    = beta * beta * (E2avg - Eavg * Eavg);
        const double F     = E0 - t * std::log(Z);   // -T ln Z_full
        const double S     = (Eavg - F) / t;
        td.energy.push_back(Eavg);
        td.specific_heat.push_back(Cv);
        td.free_energy.push_back(F);
        td.entropy.push_back(S);
    }
    return td;
}


}  // namespace ed::symmetry
