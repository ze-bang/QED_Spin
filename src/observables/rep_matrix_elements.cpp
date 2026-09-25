// =============================================================================
// src/observables/rep_matrix_elements.cpp -- CPU sweep behind rep_matrix_elements().
// See include/ed/observables/rep_matrix_elements.h for the formula and conventions.
// =============================================================================
#include <ed/observables/rep_matrix_elements.h>

#include <ed/symmetry/rep_sector_data.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::observables {

using Complex = std::complex<double>;

std::vector<Complex>
rep_matrix_elements(const ed::symmetry::RepSectorData& src,
                    const ed::symmetry::RepSectorData& tgt,
                    const MaskedProgram& prog,
                    const std::vector<RepVectorView>& kets,
                    const std::vector<RepVectorView>& bras,
                    const std::vector<std::pair<int, int>>& pairs,
                    const RepMEOptions& opt) {
    if (prog.src_characters != src.characters || prog.tgt_characters != tgt.characters
        || prog.src_n_up != src.n_up || prog.tgt_n_up != tgt.n_up)
        throw std::invalid_argument("rep_matrix_elements: program was compiled for other sectors");
    if (src.group_size != tgt.group_size || src.n_sites != tgt.n_sites)
        throw std::invalid_argument("rep_matrix_elements: the two sectors come from different groups");
    for (const auto& v : kets)
        if (v.size != src.dim() || (v.size > 0 && v.data == nullptr))
            throw std::invalid_argument("rep_matrix_elements: ket length != source sector dim");
    for (const auto& v : bras)
        if (v.size != tgt.dim() || (v.size > 0 && v.data == nullptr))
            throw std::invalid_argument("rep_matrix_elements: bra length != target sector dim");
    for (const auto& [b, k] : pairs)
        if (b < 0 || k < 0 || b >= static_cast<int>(bras.size()) || k >= static_cast<int>(kets.size()))
            throw std::invalid_argument("rep_matrix_elements: pair (" + std::to_string(b) + ", "
                                        + std::to_string(k) + ") out of range");

    const std::size_t n_obs = static_cast<std::size_t>(prog.n_obs);
    const std::size_t n_pairs = pairs.size();
    std::vector<Complex> out(n_pairs * n_obs, Complex(0.0, 0.0));
    if (n_pairs == 0 || n_obs == 0 || prog.n_terms() == 0) return out;
    if (opt.use_gpu) return rep_matrix_elements_gpu(src, tgt, prog, kets, bras, pairs, opt);

    const auto tpol = tgt.make_policy();
    const bool same_sector = (&src == &tgt);
    const long long dim = static_cast<long long>(src.dim());
    const std::size_t n_groups = prog.n_groups();

    std::vector<const Complex*> ket_of(n_pairs), bra_of(n_pairs);
    for (std::size_t p = 0; p < n_pairs; ++p) {
        bra_of[p] = bras[static_cast<std::size_t>(pairs[p].first)].data;
        ket_of[p] = kets[static_cast<std::size_t>(pairs[p].second)].data;
    }

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<Complex>> partial(static_cast<std::size_t>(n_threads));

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        std::vector<Complex> acc(n_pairs * n_obs, Complex(0.0, 0.0));
        std::vector<Complex> bk(n_pairs), y(n_pairs);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long long ri = 0; ri < dim; ++ri) {
            const std::size_t r = static_cast<std::size_t>(ri);
            bool any = false;
            for (std::size_t p = 0; p < n_pairs; ++p) {
                bk[p] = ket_of[p][r];
                any = any || (bk[p] != Complex(0.0, 0.0));
            }
            if (!any) continue;
            const std::uint64_t s = src.reps[r];
            const double w = src.inv_norms[r];
            for (std::size_t gi = 0; gi < n_groups; ++gi) {
                const std::uint64_t F = prog.group_flip[gi];
                const std::uint64_t v = s & F;
                if (masked_popcount(v) != prog.group_setbits[gi]) continue;
                const auto vb = prog.vsub_val.begin() + prog.group_vbegin[gi];
                const auto ve = prog.vsub_val.begin() + prog.group_vbegin[gi + 1];
                const auto it = std::lower_bound(vb, ve, v);
                if (it == ve || *it != v) continue;
                const std::size_t vi = static_cast<std::size_t>(it - prog.vsub_val.begin());

                Complex proj;
                std::int64_t j;
                if (same_sector && F == 0) {        // diagonal term inside one sector
                    j = static_cast<std::int64_t>(r);
                    proj = Complex(1.0 / w, 0.0);
                } else {
                    j = tpol.index_and_projection(s ^ F, proj);
                    if (j < 0) continue;            // target orbit cancelled in this sector
                }
                const Complex base = proj * w;
                const std::size_t jj = static_cast<std::size_t>(j);
                for (std::size_t p = 0; p < n_pairs; ++p)
                    y[p] = base * bk[p] * std::conj(bra_of[p][jj]);
                for (std::uint32_t k = prog.vsub_tbegin[vi]; k < prog.vsub_tbegin[vi + 1]; ++k) {
                    Complex c = prog.term_coeff[k];
                    if (masked_popcount(s & prog.term_sign[k]) & 1) c = -c;
                    const std::size_t o = prog.term_obs[k];
                    for (std::size_t p = 0; p < n_pairs; ++p) acc[p * n_obs + o] += c * y[p];
                }
            }
        }
        partial[static_cast<std::size_t>(tid)] = std::move(acc);
    }
    for (const auto& a : partial) {
        if (a.empty()) continue;
        for (std::size_t i = 0; i < out.size(); ++i) out[i] += a[i];
    }
    return out;
}

}  // namespace ed::observables

#ifndef WITH_CUDA
namespace ed::observables {
std::vector<Complex>
rep_matrix_elements_gpu(const ed::symmetry::RepSectorData&, const ed::symmetry::RepSectorData&,
                        const MaskedProgram&, const std::vector<RepVectorView>&,
                        const std::vector<RepVectorView>&, const std::vector<std::pair<int, int>>&,
                        const RepMEOptions&) {
    throw std::runtime_error("rep_matrix_elements: use_gpu requires a WITH_CUDA build");
}
bool rep_matrix_elements_gpu_available() { return false; }
}  // namespace ed::observables
#endif
