// =============================================================================
// src/bfg/ring_observables.cpp
//
// Ring-flip / chirality kernels promoted out of compute_bfg_order_parameters.cpp
// (P2.1 ring-observables slice). Behaviour is byte-for-byte preserved from the
// original CPU driver -- only the surrounding type plumbing changed:
//   * the file-local `BowtieData {s1,s2,s3,s4,center}` POD is gone; we now
//     consume `ed::bfg::Bowtie` directly and ignore the unused `s0` /
//     `orientation` fields,
//   * the file-local memory-efficient flag is now read through the
//     `memory_efficient_mode_enabled()` accessor exported by
//     `ed/bfg/structure_factor.h` (the same global, just not extern-linked).
//
// Bit convention: bit=0 -> spin UP, bit=1 -> spin DOWN. A `S^+` therefore
// requires the corresponding bit to be 1 (DOWN) and flips it to 0 (UP); a
// `S^-` requires bit=0 (UP) and flips it to 1.
// =============================================================================

#include "ed/bfg/ring_observables.h"
#include "ed/bfg/structure_factor.h"  // memory_efficient_mode_enabled()

#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::bfg {

namespace {

// Local copies of the bit helpers used throughout the BFG kernels. We keep
// them anonymous-namespaced so each TU gets its own static inline -- there is
// no canonical public bit-twiddling header in `ed_core` yet, and consolidating
// across structure_factor.cpp / correlations.cpp / ring_observables.cpp would
// be its own slice.
inline int get_bit(uint64_t state, int site) {
    return static_cast<int>((state >> site) & 1ULL);
}

inline uint64_t flip_bit(uint64_t state, int site) {
    return state ^ (1ULL << site);
}

constexpr Complex I{0.0, 1.0};

}  // namespace

// -----------------------------------------------------------------------------
// apply_bowtie_fourier
//
// Builds |φ⟩ = Σ_bt e^{i q · r_bt} (S^+_1 S^-_2 S^+_3 S^-_4 + h.c.) |ψ⟩.
// Two implementations live here: a thread-safe atomic version for the
// memory-efficient path, and a thread-local accumulation version for the
// default fast path. Both are direct lifts of the original CPU driver.
// -----------------------------------------------------------------------------
std::vector<Complex> apply_bowtie_fourier(
    const std::vector<Bowtie>& bowties,
    const std::vector<Complex>& psi,
    const std::array<double, 2>& q
) {
    const uint64_t n_states = psi.size();
    std::vector<Complex> result(n_states, Complex(0.0, 0.0));
    const int n_bowties = static_cast<int>(bowties.size());

    if (n_bowties == 0) {
        return result;
    }

    std::vector<Complex> phases(n_bowties);
    for (int p = 0; p < n_bowties; ++p) {
        const double phase_arg =
            q[0] * bowties[p].center[0] + q[1] * bowties[p].center[1];
        phases[p] = std::exp(I * phase_arg);
    }

    if (memory_efficient_mode_enabled()) {
        // Memory-efficient path: atomic scatter into shared real/imag buffers.
        std::vector<double> result_real(n_states, 0.0);
        std::vector<double> result_imag(n_states, 0.0);

        #pragma omp parallel for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            const Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            for (int p = 0; p < n_bowties; ++p) {
                const int s1 = bowties[p].s1;
                const int s2 = bowties[p].s2;
                const int s3 = bowties[p].s3;
                const int s4 = bowties[p].s4;

                const int b1 = get_bit(state, s1);
                const int b2 = get_bit(state, s2);
                const int b3 = get_bit(state, s3);
                const int b4 = get_bit(state, s4);

                const Complex phase = phases[p];

                // S+_1 S-_2 S+_3 S-_4: needs DOWN, UP, DOWN, UP.
                if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
                    const uint64_t new_state =
                        flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                    const Complex val = phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }

                // S-_1 S+_2 S-_3 S+_4: needs UP, DOWN, UP, DOWN.
                if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
                    const uint64_t new_state =
                        flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                    const Complex val = phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (uint64_t s = 0; s < n_states; ++s) {
            result[s] = Complex(result_real[s], result_imag[s]);
        }
    } else {
        // Fast path: per-thread buffer, single critical-section reduction.
        #pragma omp parallel
        {
            std::vector<Complex> local_result(n_states, Complex(0.0, 0.0));

            #pragma omp for schedule(dynamic, 1024)
            for (uint64_t state = 0; state < n_states; ++state) {
                const Complex coeff = psi[state];
                if (std::abs(coeff) < 1e-15) continue;

                for (int p = 0; p < n_bowties; ++p) {
                    const int s1 = bowties[p].s1;
                    const int s2 = bowties[p].s2;
                    const int s3 = bowties[p].s3;
                    const int s4 = bowties[p].s4;

                    const int b1 = get_bit(state, s1);
                    const int b2 = get_bit(state, s2);
                    const int b3 = get_bit(state, s3);
                    const int b4 = get_bit(state, s4);

                    const Complex phase = phases[p];

                    if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
                        const uint64_t new_state = flip_bit(
                            flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                        local_result[new_state] += phase * coeff;
                    }
                    if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
                        const uint64_t new_state = flip_bit(
                            flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                        local_result[new_state] += phase * coeff;
                    }
                }
            }

            #pragma omp critical
            {
                for (uint64_t s = 0; s < n_states; ++s) {
                    result[s] += local_result[s];
                }
            }
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// compute_bowtie_resonance
// -----------------------------------------------------------------------------
Complex compute_bowtie_resonance(
    const std::vector<Complex>& psi,
    int s1, int s2, int s3, int s4
) {
    const uint64_t n_states = psi.size();
    double result_real = 0.0;
    double result_imag = 0.0;

    #pragma omp parallel reduction(+:result_real,result_imag)
    {
        double local_real = 0.0;
        double local_imag = 0.0;

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            const Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            const int b1 = get_bit(state, s1);
            const int b2 = get_bit(state, s2);
            const int b3 = get_bit(state, s3);
            const int b4 = get_bit(state, s4);

            // S+_1 S-_2 S+_3 S-_4 term: needs (DOWN, UP, DOWN, UP).
            if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
                const uint64_t new_state =
                    flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            // h.c.: S-_1 S+_2 S-_3 S+_4: needs (UP, DOWN, UP, DOWN).
            if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
                const uint64_t new_state =
                    flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
        }

        result_real += local_real;
        result_imag += local_imag;
    }

    return Complex(result_real, result_imag);
}

// -----------------------------------------------------------------------------
// compute_triangle_chiral
//
// Symmetric three-spin ring exchange S+_1 S-_2 S+_3 + h.c. -- not the
// antisymmetric scalar chirality S_1 . (S_2 x S_3). Same kernel as the CPU
// driver historically used for the BFG triangle order.
// -----------------------------------------------------------------------------
Complex compute_triangle_chiral(
    const std::vector<Complex>& psi,
    int s1, int s2, int s3
) {
    const uint64_t n_states = psi.size();
    double result_real = 0.0;
    double result_imag = 0.0;

    #pragma omp parallel reduction(+:result_real,result_imag)
    {
        double local_real = 0.0;
        double local_imag = 0.0;

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            const Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            const int b1 = get_bit(state, s1);
            const int b2 = get_bit(state, s2);
            const int b3 = get_bit(state, s3);

            // S+_1 S-_2 S+_3: needs (DOWN, UP, DOWN).
            if (b1 == 1 && b2 == 0 && b3 == 1) {
                const uint64_t new_state =
                    flip_bit(flip_bit(flip_bit(state, s1), s2), s3);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            // S-_1 S+_2 S-_3: needs (UP, DOWN, UP).
            if (b1 == 0 && b2 == 1 && b3 == 0) {
                const uint64_t new_state =
                    flip_bit(flip_bit(flip_bit(state, s1), s2), s3);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
        }

        result_real += local_real;
        result_imag += local_imag;
    }

    return Complex(result_real, result_imag);
}

}  // namespace ed::bfg
