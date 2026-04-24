// =============================================================================
// src/bfg/structure_factor.cpp
//
// Implementation of the bond-bilinear structure-factor / Fourier-applied
// dimer kernels (P2.1 structure-factor slice). Lifted verbatim from
// `src/apps/compute_bfg_order_parameters.cpp` and consolidated inside
// `ed_bfg` so the CPU driver, the GPU driver, and the future Python
// bindings call the same kernels.
//
// The bit-layout helpers are duplicated in an anonymous namespace because
// they are too small to justify a public spin_bits header and matching the
// same convention as `correlations.cpp` here keeps the diff easy to
// review.
// =============================================================================

#include "ed/bfg/structure_factor.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::bfg {

namespace {

constexpr Complex I_unit{0.0, 1.0};

// File-private memory-efficient mode flag. The previous implementation
// kept these as `static` file-locals in the CPU driver; here they become
// translation-unit locals inside `ed_bfg` so the only public surface is
// `set_memory_efficient_mode` / `memory_efficient_mode_enabled`.
bool g_use_memory_efficient_mode = false;

inline int get_bit(uint64_t state, int site) {
    return static_cast<int>((state >> site) & 1ULL);
}

inline uint64_t flip_bit(uint64_t state, int site) {
    return state ^ (1ULL << site);
}

}  // namespace

void set_memory_efficient_mode(uint64_t n_states) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    const uint64_t mem_per_thread  = n_states * sizeof(Complex);
    const uint64_t total_thread_mem = static_cast<uint64_t>(n_threads) * mem_per_thread;

    g_use_memory_efficient_mode =
        (total_thread_mem > 4ULL * 1024ULL * 1024ULL * 1024ULL);

    if (g_use_memory_efficient_mode) {
        std::cout << "[Memory] Enabling memory-efficient mode for " << n_states
                  << " states (" << n_threads << " threads would need "
                  << (total_thread_mem / (1024ULL * 1024ULL * 1024ULL))
                  << " GB)" << std::endl;
    }
}

bool memory_efficient_mode_enabled() {
    return g_use_memory_efficient_mode;
}

DimerSFResult compute_dimer_sf_direct(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    const uint64_t n_states = psi.size();
    const int n_bonds = static_cast<int>(bonds.size());

    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        const double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I_unit * phase_arg);
    }

    std::vector<double> result_real(n_states, 0.0);
    std::vector<double> result_imag(n_states, 0.0);

    double expect_real = 0.0;
    double expect_imag = 0.0;

    #pragma omp parallel for schedule(dynamic, 1024) reduction(+:expect_real, expect_imag)
    for (uint64_t state = 0; state < n_states; ++state) {
        const Complex coeff = psi[state];
        if (std::abs(coeff) < 1e-15) continue;

        for (int b = 0; b < n_bonds; ++b) {
            const int i = bonds[b].first;
            const int j = bonds[b].second;
            const int s_i = get_bit(state, i);
            const int s_j = get_bit(state, j);
            const Complex phase = phases[b];

            // S^+ S^-: i=DOWN(1), j=UP(0)
            if (s_i == 1 && s_j == 0) {
                const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                const Complex val = phase * coeff;
                #pragma omp atomic
                result_real[new_state] += val.real();
                #pragma omp atomic
                result_imag[new_state] += val.imag();

                const Complex exp_contrib = std::conj(psi[new_state]) * coeff * phase;
                expect_real += exp_contrib.real();
                expect_imag += exp_contrib.imag();
            }

            // S^- S^+: i=UP(0), j=DOWN(1)
            if (s_i == 0 && s_j == 1) {
                const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                const Complex val = phase * coeff;
                #pragma omp atomic
                result_real[new_state] += val.real();
                #pragma omp atomic
                result_imag[new_state] += val.imag();

                const Complex exp_contrib = std::conj(psi[new_state]) * coeff * phase;
                expect_real += exp_contrib.real();
                expect_imag += exp_contrib.imag();
            }
        }
    }

    double overlap_val = 0.0;
    #pragma omp parallel for reduction(+:overlap_val) schedule(static)
    for (uint64_t s = 0; s < n_states; ++s) {
        overlap_val += result_real[s] * result_real[s] + result_imag[s] * result_imag[s];
    }

    const Complex expect(expect_real, expect_imag);
    return {Complex(overlap_val, 0.0), expect, expect};
}

DimerSFResult compute_heisenberg_sf_direct(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    const uint64_t n_states = psi.size();
    const int n_bonds = static_cast<int>(bonds.size());

    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        const double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I_unit * phase_arg);
    }

    std::vector<double> result_real(n_states, 0.0);
    std::vector<double> result_imag(n_states, 0.0);

    double expect_real = 0.0;
    double expect_imag = 0.0;

    #pragma omp parallel for schedule(dynamic, 1024) reduction(+:expect_real, expect_imag)
    for (uint64_t state = 0; state < n_states; ++state) {
        const Complex coeff = psi[state];
        if (std::abs(coeff) < 1e-15) continue;

        for (int b = 0; b < n_bonds; ++b) {
            const int i = bonds[b].first;
            const int j = bonds[b].second;
            const int s_i = get_bit(state, i);
            const int s_j = get_bit(state, j);
            const Complex phase = phases[b];

            // SzSz part (diagonal): contributes to result[state]
            const double sz_i = s_i ? -0.5 : 0.5;
            const double sz_j = s_j ? -0.5 : 0.5;
            const Complex szsz_contrib = phase * coeff * sz_i * sz_j;
            #pragma omp atomic
            result_real[state] += szsz_contrib.real();
            #pragma omp atomic
            result_imag[state] += szsz_contrib.imag();

            const Complex exp_szsz = std::conj(psi[state]) * coeff * phase * sz_i * sz_j;
            expect_real += exp_szsz.real();
            expect_imag += exp_szsz.imag();

            // (1/2) S+S-: i=DOWN(1), j=UP(0)
            if (s_i == 1 && s_j == 0) {
                const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                const Complex val = phase * coeff * 0.5;
                #pragma omp atomic
                result_real[new_state] += val.real();
                #pragma omp atomic
                result_imag[new_state] += val.imag();

                const Complex exp_contrib = std::conj(psi[new_state]) * coeff * phase * 0.5;
                expect_real += exp_contrib.real();
                expect_imag += exp_contrib.imag();
            }

            // (1/2) S-S+: i=UP(0), j=DOWN(1)
            if (s_i == 0 && s_j == 1) {
                const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                const Complex val = phase * coeff * 0.5;
                #pragma omp atomic
                result_real[new_state] += val.real();
                #pragma omp atomic
                result_imag[new_state] += val.imag();

                const Complex exp_contrib = std::conj(psi[new_state]) * coeff * phase * 0.5;
                expect_real += exp_contrib.real();
                expect_imag += exp_contrib.imag();
            }
        }
    }

    double overlap_val = 0.0;
    #pragma omp parallel for reduction(+:overlap_val) schedule(static)
    for (uint64_t s = 0; s < n_states; ++s) {
        overlap_val += result_real[s] * result_real[s] + result_imag[s] * result_imag[s];
    }

    const Complex expect(expect_real, expect_imag);
    return {Complex(overlap_val, 0.0), expect, expect};
}

std::vector<Complex> apply_dimer_fourier(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    const uint64_t n_states = psi.size();
    std::vector<Complex> result(n_states, 0.0);
    const int n_bonds = static_cast<int>(bonds.size());

    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        const double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I_unit * phase_arg);
    }

    if (g_use_memory_efficient_mode) {
        std::vector<double> result_real(n_states, 0.0);
        std::vector<double> result_imag(n_states, 0.0);

        #pragma omp parallel for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            const Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            for (int b = 0; b < n_bonds; ++b) {
                const int i = bonds[b].first;
                const int j = bonds[b].second;
                const int s_i = get_bit(state, i);
                const int s_j = get_bit(state, j);
                const Complex phase = phases[b];

                if (s_i == 1 && s_j == 0) {
                    const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    const Complex val = phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
                if (s_i == 0 && s_j == 1) {
                    const uint64_t new_state = flip_bit(flip_bit(state, i), j);
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
        #pragma omp parallel
        {
            std::vector<Complex> local_result(n_states, 0.0);

            #pragma omp for schedule(dynamic, 1024)
            for (uint64_t state = 0; state < n_states; ++state) {
                const Complex coeff = psi[state];
                if (std::abs(coeff) < 1e-15) continue;

                for (int b = 0; b < n_bonds; ++b) {
                    const int i = bonds[b].first;
                    const int j = bonds[b].second;
                    const int s_i = get_bit(state, i);
                    const int s_j = get_bit(state, j);
                    const Complex phase = phases[b];

                    if (s_i == 1 && s_j == 0) {
                        const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_result[new_state] += phase * coeff;
                    }
                    if (s_i == 0 && s_j == 1) {
                        const uint64_t new_state = flip_bit(flip_bit(state, i), j);
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

std::pair<std::vector<Complex>, Complex> apply_heisenberg_dimer_fourier(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    const uint64_t n_states = psi.size();
    std::vector<Complex> result(n_states, 0.0);
    const int n_bonds = static_cast<int>(bonds.size());

    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        const double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I_unit * phase_arg);
    }

    if (g_use_memory_efficient_mode) {
        std::vector<double> result_real(n_states, 0.0);
        std::vector<double> result_imag(n_states, 0.0);

        #pragma omp parallel for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            const Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            for (int b = 0; b < n_bonds; ++b) {
                const int i = bonds[b].first;
                const int j = bonds[b].second;
                const int s_i = get_bit(state, i);
                const int s_j = get_bit(state, j);
                const Complex phase = phases[b];

                const double sz_i = s_i ? -0.5 : 0.5;
                const double sz_j = s_j ? -0.5 : 0.5;
                const double szsz = sz_i * sz_j;

                const Complex diag_contrib = phase * szsz * coeff;
                #pragma omp atomic
                result_real[state] += diag_contrib.real();
                #pragma omp atomic
                result_imag[state] += diag_contrib.imag();

                if (s_i == 1 && s_j == 0) {
                    const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    const Complex val = 0.5 * phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
                if (s_i == 0 && s_j == 1) {
                    const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    const Complex val = 0.5 * phase * coeff;
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
        #pragma omp parallel
        {
            std::vector<Complex> local_result(n_states, 0.0);

            #pragma omp for schedule(dynamic, 1024)
            for (uint64_t state = 0; state < n_states; ++state) {
                const Complex coeff = psi[state];
                if (std::abs(coeff) < 1e-15) continue;

                for (int b = 0; b < n_bonds; ++b) {
                    const int i = bonds[b].first;
                    const int j = bonds[b].second;
                    const int s_i = get_bit(state, i);
                    const int s_j = get_bit(state, j);
                    const Complex phase = phases[b];

                    const double sz_i = s_i ? -0.5 : 0.5;
                    const double sz_j = s_j ? -0.5 : 0.5;
                    const double szsz = sz_i * sz_j;

                    local_result[state] += phase * szsz * coeff;

                    if (s_i == 1 && s_j == 0) {
                        const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_result[new_state] += 0.5 * phase * coeff;
                    }
                    if (s_i == 0 && s_j == 1) {
                        const uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_result[new_state] += 0.5 * phase * coeff;
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

    // <D(q)> = <psi| D(q) |psi> via the just-built ket. Matches the
    // (overwriting) final loop in the original CPU driver -- the inline
    // accumulation that lived inside the OMP loops above is intentionally
    // dropped here because the original assignment to `expect` happened
    // after both branches and therefore discarded it.
    double expect_real = 0.0;
    double expect_imag = 0.0;
    #pragma omp parallel for reduction(+:expect_real, expect_imag) schedule(static)
    for (uint64_t s = 0; s < n_states; ++s) {
        const Complex contrib = std::conj(psi[s]) * result[s];
        expect_real += contrib.real();
        expect_imag += contrib.imag();
    }

    return {result, Complex(expect_real, expect_imag)};
}

Complex compute_dimer_dimer_correlation(
    const std::vector<Complex>& psi,
    int i1, int j1, int i2, int j2
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

            const int s_i1 = get_bit(state, i1);
            const int s_j1 = get_bit(state, j1);
            const int s_i2 = get_bit(state, i2);
            const int s_j2 = get_bit(state, j2);

            // S+_{i1} S-_{j1} S+_{i2} S-_{j2}
            if (s_j1 == 0 && s_i1 == 1 && s_j2 == 0 && s_i2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, j1);
                new_state = flip_bit(new_state, i1);
                new_state = flip_bit(new_state, j2);
                new_state = flip_bit(new_state, i2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }

            // S+_{i1} S-_{j1} S-_{i2} S+_{j2}
            if (s_j1 == 0 && s_i1 == 1 && s_i2 == 0 && s_j2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, j1);
                new_state = flip_bit(new_state, i1);
                new_state = flip_bit(new_state, i2);
                new_state = flip_bit(new_state, j2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }

            // S-_{i1} S+_{j1} S+_{i2} S-_{j2}
            if (s_i1 == 0 && s_j1 == 1 && s_j2 == 0 && s_i2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, i1);
                new_state = flip_bit(new_state, j1);
                new_state = flip_bit(new_state, j2);
                new_state = flip_bit(new_state, i2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }

            // S-_{i1} S+_{j1} S-_{i2} S+_{j2}
            if (s_i1 == 0 && s_j1 == 1 && s_i2 == 0 && s_j2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, i1);
                new_state = flip_bit(new_state, j1);
                new_state = flip_bit(new_state, i2);
                new_state = flip_bit(new_state, j2);
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

double compute_heisenberg_dimer_dimer_correlation(
    const std::vector<Complex>& psi,
    int i1, int j1, int i2, int j2
) {
    const uint64_t n_states = psi.size();
    double result = 0.0;

    #pragma omp parallel reduction(+:result)
    {
        double local_result = 0.0;

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            const Complex coeff = psi[state];
            const double prob = std::norm(coeff);
            if (prob < 1e-30) continue;

            const int s_i1 = get_bit(state, i1);
            const int s_j1 = get_bit(state, j1);
            const int s_i2 = get_bit(state, i2);
            const int s_j2 = get_bit(state, j2);

            const double sz_i1 = s_i1 ? -0.5 : 0.5;
            const double sz_j1 = s_j1 ? -0.5 : 0.5;
            const double sz_i2 = s_i2 ? -0.5 : 0.5;
            const double sz_j2 = s_j2 ? -0.5 : 0.5;

            const double szsz_1 = sz_i1 * sz_j1;
            const double szsz_2 = sz_i2 * sz_j2;
            local_result += prob * szsz_1 * szsz_2;

            // (SzSz)_1 x (1/2)(S+S- + S-S+)_2:
            if (s_i2 == 1 && s_j2 == 0) {
                const uint64_t new_state = flip_bit(flip_bit(state, i2), j2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_1 * contrib.real();
            }
            if (s_i2 == 0 && s_j2 == 1) {
                const uint64_t new_state = flip_bit(flip_bit(state, i2), j2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_1 * contrib.real();
            }

            // (1/2)(S+S- + S-S+)_1 x (SzSz)_2:
            if (s_i1 == 1 && s_j1 == 0) {
                const uint64_t new_state = flip_bit(flip_bit(state, i1), j1);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_2 * contrib.real();
            }
            if (s_i1 == 0 && s_j1 == 1) {
                const uint64_t new_state = flip_bit(flip_bit(state, i1), j1);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_2 * contrib.real();
            }

            // (1/4)(XY)_1 x (XY)_2:
            if (s_i1 == 1 && s_j1 == 0 && s_i2 == 1 && s_j2 == 0) {
                const uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
            if (s_i1 == 1 && s_j1 == 0 && s_i2 == 0 && s_j2 == 1) {
                const uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
            if (s_i1 == 0 && s_j1 == 1 && s_i2 == 1 && s_j2 == 0) {
                const uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
            if (s_i1 == 0 && s_j1 == 1 && s_i2 == 0 && s_j2 == 1) {
                const uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                const Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
        }

        result += local_result;
    }

    return result;
}

}  // namespace ed::bfg
