// ltlm.cpp - Low Temperature Lanczos Method implementation
#include <ed/core/system_utils.h>

#include <ed/solvers/ltlm.h>
#include <ed/solvers/ftlm.h>     // For build_lanczos_tridiagonal function
#include <ed/solvers/lanczos.h>  // For helper functions
#include <ed/parallel/thread_budget.h>  // Phase 6.1: dim-aware OMP+BLAS cap
#include <filesystem>  // P0.12
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cstring>
#include <cstdlib>

namespace {

inline bool ltlm_static_verbose() {
    const char* env = std::getenv("ED_DSSF_VERBOSE");
    return env != nullptr && std::string(env) != "0";
}

std::vector<double> make_log_temperature_grid(
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins)
{
    if (!(temp_min > 0.0) || !(temp_max > 0.0)) {
        throw std::invalid_argument(
            "LTLM static response requires temp_min and temp_max to be > 0.");
    }

    std::vector<double> temperatures(num_temp_bins);
    const double log_tmin = std::log(temp_min);
    const double log_tmax = std::log(temp_max);
    const double log_step =
        (log_tmax - log_tmin) / std::max(uint64_t(1), num_temp_bins - 1);

    for (uint64_t i = 0; i < num_temp_bins; ++i) {
        temperatures[i] = std::exp(log_tmin + static_cast<double>(i) * log_step);
    }
    return temperatures;
}

} // namespace

/**
 * @brief Find ground state using Lanczos iteration
 */
double find_ground_state_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    uint64_t krylov_dim,
    double tolerance,
    bool full_reorth,
    uint64_t reorth_freq,
    ComplexVector& ground_state
) {
    std::cout << "  Finding ground state via Lanczos...\n";
    
    // Generate random initial vector using helper function
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v0 = generateRandomVector(N, gen, dist);
    
    // Build Lanczos tridiagonal with basis storage
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;
    uint64_t iterations = build_lanczos_tridiagonal_with_basis(
        H, v0, N, krylov_dim, tolerance,
        full_reorth, reorth_freq,
        alpha, beta, &lanczos_vectors
    );
    
    std::cout << "  Lanczos iterations for ground state: " << iterations << std::endl;
    
    uint64_t m = alpha.size();
    
    // Diagonalize tridiagonal matrix using helper function
    std::vector<double> ritz_values, weights;
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
    
    if (ritz_values.empty()) {
        std::cerr << "  Error: Ground state tridiagonal diagonalization failed" << std::endl;
        ground_state = v0;  // Return initial state as fallback
        return 0.0;
    }
    
    double ground_energy = ritz_values[0];
    std::cout << "  Ground state energy: " << ground_energy << std::endl;
    
    // Reconstruct ground state in full Hilbert space
    // |ψ_0⟩ = Σ_j c_j |v_j⟩ where c_j = evecs[j] (first eigenvector)
    ground_state.resize(N, Complex(0.0, 0.0));
    
    for (int j = 0; j < m; j++) {
        double coeff = evecs[j];  // First eigenvector (ground state)
        Complex alpha_c(coeff, 0.0);
        cblas_zaxpy(N, &alpha_c, lanczos_vectors[j].data(), 1, ground_state.data(), 1);
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, ground_state.data(), 1);
    Complex scale(1.0/norm, 0.0);
    cblas_zscal(N, &scale, ground_state.data(), 1);
    
    return ground_energy;
}

/**
 * @brief Build Krylov subspace from ground state for low-lying excitations
 */
[[maybe_unused]] static int build_excitation_spectrum(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& ground_state,
    double ground_energy,
    uint64_t N,
    uint64_t krylov_dim,
    double tolerance,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& excitation_energies,
    std::vector<double>& weights
) {
    std::cout << "  Building excitation spectrum from ground state...\n";
    
    // Build Lanczos tridiagonal starting from ground state
    std::vector<double> alpha, beta;
    uint64_t iterations = build_lanczos_tridiagonal(
        H, ground_state, N, krylov_dim, tolerance,
        full_reorth, reorth_freq,
        alpha, beta
    );
    
    std::cout << "  Lanczos iterations for excitations: " << iterations << std::endl;
    
    uint64_t m = alpha.size();
    
    // Diagonalize tridiagonal matrix using helper function
    diagonalize_tridiagonal_ritz(alpha, beta, excitation_energies, weights);
    
    if (excitation_energies.empty()) {
        std::cerr << "  Warning: Excitation spectrum diagonalization failed" << std::endl;
        return 0;
    }
    
    std::cout << "  Found " << m << " excitation states\n";
    std::cout << "  Energy range: [" << excitation_energies[0] << ", " 
              << excitation_energies[m-1] << "]\n";
    
    return m;
}

// ---------------------------------------------------------------------------
// Correct LTLM thermodynamics (Jul 2026 rewrite).
//
// The previous body seeded ONE Krylov space from the ground state and summed
// Z = sum_n |<0|psi_n>|^2 e^{-beta E_n}: that is the local density of states
// seen from |0>, i.e. <0|H e^{-bH}|0> / <0|e^{-bH}|0>, NOT the thermal trace
// Tr(H e^{-bH})/Tr(e^{-bH}). It stays pinned near E_0 at every T (the GS
// overlap |<0|psi_0>|^2 ~ 1 dominates) -- E(0.2) came out -8.55 vs the exact
// -8.26, a 205% error in the excitation content.
//
// Correct LTLM (Aichhorn et al., PRB 67 161103): separate the exact ground
// state and sample the (D-1)-dim complement with random vectors. For a
// function of H alone (all thermodynamics here) the LTLM symmetric estimator
// reduces to the FTLM one, so this is provably the correct thermal trace:
//
//   Ztilde = Tr e^{-b(H-E0)}
//          = 1 (GS) + ((D-1)/R) sum_r sum_j w_j^r e^{-b(eps_j^r - E0)}
//   U1 = Tr(H e^{-b(H-E0)}) = E0 + ((D-1)/R) sum_r sum_j w_j eps_j e^{...}
//   U2 = Tr(H^2 e^{-b(H-E0)})= E0^2 + ((D-1)/R) sum_r sum_j w_j eps_j^2 e^{...}
//   <E> = U1/Ztilde,  Cv = b^2 (U2/Ztilde - <E>^2),
//   F = E0 - T ln Ztilde,  S = b(<E> - E0) + ln Ztilde.
//
// w_j^r = |<r_perp|psi_j>|^2 with |r_perp> the random vector projected off
// the GS and renormalised (so sum_j w_j ~ 1 and the complement carries the
// (D-1) trace weight). Keeping the GS exact makes the T->0 limit exact with
// few samples -- the reason LTLM beats FTLM at low T.
static ThermodynamicData compute_ltlm_thermodynamics_sampled(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    double ground_energy,
    const ComplexVector& ground_state,
    const LTLMParameters& params,
    const std::vector<double>& temperatures)
{
    const uint64_t R = std::max<uint64_t>(1, params.num_samples);
    const double D = static_cast<double>(N);
    const double complement_factor = (D > 1.0) ? (D - 1.0) / static_cast<double>(R)
                                               : 0.0;

    // Per-sample (eps_j, w_j) over the GS complement.
    std::vector<std::vector<double>> all_eps(R), all_w(R);
    std::uint64_t base_seed = params.random_seed;
    if (base_seed == 0) base_seed = 0x9E3779B97F4A7C15ULL;
    for (uint64_t s = 0; s < R; ++s) {
        std::uint64_t z = base_seed + 0x9E3779B97F4A7C15ULL * (s + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z =  z ^ (z >> 31);
        std::mt19937 gen(static_cast<std::mt19937::result_type>(z));
        ComplexVector v = generateGaussianRandomVector(N, gen);
        // Project off the exact ground state: |v> -= |0><0|v>.
        Complex proj;
        cblas_zdotc_sub(N, ground_state.data(), 1, v.data(), 1, &proj);
        Complex neg(-proj.real(), -proj.imag());
        cblas_zaxpy(N, &neg, ground_state.data(), 1, v.data(), 1);
        double nrm = cblas_dznrm2(N, v.data(), 1);
        if (nrm < 1e-300) continue;
        Complex inv(1.0 / nrm, 0.0);
        cblas_zscal(N, &inv, v.data(), 1);

        std::vector<double> alpha, beta;
        build_lanczos_tridiagonal(H, v, N, params.krylov_dim, params.tolerance,
                                  params.full_reorthogonalization,
                                  params.reorth_frequency, alpha, beta);
        diagonalize_tridiagonal_ritz(alpha, beta, all_eps[s], all_w[s]);
    }

    ThermodynamicData thermo;
    thermo.temperatures = temperatures;
    const std::size_t nT = temperatures.size();
    thermo.energy.resize(nT);
    thermo.specific_heat.resize(nT);
    thermo.entropy.resize(nT);
    thermo.free_energy.resize(nT);

    for (std::size_t t = 0; t < nT; ++t) {
        const double T = temperatures[t];
        const double beta = 1.0 / T;
        // GS term (shifted by E0): weight 1 at energy E0.
        double Ztil = 1.0;
        double U1   = ground_energy;
        double U2   = ground_energy * ground_energy;
        for (uint64_t s = 0; s < R; ++s) {
            const auto& eps = all_eps[s];
            const auto& w   = all_w[s];
            for (std::size_t j = 0; j < eps.size(); ++j) {
                const double boltz = w[j] * std::exp(-beta * (eps[j] - ground_energy));
                const double contrib = complement_factor * boltz;
                Ztil += contrib;
                U1   += contrib * eps[j];
                U2   += contrib * eps[j] * eps[j];
            }
        }
        if (Ztil > 1e-300) {
            const double E_avg  = U1 / Ztil;
            const double E2_avg = U2 / Ztil;
            thermo.energy[t]        = E_avg;
            thermo.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
            thermo.entropy[t]       = beta * (E_avg - ground_energy) + std::log(Ztil);
            thermo.free_energy[t]   = ground_energy - T * std::log(Ztil);
        } else {
            thermo.energy[t]        = ground_energy;
            thermo.specific_heat[t] = 0.0;
            thermo.entropy[t]       = 0.0;
            thermo.free_energy[t]   = ground_energy;
        }
    }
    return thermo;
}

/**
 * @brief (Legacy, retained for reference) GS-local-DOS thermodynamics.
 */
[[maybe_unused]] static ThermodynamicData compute_ltlm_thermodynamics(
    double ground_energy,
    const std::vector<double>& excitation_energies,
    const std::vector<double>& weights,
    const std::vector<double>& temperatures
) {
    ThermodynamicData thermo;
    thermo.temperatures = temperatures;
    
    uint64_t n_temps = temperatures.size();
    uint64_t n_states = excitation_energies.size();
    
    thermo.energy.resize(n_temps);
    thermo.specific_heat.resize(n_temps);
    thermo.entropy.resize(n_temps);
    thermo.free_energy.resize(n_temps);
    
    for (int t = 0; t < n_temps; t++) {
        double T = temperatures[t];
        double beta = 1.0 / T;
        
        // Compute partition function using shifted energies
        // All energies are already relative to some reference
        // Z = Σ_i w_i * exp(-β * E_i)
        double Z = 0.0;
        double E_avg = 0.0;
        double E2_avg = 0.0;
        
        std::vector<double> boltzmann_factors(n_states);
        
        // Compute Boltzmann factors (excitation_energies already include ground state energy)
        for (int i = 0; i < n_states; i++) {
            // For numerical stability, shift by ground state energy
            double shifted_energy = excitation_energies[i] - ground_energy;
            boltzmann_factors[i] = weights[i] * std::exp(-beta * shifted_energy);
            Z += boltzmann_factors[i];
        }
        
        // Normalize and compute expectations
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                double prob = boltzmann_factors[i] / Z;
                E_avg += prob * excitation_energies[i];
                E2_avg += prob * excitation_energies[i] * excitation_energies[i];
            }
            
            // Thermodynamic quantities
            thermo.energy[t] = E_avg;
            thermo.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
            thermo.entropy[t] = beta * (E_avg - ground_energy) + std::log(Z);
            thermo.free_energy[t] = ground_energy - T * std::log(Z);
        } else {
            // Very low temperature - use ground state only
            thermo.energy[t] = ground_energy;
            thermo.specific_heat[t] = 0.0;
            thermo.entropy[t] = 0.0;
            thermo.free_energy[t] = ground_energy;
        }
    }
    
    return thermo;
}

/**
 * @brief Main LTLM driver function
 */
LTLMResults low_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const LTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const ComplexVector* ground_state_input,
    const std::string& output_dir
) {
    // Phase 6.1: dim-aware OMP+BLAS thread cap (see lanczos() rationale).
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    std::cout << "\n==========================================\n";
    std::cout << "Low Temperature Lanczos Method (LTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Ground state Krylov dim: " << params.ground_state_krylov << std::endl;
    std::cout << "Excitation Krylov dim: " << params.krylov_dim << std::endl;
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    
    // Generate temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(static_cast<uint64_t>(1), num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    LTLMResults results;
    results.total_samples = 1;
    
    // P0.12: was safe_system_call("mkdir -p ...").
    if (!output_dir.empty() && params.store_intermediate) {
        std::error_code ec;
        std::filesystem::create_directories(output_dir + "/ltlm_data", ec);
    }
    
    // Step 1: Find or use ground state
    ComplexVector ground_state;
    double ground_energy;
    
    if (ground_state_input != nullptr && params.use_exact_ground_state) {
        std::cout << "\n--- Using provided ground state ---\n";
        ground_state = *ground_state_input;

        // Compute ground state energy
        ComplexVector H_gs(N);
        H(ground_state.data(), H_gs.data(), N);
        Complex energy_complex;
        cblas_zdotc_sub(N, ground_state.data(), 1, H_gs.data(), 1, &energy_complex);
        ground_energy = std::real(energy_complex);

        std::cout << "Ground state energy: " << ground_energy << std::endl;

        // Krylov-collapse guard. If the supplied vector is (numerically)
        // an exact eigenstate, the next Lanczos chain in
        // build_excitation_spectrum() collapses to dim 1 because the first
        // Lanczos residual w = (H - E) v0 is zero. Detect that case and
        // perturb v0 with a small perpendicular Gaussian so the Krylov
        // expansion still spans a useful low-energy subspace.
        ComplexVector residual = H_gs;
        Complex neg_E(-ground_energy, 0.0);
        cblas_zaxpy(N, &neg_E, ground_state.data(), 1, residual.data(), 1);
        double res_norm = cblas_dznrm2(N, residual.data(), 1);
        const double res_floor = 1e-8;
        if (res_norm < res_floor) {
            std::cout << "  [LTLM] ground state appears to be a numerical eigenvector "
                      << "(||H v - E v|| = " << res_norm
                      << " < " << res_floor << ").\n"
                      << "  [LTLM] Mixing in a small orthogonal perturbation to keep the "
                      << "excitation Krylov space non-trivial.\n";
            std::mt19937 gen(0xCA53D9D2u);  // fixed seed: reproducible perturbation
            std::normal_distribution<double> nd(0.0, 1.0);
            ComplexVector pert(N);
            for (uint64_t i = 0; i < N; ++i) {
                pert[i] = Complex(nd(gen), nd(gen));
            }
            // Project out the ground-state direction.
            Complex proj;
            cblas_zdotc_sub(N, ground_state.data(), 1, pert.data(), 1, &proj);
            Complex neg_proj(-proj.real(), -proj.imag());
            cblas_zaxpy(N, &neg_proj, ground_state.data(), 1, pert.data(), 1);
            double pert_norm = cblas_dznrm2(N, pert.data(), 1);
            if (pert_norm > 0.0) {
                // Mix at amplitude 1e-3 so the dominant content is still v0
                // but Lanczos sees a non-zero off-diagonal at step 1.
                Complex mix(1e-3 / pert_norm, 0.0);
                cblas_zaxpy(N, &mix, pert.data(), 1, ground_state.data(), 1);
                double new_norm = cblas_dznrm2(N, ground_state.data(), 1);
                Complex inv(1.0 / new_norm, 0.0);
                cblas_zscal(N, &inv, ground_state.data(), 1);
            }
        }
    } else {
        std::cout << "\n--- Step 1: Finding Ground State ---\n";
        ground_energy = find_ground_state_lanczos(
            H, N, params.ground_state_krylov, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            ground_state
        );
    }
    
    results.ground_state_energy = ground_energy;
    
    // Step 2 (Jul 2026): correct LTLM thermal trace -- exact GS + random-
    // vector sampling of the complement (compute_ltlm_thermodynamics_sampled).
    // The old build_excitation_spectrum path (single GS-seeded Krylov) is
    // retained above only as reference; it computed the GS-local DOS, not the
    // thermal trace, and is no longer on the production path.
    std::cout << "\n--- Step 2: LTLM thermal trace (exact GS + "
              << params.num_samples << " complement samples) ---\n";
    results.krylov_dimension = params.krylov_dim;
    results.low_lying_spectrum = { ground_energy };

    // Step 3: Compute thermodynamics
    std::cout << "\n--- Step 3: Computing Thermodynamics ---\n";
    results.thermo_data = compute_ltlm_thermodynamics_sampled(
        H, N, ground_energy, ground_state, params, temperatures
    );
    
    // Initialize error bars to zero (LTLM is deterministic with single sample)
    results.energy_error.resize(num_temp_bins, 0.0);
    results.specific_heat_error.resize(num_temp_bins, 0.0);
    results.entropy_error.resize(num_temp_bins, 0.0);
    results.free_energy_error.resize(num_temp_bins, 0.0);
    
    std::cout << "\n==========================================\n";
    std::cout << "LTLM Calculation Complete\n";
    std::cout << "==========================================\n";
    std::cout << "Ground state energy: " << ground_energy << std::endl;
    std::cout << "Complement samples: " << params.num_samples << std::endl;

    return results;
}

StaticResponseResults compute_connected_qh_response_ltlm(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const LTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir
) {
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    if (params.krylov_dim == 0) {
        throw std::invalid_argument(
            "compute_connected_qh_response_ltlm: krylov_dim must be > 0.");
    }

    const bool verbose = ltlm_static_verbose();
    const uint64_t outer_krylov_dim =
        std::max<uint64_t>(1, std::max(params.ground_state_krylov, params.krylov_dim));

    StaticResponseResults results;
    results.total_samples = 1;
    results.temperatures = make_log_temperature_grid(temp_min, temp_max, num_temp_bins);
    results.expectation.assign(num_temp_bins, 0.0);
    results.variance.assign(num_temp_bins, 0.0);
    results.susceptibility.assign(num_temp_bins, 0.0);
    results.expectation_error.assign(num_temp_bins, 0.0);
    results.variance_error.assign(num_temp_bins, 0.0);
    results.susceptibility_error.assign(num_temp_bins, 0.0);

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Connected Q-H Response (LTLM)\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Retained lowest Ritz states: " << params.krylov_dim << std::endl;
        std::cout << "Outer Krylov dimension: " << outer_krylov_dim << std::endl;
        std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    }

    if (!output_dir.empty() && params.store_intermediate) {
        std::error_code ec;
        std::filesystem::create_directories(output_dir + "/static_connected_qh_samples", ec);
    }

    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(static_cast<std::uint32_t>(params.random_seed));
    }

    const ComplexVector v0 = generateGaussianRandomVector(static_cast<int>(N), gen);

    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;
    const uint64_t outer_iters = build_lanczos_tridiagonal_with_basis(
        H, v0, N, outer_krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta, &lanczos_vectors
    );

    if (outer_iters == 0 || alpha.empty()) {
        throw std::runtime_error(
            "compute_connected_qh_response_ltlm: outer Lanczos produced 0 iterations.");
    }

    std::vector<double> energies, weights_unused, evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, energies, weights_unused, &evecs);
    const uint64_t n_ritz = energies.size();
    if (n_ritz == 0) {
        throw std::runtime_error(
            "compute_connected_qh_response_ltlm: Ritz diagonalization returned 0 states.");
    }

    const uint64_t n_keep = std::min<uint64_t>(params.krylov_dim, n_ritz);
    const double energy_shift = energies.front();

    if (verbose) {
        std::cout << "Outer Ritz states available: " << n_ritz << std::endl;
        std::cout << "Lowest-energy shift used: " << energy_shift << std::endl;
    }

    ComplexVector psi_n(N, Complex(0.0, 0.0));
    ComplexVector O_psi_n(N, Complex(0.0, 0.0));
    std::vector<double> q_values(n_keep, 0.0);

    for (uint64_t n = 0; n < n_keep; ++n) {
        std::fill(psi_n.begin(), psi_n.end(), Complex(0.0, 0.0));
        for (uint64_t j = 0; j < n_ritz; ++j) {
            const Complex coeff(evecs[n * n_ritz + j], 0.0);
            cblas_zaxpy(N, &coeff, lanczos_vectors[j].data(), 1, psi_n.data(), 1);
        }

        O(psi_n.data(), O_psi_n.data(), N);
        Complex q_complex;
        cblas_zdotc_sub(N, psi_n.data(), 1, O_psi_n.data(), 1, &q_complex);
        q_values[n] = std::real(q_complex);
    }

    for (uint64_t t = 0; t < num_temp_bins; ++t) {
        const double T = results.temperatures[t];
        const double beta_T = 1.0 / T;

        double Z = 0.0;
        double sum_Q = 0.0;
        double sum_H = 0.0;
        double sum_QH = 0.0;

        for (uint64_t n = 0; n < n_keep; ++n) {
            const double boltz = std::exp(-beta_T * (energies[n] - energy_shift));
            const double qn = q_values[n];
            const double en = energies[n];
            Z += boltz;
            sum_Q += boltz * qn;
            sum_H += boltz * en;
            sum_QH += boltz * qn * en;
        }

        if (Z <= 0.0) {
            continue;
        }

        const double inv_Z = 1.0 / Z;
        const double q_avg = sum_Q * inv_Z;
        const double h_avg = sum_H * inv_Z;
        const double qh_avg = sum_QH * inv_Z;
        const double connected = qh_avg - q_avg * h_avg;

        results.variance[t] = connected;
        results.susceptibility[t] = connected / T;
        results.expectation[t] = connected / (T * T);
    }

    if (params.store_intermediate) {
        StaticResponseSample sample_data;
        sample_data.ritz_values.assign(energies.begin(), energies.begin() + n_keep);
        sample_data.weights.assign(n_keep, 1.0);
        sample_data.expectation_values = q_values;
        sample_data.lanczos_iterations = outer_iters;
        results.per_sample_data.push_back(std::move(sample_data));
    }

    return results;
}

