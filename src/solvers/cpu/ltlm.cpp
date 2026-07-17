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

