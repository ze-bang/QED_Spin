#include "observables.h"


// Calculate thermodynamic quantities directly from eigenvalues
ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min,        // Minimum temperature
    double T_max,        // Maximum temperature
    uint64_t num_points        // Number of temperature points
) {
    ThermodynamicData results;
    
    // Generate logarithmically spaced temperature points
    results.temperatures.resize(num_points);
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (log_T_max - log_T_min) / (num_points - 1);
    
    for (int i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // Resize other arrays
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Find ground state energy (useful for numerical stability)
    double E0 = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    
    // For each temperature
    for (int i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        // Use log-sum-exp trick for numerical stability in calculating Z
        // Find the maximum value for normalization
        double max_exp = -beta * E0;  // Start with ground state
        
        // Calculate partition function Z and energy using log-sum-exp trick
        double sum_exp = 0.0;
        double sum_E_exp = 0.0;
        double sum_E2_exp = 0.0;
        
        for (double E : eigenvalues) {
            double delta_E = E - E0;
            double exp_term = std::exp(-beta * delta_E);
            
            sum_exp += exp_term;
            sum_E_exp += E * exp_term;
            sum_E2_exp += E * E * exp_term;
        }
        
        // Calculate log(Z) = log(sum_exp) + (-beta*E0)
        double log_Z = std::log(sum_exp) - beta * E0;
        
        // Free energy F = -T * log(Z)
        results.free_energy[i] = -T * log_Z;
        
        // Energy E = (1/Z) * sum_i E_i * exp(-beta*E_i)
        results.energy[i] = sum_E_exp / sum_exp;
        
        // Specific heat C_v = beta^2 * (⟨E^2⟩ - ⟨E⟩^2)
        double avg_E2 = sum_E2_exp / sum_exp;
        double avg_E_squared = results.energy[i] * results.energy[i];
        results.specific_heat[i] = beta * beta * (avg_E2 - avg_E_squared);
        
        // Entropy S = (E - F) / T
        results.entropy[i] = (results.energy[i] - results.free_energy[i]) / T;
    }
    
    // Handle special case for T → 0 (avoid numerical issues)
    if (T_min < 1e-6) {
        // In the limit T → 0, only the ground state contributes
        // Energy → E0
        results.energy[0] = E0;
        
        // Specific heat → 0
        results.specific_heat[0] = 0.0;
        
        // Entropy → 0 (third law of thermodynamics) or ln(g) if g-fold degenerate
        uint64_t degeneracy = 0;
        for (double E : eigenvalues) {
            if (std::abs(E - E0) < 1e-10) degeneracy++;
        }
        results.entropy[0] = (degeneracy > 1) ? std::log(degeneracy) : 0.0;
        
        // Free energy → E0 - TS
        results.free_energy[0] = E0 - results.temperatures[0] * results.entropy[0];
    }
    
    return results;
}

// Calculate thermal expectation value of operator A using eigenvalues and eigenvectors
// <A> = (1/Z) * ∑_i exp(-β*E_i) * <ψ_i|A|ψ_i>
Complex calculate_thermal_expectation(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    uint64_t N,                                               // Hilbert space dimension
    double beta,                                         // Inverse temperature β = 1/kT
    const std::string& eig_dir                           // Directory with eigenvector files
) {

    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.dat";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return Complex(0.0, 0.0);
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();

    // Using the log-sum-exp trick for numerical stability
    // Find the maximum value for normalization
    double max_val = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_val = std::max(max_val, -beta * eigenvalues[i]);
    }
    
    // Calculate the numerator <A> = ∑_i exp(-β*E_i) * <ψ_i|A|ψ_i>
    Complex numerator(0.0, 0.0);
    double sum_exp = 0.0;
    
    // Temporary vector to store A|ψ_i⟩
    ComplexVector A_psi(N);
    ComplexVector psi_i(N);
    
    // Calculate both the numerator and Z in one loop
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        // Calculate the Boltzmann factor with numerical stability
        double boltzmann = std::exp(-beta * eigenvalues[i] - max_val);
        sum_exp += boltzmann;
        
        // Load eigenvector from file
        std::string evec_file = eig_dir + "/eigenvector_" + std::to_string(i) + ".dat";
        std::ifstream infile(evec_file, std::ios::binary);
        if (!infile) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file << std::endl;
            continue;
        }
        infile.read(reinterpret_cast<char*>(psi_i.data()), N * sizeof(Complex));
        infile.close();
        
        // Calculate <ψ_i|A|ψ_i>
        A(psi_i.data(), A_psi.data(), N);
        
        Complex expectation;
        cblas_zdotc_sub(N, psi_i.data(), 1, A_psi.data(), 1, &expectation);
        
        // Add contribution to numerator
        numerator += boltzmann * expectation;
    }
    
    // Return <A> = numerator/Z
    return numerator / sum_exp;
}


// Calculate matrix element <ψ₁|A|ψ₂> between two state vectors
Complex calculate_matrix_element(
    std::function<void(const Complex*, Complex*, int)> A,  // Operator A
    const ComplexVector& psi1,                           // First state vector |ψ₁⟩
    const ComplexVector& psi2,                           // Second state vector |ψ₂⟩
    uint64_t N                                               // Dimension of Hilbert space
) {
    // Check that dimensions match
    if (psi1.size() != N || psi2.size() != N) {
        std::cerr << "Error: State vector dimensions don't match Hilbert space dimension" << std::endl;
        return Complex(0.0, 0.0);
    }
    
    // Apply operator A to |ψ₂⟩: A|ψ₂⟩
    ComplexVector A_psi2(N);
    A(psi2.data(), A_psi2.data(), N);
    
    // Calculate <ψ₁|A|ψ₂>
    Complex matrix_element;
    cblas_zdotc_sub(N, psi1.data(), 1, A_psi2.data(), 1, &matrix_element);
    
    return matrix_element;
}

SpectralFunctionData calculate_spectral_function(
    std::function<void(const Complex*, Complex*, int)> O,  // Operator O
    uint64_t N,                                                // Hilbert space dimension
    const std::string& eig_dir,                           // Directory with eigenvector files
    double omega_min,                            // Minimum frequency
    double omega_max,                             // Maximum frequency
    uint64_t num_points,                               // Number of frequency points
    double eta,                                    // Broadening parameter
    double temperature,                            // Temperature (0 for ground state only)
    bool use_lorentzian                          // Use Lorentzian (true) or Gaussian (false) broadening
) {
    SpectralFunctionData result;
    
    // Generate frequency points
    result.frequencies.resize(num_points);
    double omega_step = (omega_max - omega_min) / (num_points - 1);
    for (int i = 0; i < num_points; i++) {
        result.frequencies[i] = omega_min + i * omega_step;
    }
    result.spectral_function.resize(num_points, 0.0);
    
    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.dat";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return result;
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();
    
    // Calculate weights based on temperature
    std::vector<double> weights(num_eigenvalues, 0.0);
    
    if (temperature <= 0.0) {
        // Zero temperature: Only ground state has weight 1.0
        weights[0] = 1.0;
    } else {
        // Finite temperature: Boltzmann weights
        double beta = 1.0 / temperature;
        double min_energy = *std::min_element(eigenvalues.begin(), eigenvalues.end());
        
        // Use log-sum-exp trick for numerical stability
        double max_exp = -beta * min_energy;
        double Z = 0.0;
        
        for (size_t i = 0; i < num_eigenvalues; i++) {
            double boltzmann = std::exp(-beta * (eigenvalues[i] - min_energy));
            weights[i] = boltzmann;
            Z += boltzmann;
        }
        
        // Normalize weights
        for (size_t i = 0; i < num_eigenvalues; i++) {
            weights[i] /= Z;
        }
    }
    
    // Process all transitions between eigenstates
    for (size_t m = 0; m < num_eigenvalues; m++) {
        // Skip states with negligible weight
        if (weights[m] < 1e-12) continue;
        
        // Load eigenstate |m⟩
        ComplexVector psi_m(N);
        std::string evec_file_m = eig_dir + "/eigenvector_" + std::to_string(m) + ".dat";
        std::ifstream infile_m(evec_file_m, std::ios::binary);
        if (!infile_m) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file_m << std::endl;
            continue;
        }
        infile_m.read(reinterpret_cast<char*>(psi_m.data()), N * sizeof(Complex));
        infile_m.close();
        
        // Apply operator O to state |m⟩
        ComplexVector O_psi_m(N);
        O(psi_m.data(), O_psi_m.data(), N);
        
        for (size_t n = 0; n < num_eigenvalues; n++) {
            // Load eigenstate |n⟩
            ComplexVector psi_n(N);
            std::string evec_file_n = eig_dir + "/eigenvector_" + std::to_string(n) + ".dat";
            std::ifstream infile_n(evec_file_n, std::ios::binary);
            if (!infile_n) {
                std::cerr << "Error: Cannot open eigenvector file " << evec_file_n << std::endl;
                continue;
            }
            infile_n.read(reinterpret_cast<char*>(psi_n.data()), N * sizeof(Complex));
            infile_n.close();
            
            // Calculate matrix element ⟨n|O|m⟩
            Complex matrix_element;
            cblas_zdotc_sub(N, psi_n.data(), 1, O_psi_m.data(), 1, &matrix_element);
            
            // Compute |⟨n|O|m⟩|^2
            double intensity = std::norm(matrix_element);
            
            // Skip if intensity is negligible
            if (intensity < 1e-12) continue;
            
            // Compute energy difference
            double delta_E = eigenvalues[n] - eigenvalues[m];
            
            // Add contribution to spectral function
            for (int i = 0; i < num_points; i++) {
                double omega = result.frequencies[i];
                double delta_omega = omega - delta_E;
                
                double broadening;
                if (use_lorentzian) {
                    // Lorentzian broadening
                    broadening = (1.0/M_PI) * (eta / (delta_omega*delta_omega + eta*eta));
                } else {
                    // Gaussian broadening
                    broadening = (1.0/(eta*std::sqrt(2.0*M_PI))) * 
                                 std::exp(-(delta_omega*delta_omega)/(2.0*eta*eta));
                }
                
                // Add weighted contribution
                result.spectral_function[i] += weights[m] * intensity * broadening;
            }
        }
        
        // Progress reporting
        std::cout << "Processed state " << m+1 << " of " << num_eigenvalues << std::endl;
    }
    
    return result;
}



DynamicalSusceptibilityData calculate_dynamical_susceptibility(
    std::function<void(const Complex*, Complex*, int)> A,  // Operator A
    uint64_t N,                                                // Hilbert space dimension
    const std::string& eig_dir,                           // Directory with eigenvector files
    double omega_min,                            // Minimum frequency
    double omega_max,                             // Maximum frequency
    uint64_t num_points,                               // Number of frequency points
    double eta,                                    // Broadening parameter
    double temperature                            // Temperature (in energy units)
) {
    DynamicalSusceptibilityData result;
    
    // Generate frequency points
    result.frequencies.resize(num_points);
    double omega_step = (omega_max - omega_min) / (num_points - 1);
    for (int i = 0; i < num_points; i++) {
        result.frequencies[i] = omega_min + i * omega_step;
    }
    result.chi.resize(num_points, std::complex<double>(0.0, 0.0));
    
    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::string eig_file = eig_dir + "/eigenvalues.txt";
    std::ifstream infile(eig_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_file << std::endl;
        return result;
    }
    size_t num_eigenvalues;
    infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
    eigenvalues.resize(num_eigenvalues);
    infile.read(reinterpret_cast<char*>(eigenvalues.data()), num_eigenvalues * sizeof(double));
    infile.close();
    
    // Calculate Boltzmann probabilities
    double beta = 1.0 / temperature;
    std::vector<double> probabilities(num_eigenvalues);
    
    // Use log-sum-exp trick for numerical stability
    double min_energy = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    double Z = 0.0;
    
    for (size_t i = 0; i < num_eigenvalues; i++) {
        double exp_factor = std::exp(-beta * (eigenvalues[i] - min_energy));
        probabilities[i] = exp_factor; // Temporarily store unnormalized probabilities
        Z += exp_factor;
    }
    
    // Normalize probabilities
    for (size_t i = 0; i < num_eigenvalues; i++) {
        probabilities[i] /= Z;
    }
    
    // Temporary vectors for eigenvectors and operator application
    ComplexVector psi_m(N);
    ComplexVector psi_n(N);
    ComplexVector A_psi_m(N);
    
    // Process all transitions between eigenstates
    for (size_t m = 0; m < num_eigenvalues; m++) {
        // Skip states with negligible weight
        if (probabilities[m] < 1e-12) continue;
        
        // Load eigenstate |m⟩
        std::string evec_file_m = eig_dir + "/eigenvector_" + std::to_string(m) + ".dat";
        std::ifstream infile_m(evec_file_m, std::ios::binary);
        if (!infile_m) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file_m << std::endl;
            continue;
        }
        infile_m.read(reinterpret_cast<char*>(psi_m.data()), N * sizeof(Complex));
        infile_m.close();
        
        // Apply operator A to state |m⟩
        A_psi_m.resize(N);
        A(psi_m.data(), A_psi_m.data(), N);
        
        for (size_t n = 0; n < num_eigenvalues; n++) {
            // Skip if probability difference is too small
            double prob_diff = probabilities[m] - probabilities[n];
            if (std::abs(prob_diff) < 1e-12) continue;
            
            // Load eigenstate |n⟩
            std::string evec_file_n = eig_dir + "/eigenvector_" + std::to_string(n) + ".dat";
            std::ifstream infile_n(evec_file_n, std::ios::binary);
            if (!infile_n) {
                std::cerr << "Error: Cannot open eigenvector file " << evec_file_n << std::endl;
                continue;
            }
            infile_n.read(reinterpret_cast<char*>(psi_n.data()), N * sizeof(Complex));
            infile_n.close();
            
            // Calculate matrix element ⟨n|A|m⟩
            Complex matrix_element;
            cblas_zdotc_sub(N, psi_n.data(), 1, A_psi_m.data(), 1, &matrix_element);
            
            // Compute |⟨n|A|m⟩|^2
            double intensity = std::norm(matrix_element);
            
            // Skip if intensity is negligible
            if (intensity < 1e-12) continue;
            
            // Compute energy difference
            double delta_E = eigenvalues[n] - eigenvalues[m];
            
            // Calculate contribution to susceptibility for each frequency
            for (int i = 0; i < num_points; i++) {
                double omega = result.frequencies[i];
                std::complex<double> denominator(omega - delta_E, eta);
                std::complex<double> contribution = prob_diff * intensity / denominator;
                result.chi[i] += contribution;
            }
        }
        
        // Progress reporting
        std::cout << "Processed state " << m+1 << " of " << num_eigenvalues << std::endl;
    }
    
    return result;
}

// Calculate quantum Fisher information for operator A at temperature T
double calculate_quantum_fisher_information(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    uint64_t N,                                               // Hilbert space dimension
    double temperature,                                  // Temperature (in energy units)
    const std::string& eig_dir                           // Directory with eigenvector files
) {
    // Calculate beta = 1/kT
    double beta = 1.0 / temperature;
    
    // Load eigenvalues from file
    std::vector<double> eigenvalues;
    std::ifstream eigenvalue_file(eig_dir + "/eigenvalues.txt");
    if (!eigenvalue_file) {
        std::cerr << "Error: Cannot open eigenvalue file " << eig_dir + "/eigenvalues.txt" << std::endl;
        return -1;
    }
    
    double eigenvalue;
    eigenvalues.clear();
    while (eigenvalue_file >> eigenvalue) {
        eigenvalues.push_back(eigenvalue);
    }
    size_t num_eigenvalues = eigenvalues.size();
    eigenvalue_file.close();

    std::cout << "Loaded " << num_eigenvalues << " eigenvalues from " << eig_dir + "/eigenvalues.txt" << std::endl;
    
    // Calculate partition function Z and Boltzmann probabilities
    // Using the log-sum-exp trick for numerical stability
    double max_val = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_val = std::max(max_val, -beta * eigenvalues[i]);
    }
    
    double Z = 0.0;
    std::vector<double> probabilities(num_eigenvalues);
    
    for (size_t i = 0; i < num_eigenvalues; i++) {
        double exp_factor = std::exp(-beta * eigenvalues[i] - max_val);
        Z += exp_factor;
        probabilities[i] = exp_factor; // Will normalize by Z later
    }
    
    std::cout << "Calculated partition function Z = " << Z << std::endl;

    // Normalize probabilities
    for (size_t i = 0; i < num_eigenvalues; i++) {
        probabilities[i] /= Z;
    }
    
    // Calculate quantum Fisher information
    double qfi = 0.0;
    
    // Temporary vectors for eigenvectors and operator application
    ComplexVector psi_m(N);
    ComplexVector psi_n(N);
    ComplexVector A_psi_n(N);
    

    // Create a thread pool
    const size_t num_threads = std::thread::hardware_concurrency()/4;
    std::cout << "Using " << num_threads << " threads" << std::endl;
    
    // Split work among threads
    std::vector<std::thread> threads;
    std::vector<double> partial_qfi(num_threads, 0.0);
    std::mutex cout_mutex;
    
    // Launch threads
    for (size_t t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            double local_qfi = 0.0;
            size_t processed_count = 0;
            
            // Each thread processes a subset of states
            for (size_t m = t; m < N; m += num_threads) {
                double p_m = probabilities[m];
                
                // Load eigenvector |m⟩ (only once per thread iteration)
                std::string evec_file_m = eig_dir + "/eigenvectors/eigenvector_" + std::to_string(m) + ".dat";
                std::ifstream evec_stream_m(evec_file_m);
                if (!evec_stream_m) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cerr << "Thread " << t << ": Cannot open eigenvector file " << evec_file_m << std::endl;
                    continue;
                }
                
                ComplexVector psi_m(N);
                uint64_t dimension;
                evec_stream_m >> dimension;
                if (dimension != N) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cerr << "Thread " << t << ": Eigenvector dimension mismatch in " << evec_file_m << std::endl;
                    evec_stream_m.close();
                    continue;
                }
                
                std::fill(psi_m.begin(), psi_m.end(), Complex(0.0, 0.0));
                uint64_t index;
                double real_part, imag_part;
                while (evec_stream_m >> index >> real_part >> imag_part) {
                    psi_m[index] = Complex(real_part, imag_part);
                }
                evec_stream_m.close();
                
                // Apply operator A to |m⟩ once and store the result
                ComplexVector A_psi_m(N);
                A(psi_m.data(), A_psi_m.data(), N);
                
                // Process all significant states for this m
                for (size_t n = 0; n < N; n++) {
                    if (m == n) continue;  // Skip diagonal elements
                    
                    double p_n = probabilities[n];
                    double p_sum = p_m + p_n;
                    
                    double coef = (p_m - p_n) * (p_m - p_n) / p_sum;
                    
                    // Load eigenvector |n⟩
                    std::string evec_file_n = eig_dir + "/eigenvectors/eigenvector_" + std::to_string(n) + ".dat";
                    std::ifstream evec_stream_n(evec_file_n);
                    if (!evec_stream_n) continue;
                    
                    ComplexVector psi_n(N);
                    uint64_t dim_n;
                    evec_stream_n >> dim_n;
                    if (dim_n != N) {
                        evec_stream_n.close();
                        continue;
                    }
                    
                    std::fill(psi_n.begin(), psi_n.end(), Complex(0.0, 0.0));
                    while (evec_stream_n >> index >> real_part >> imag_part) {
                        psi_n[index] = Complex(real_part, imag_part);
                    }
                    evec_stream_n.close();
                    
                    // Calculate matrix element ⟨m|A|n⟩
                    Complex matrix_element;
                    cblas_zdotc_sub(N, psi_m.data(), 1, A_psi_n.data(), 1, &matrix_element);
                    
                    // Add contribution to local QFI
                    local_qfi += 2.0 * coef * std::norm(matrix_element);
                }
                
                // Occasionally report progress
                if (++processed_count % 10 == 0) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "Thread " << t << " processed " << processed_count << "/" 
                              << (N / num_threads + 1) << " states" << std::endl;
                }
            }
            
            // Store the thread's result
            partial_qfi[t] = local_qfi;
        });
    }
    
    // Wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Combine results from all threads
    for (double partial : partial_qfi) {
        qfi += partial;
    }
    
    std::cout << "Quantum Fisher Information calculation complete" << std::endl;
    return qfi;
}

// Compute thermal expectation values of S^+, S^-, S^z operators at each site
void compute_spin_expectations(
    const std::string& eigdir,  // Directory with eigenvalues and eigenvectors
    const std::string output_dir, // Directory for output files
    uint64_t num_sites,              // Number of sites
    float spin_l,              // Spin length (e.g., 0.5 for spin-1/2)
    double temperature,         // Temperature T (in energy units)
    bool print_output    // Whether to print the results to console
) {
    // Calculate the dimension of the Hilbert space
    uint64_t N = 1ULL << num_sites;  // 2^num_sites
    
    // Initialize expectations matrix: 3 rows (S^+, S^-, S^z) x num_sites columns
    std::vector<std::vector<Complex>> expectations(3, std::vector<Complex>(num_sites, Complex(0.0, 0.0)));
    
    std::cout << "Reading eigenvalues and eigenvectors from " << eigdir << std::endl;

    // Load eigenvalues
    std::vector<double> eigenvalues;
    std::ifstream eigenvalue_file(eigdir + "/eigenvalues.txt");
    if (!eigenvalue_file) {
        std::cerr << "Error: Cannot open eigenvalue file " << eigdir + "/eigenvalues.txt" << std::endl;
        return;
    }
    
    double eigenvalue;
    eigenvalues.clear();
    while (eigenvalue_file >> eigenvalue) {
        eigenvalues.push_back(eigenvalue);
    }
    size_t num_eigenvalues = eigenvalues.size();
    eigenvalue_file.close();
    
    std::cout << "Loaded " << num_eigenvalues << " eigenvalues from " << eigdir + "/eigenvalues.txt" << std::endl;
    
    // Calculate beta = 1/kT
    double beta = 1.0 / temperature;
    
    // Using the log-sum-exp trick for numerical stability
    double max_exp = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_exp = std::max(max_exp, -beta * eigenvalues[i]);
    }
    
    // Calculate partition function Z
    double Z = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        Z += std::exp(-beta * eigenvalues[i] - max_exp);
    }
    
    // Create S operators for each site
    std::vector<SingleSiteOperator> Sp_ops;
    std::vector<SingleSiteOperator> Sm_ops;
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sp_ops.emplace_back(num_sites, spin_l, 0, site);
        Sm_ops.emplace_back(num_sites, spin_l, 1, site);
        Sz_ops.emplace_back(num_sites, spin_l, 2, site);
    }
    
    // Process each eigenvector
    for (size_t idx = 0; idx < num_eigenvalues; idx++) {
        // Load eigenvector
        std::string evec_file = eigdir + "/eigenvectors/eigenvector_" + std::to_string(idx) + ".dat";
        std::ifstream evec_stream(evec_file);
        if (!evec_stream) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file << std::endl;
            continue;
        }
        
        ComplexVector psi(N);
        // Read the file contents - first line is dimension
        uint64_t dimension;
        evec_stream >> dimension;
        if (dimension != N) {
            std::cerr << "Error: Eigenvector dimension " << dimension << " doesn't match expected size " << N << std::endl;
            evec_stream.close();
            continue;
        }

        // Initialize psi to zeros
        std::fill(psi.begin(), psi.end(), Complex(0.0, 0.0));

        // Read non-zero entries
        uint64_t index;
        double real_part, imag_part;
        while (evec_stream >> index >> real_part >> imag_part) {
            psi[index] = Complex(real_part, imag_part);
        }
        evec_stream.close();
        
        // Calculate Boltzmann factor
        double boltzmann = std::exp(-beta * eigenvalues[idx] - max_exp) / Z;
        
        // For each site, compute the expectation values
        for (int site = 0; site < num_sites; site++) {
            // Apply operators
            std::vector<Complex> Sp_psi(N);
            std::vector<Complex> Sm_psi(N);
            std::vector<Complex> Sz_psi(N);
            Sp_ops[site].apply(psi.data(), Sp_psi.data(), N);
            Sm_ops[site].apply(psi.data(), Sm_psi.data(), N);
            Sz_ops[site].apply(psi.data(), Sz_psi.data(), N);
            
            // Calculate expectation values
            Complex Sp_exp = Complex(0.0, 0.0);
            Complex Sm_exp = Complex(0.0, 0.0);
            Complex Sz_exp = Complex(0.0, 0.0);
            
            for (int i = 0; i < N; i++) {
                Sp_exp += std::conj(psi[i]) * Sp_psi[i];
                Sm_exp += std::conj(psi[i]) * Sm_psi[i];
                Sz_exp += std::conj(psi[i]) * Sz_psi[i];
            }
            
            // Add weighted contribution to thermal average
            expectations[0][site] += boltzmann * Sp_exp;
            expectations[1][site] += boltzmann * Sm_exp;
            expectations[2][site] += boltzmann * Sz_exp;
        }
        
        // Progress reporting
        if ((idx + 1) % 10 == 0 || idx == num_eigenvalues - 1) {
            std::cout << "Processed " << idx + 1 << "/" << num_eigenvalues << " eigenvectors" << std::endl;
        }
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin Expectation Values at T = " << temperature << ":" << std::endl;
        std::cout << std::setw(5) << "Site" 
                << std::setw(20) << "S^+ (real)" 
                << std::setw(20) << "S^+ (imag)" 
                << std::setw(20) << "S^- (real)"
                << std::setw(20) << "S^- (imag)"
                << std::setw(20) << "S^z (real)"
                << std::setw(20) << "S^z (imag)" << std::endl;
        
        for (int site = 0; site < num_sites; site++) {
            std::cout << std::setw(5) << site 
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
    }
    
    // Save to file
    std::string outfile = output_dir + "/spin_expectations_T" + std::to_string(temperature) + ".dat";
    std::ofstream out(outfile);
    if (out.is_open()) {
        out << "# Site S+_real S+_imag S-_real S-_imag Sz_real Sz_imag" << std::endl;
        for (int site = 0; site < num_sites; site++) {
            out << site << " "
                << std::setprecision(10) << expectations[0][site].real() << " "
                << std::setprecision(10) << expectations[0][site].imag() << " "
                << std::setprecision(10) << expectations[1][site].real() << " "
                << std::setprecision(10) << expectations[1][site].imag() << " "
                << std::setprecision(10) << expectations[2][site].real() << " "
                << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
        out.close();
        std::cout << "Spin expectations saved to " << outfile << std::endl;
    }
}

// Load eigenstate from file with format like eigenvector_block0_0.dat
ComplexVector load_eigenstate_from_file(const std::string& filename, uint64_t expected_dimension) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open eigenvector file " << filename << std::endl;
        return ComplexVector();
    }
    
    // Read the first line which contains dimension
    uint64_t dimension;
    file >> dimension;
    
    if (expected_dimension > 0 && dimension != expected_dimension) {
        std::cerr << "Warning: File dimension " << dimension << " doesn't match expected dimension " 
                  << expected_dimension << std::endl;
    }
    
    // Initialize vector with zeros
    ComplexVector eigenstate(dimension, Complex(0.0, 0.0));
    
    // Read entries
    uint64_t index;
    double real_part, imag_part;

    // skip the first line
    std::string line;
    std::getline(file, line);

    while (file >> index >> real_part >> imag_part) {
        eigenstate[index] = Complex(real_part, imag_part);
    }
    
    file.close();
    return eigenstate;
}

// Load classical eigenstate (basis state with Nth largest amplitude) from file
ComplexVector load_classical_eigenstate_from_file(
    const std::string& filename, 
    uint64_t expected_dimension,
    uint64_t nth_state            // Select the nth most probable state (default: most probable)
) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open eigenvector file " << filename << std::endl;
        return {-1, 0.0};
    }
    
    // Read the first line which contains dimension
    uint64_t dimension;
    file >> dimension;
    
    if (expected_dimension > 0 && dimension != expected_dimension) {
        std::cerr << "Warning: File dimension " << dimension << " doesn't match expected dimension " 
                  << expected_dimension << std::endl;
    }
    
    // Vector to store all basis states and their probabilities
    std::vector<std::pair<int, double>> state_probs;
    
    // Read entries
    uint64_t index;
    double real_part, imag_part;
    
    // Skip the first line
    std::string line;
    std::getline(file, line);
    
    // Sort states by probability (highest first)
    std::vector<std::tuple<int, double, Complex>> state_data;
    state_data.reserve(dimension);

    while (file >> index >> real_part >> imag_part) {
        Complex value(real_part, imag_part);
        double probability = std::norm(value);  // |z|^2 = real^2 + imag^2
        state_data.emplace_back(index, probability, value);
    }

    file.close();

    // Sort by probability (highest first)
    std::sort(state_data.begin(), state_data.end(), 
              [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });

    // Group states with the same probability
    std::vector<std::vector<std::tuple<int, double, Complex>>> groups;
    if (!state_data.empty()) {
        std::vector<std::tuple<int, double, Complex>> current_group;
        double current_prob = std::get<1>(state_data[0]);
        
        for (const auto& state : state_data) {
            // If probability is significantly different, start a new group
            if (std::abs(std::get<1>(state) - current_prob) > 1e-10) {
                if (!current_group.empty()) {
                    groups.push_back(current_group);
                }
                current_group.clear();
                current_prob = std::get<1>(state);
            }
            current_group.push_back(state);
        }
        
        // Add the last group
        if (!current_group.empty()) {
            groups.push_back(current_group);
        }
    }

    // Print information about the groups
    std::cout << "Found " << groups.size() << " groups of states with different probabilities" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), groups.size()); ++i) {
        std::cout << "Group " << (i+1) << ": " << groups[i].size() << " states with probability " 
                  << std::get<1>(groups[i][0]) << std::endl;
    }

    // Check if we have enough groups
    if (nth_state > groups.size()) {
        std::cerr << "Error: Requested " << nth_state << "th group, but only " << groups.size() 
                  << " groups available." << std::endl;
        return ComplexVector();
    }

    // Select the nth group (1-indexed)
    const auto& selected_group = groups[nth_state - 1];
    std::cout << "Selected group " << nth_state << " with " << selected_group.size() 
              << " states of probability " << std::get<1>(selected_group[0]) << std::endl;

    // Create eigenstate with selected states
    ComplexVector eigenstate(dimension, Complex(0.0, 0.0));
    for (const auto& state : selected_group) {
        uint64_t idx = std::get<0>(state);
        Complex value = std::get<2>(state);
        eigenstate[idx] = value;
    }

    // Renormalize the eigenstate
    double norm = 0.0;
    for (const Complex& c : eigenstate) {
        norm += std::norm(c);
    }
    norm = std::sqrt(norm);

    if (norm > 1e-10) {
        for (Complex& c : eigenstate) {
            c /= norm;
        }
    }

    return eigenstate;
}


// Calculate spin expectations for a single eigenstate
std::vector<std::vector<Complex>> compute_eigenstate_spin_expectations(
    const ComplexVector& eigenstate,   // Eigenstate as complex vector
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file,  // Optional: output file path
    bool print_output           // Whether to print the results to console
) {
    // Check dimension
    uint64_t N = 1ULL << num_sites;  // 2^num_sites
    if (eigenstate.size() != N) {
        std::cerr << "Error: Eigenstate dimension " << eigenstate.size() 
                  << " doesn't match expected size " << N << std::endl;
        return {};
    }
    
    // Initialize expectations matrix: 3 rows (S^+, S^-, S^z) x num_sites columns
    std::vector<std::vector<Complex>> expectations(3, std::vector<Complex>(num_sites, Complex(0.0, 0.0)));
    
    // Create spin operators for each site
    std::vector<SingleSiteOperator> Sp_ops;
    std::vector<SingleSiteOperator> Sm_ops;
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sp_ops.emplace_back(num_sites, spin_l, 0, site);
        Sm_ops.emplace_back(num_sites, spin_l, 1, site);
        Sz_ops.emplace_back(num_sites, spin_l, 2, site);
    }
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        // Apply operators
        std::vector<Complex> Sp_psi(N);
        std::vector<Complex> Sm_psi(N);
        std::vector<Complex> Sz_psi(N);
        Sp_ops[site].apply(eigenstate.data(), Sp_psi.data(), N);
        Sm_ops[site].apply(eigenstate.data(), Sm_psi.data(), N);
        Sz_ops[site].apply(eigenstate.data(), Sz_psi.data(), N);
        
        std::vector<Complex> Sx_psi(N);
        std::vector<Complex> Sy_psi(N);

        for (int i = 0; i < N; i++) {
            Sx_psi[i] = 0.5 * (Sp_psi[i] + Sm_psi[i]);
            Sy_psi[i] = Complex(0.0, -0.5) * (Sp_psi[i] - Sm_psi[i]);
        }

        // Calculate expectation values
        Complex Sp_exp = Complex(0.0, 0.0);
        Complex Sm_exp = Complex(0.0, 0.0);
        Complex Sz_exp = Complex(0.0, 0.0);

        for (int i = 0; i < N; i++) {
            Sp_exp += std::conj(eigenstate[i]) * Sx_psi[i];
            Sm_exp += std::conj(eigenstate[i]) * Sy_psi[i];
            Sz_exp += std::conj(eigenstate[i]) * Sz_psi[i];
        }
        
        expectations[0][site] = Sp_exp;
        expectations[1][site] = Sm_exp;
        expectations[2][site] = Sz_exp;
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin Expectation Values for eigenstate:" << std::endl;
        std::cout << std::setw(5) << "Site" 
                << std::setw(20) << "S^x (real)" 
                << std::setw(20) << "S^x (imag)" 
                << std::setw(20) << "S^y (real)"
                << std::setw(20) << "S^y (imag)"
                << std::setw(20) << "S^z (real)"
                << std::setw(20) << "S^z (imag)" << std::endl;
        
        for (int site = 0; site < num_sites; site++) {
            std::cout << std::setw(5) << site 
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
    }
    
    // Save to file if path provided
    if (!output_file.empty()) {
        std::ofstream out(output_file);
        if (out.is_open()) {
            out << "# Site Sx_real Sx_imag Sy_real Sy_imag Sz_real Sz_imag" << std::endl;
            for (int site = 0; site < num_sites; site++) {
                out << site << " "
                    << std::setprecision(10) << expectations[0][site].real() << " "
                    << std::setprecision(10) << expectations[0][site].imag() << " "
                    << std::setprecision(10) << expectations[1][site].real() << " "
                    << std::setprecision(10) << expectations[1][site].imag() << " "
                    << std::setprecision(10) << expectations[2][site].real() << " "
                    << std::setprecision(10) << expectations[2][site].imag() << std::endl;
            }
            out.close();
            std::cout << "Spin expectations saved to " << output_file << std::endl;
        }
    }
    
    return expectations;
}



// Compute two-site correlations (Sz*Sz and S+*S-) for a single eigenstate
std::vector<std::vector<std::vector<Complex>>> compute_eigenstate_spin_correlations(
    const ComplexVector& eigenstate,   // Eigenstate as complex vector
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file,  // Optional: output file path
    bool print_output           // Whether to print the results to console
) {
    // Check dimension
    uint64_t N = 1ULL << num_sites;  // 2^num_sites
    if (eigenstate.size() != N) {
        std::cerr << "Error: Eigenstate dimension " << eigenstate.size() 
                  << " doesn't match expected size " << N << std::endl;
        return {};
    }
    
    // Initialize correlations tensor: 2 types (Sz*Sz, S+*S-) x num_sites x num_sites
    std::vector<std::vector<std::vector<Complex>>> correlations(
        2, std::vector<std::vector<Complex>>(
            num_sites, std::vector<Complex>(num_sites, Complex(0.0, 0.0))
        )
    );
    
    // For each site pair, compute correlations
    for (int site_i = 0; site_i < num_sites; site_i++) {
        for (int site_j = 0; site_j < num_sites; site_j++) {
            // Skip if it's the same site
            
            // Create double site operators
            DoubleSiteOperator SzSz_op(num_sites, spin_l, 2, site_i, 2, site_j); // Sz*Sz
            DoubleSiteOperator SpSm_op(num_sites, spin_l, 0, site_i, 1, site_j); // S+*S-
            
            // Apply operators
            std::vector<Complex> SzSz_psi(N);
            std::vector<Complex> SpSm_psi(N);
            SzSz_op.apply(eigenstate.data(), SzSz_psi.data(), N);
            SpSm_op.apply(eigenstate.data(), SpSm_psi.data(), N);
            
            // Calculate expectation values
            Complex SzSz_exp = Complex(0.0, 0.0);
            Complex SpSm_exp = Complex(0.0, 0.0);
            
            for (int i = 0; i < N; i++) {
                SzSz_exp += std::conj(eigenstate[i]) * SzSz_psi[i];
                SpSm_exp += std::conj(eigenstate[i]) * SpSm_psi[i];
            }
            
            // Store correlation values
            correlations[0][site_i][site_j] = SzSz_exp;  // Sz*Sz
            correlations[1][site_i][site_j] = SpSm_exp;  // S+*S-
        }
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin-Spin Correlations for eigenstate:" << std::endl;
        
        // Print Sz*Sz correlations
        std::cout << "\nSz*Sz Correlations:" << std::endl;
        std::cout << std::setw(5) << "i\\j";
        for (int j = 0; j < num_sites; j++) {
            std::cout << std::setw(10) << j;
        }
        std::cout << std::endl;
        
        for (int i = 0; i < num_sites; i++) {
            std::cout << std::setw(5) << i;
            for (int j = 0; j < num_sites; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << correlations[0][i][j].real();
            }
            std::cout << std::endl;
        }
        
        // Print S+*S- correlations
        std::cout << "\nS+*S- Correlations:" << std::endl;
        std::cout << std::setw(5) << "i\\j";
        for (int j = 0; j < num_sites; j++) {
            std::cout << std::setw(10) << j;
        }
        std::cout << std::endl;
        
        for (int i = 0; i < num_sites; i++) {
            std::cout << std::setw(5) << i;
            for (int j = 0; j < num_sites; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << correlations[1][i][j].real();
            }
            std::cout << std::endl;
        }
    }
    
    // Save to file if path provided
    if (!output_file.empty()) {
        std::ofstream out(output_file);
        if (out.is_open()) {
            out << "# Site_i Site_j SzSz_real SzSz_imag SpSm_real SpSm_imag" << std::endl;
            for (int i = 0; i < num_sites; i++) {
                for (int j = 0; j < num_sites; j++) {
                    out << i << " " << j << " "
                        << std::setprecision(10) << correlations[0][i][j].real() << " "
                        << std::setprecision(10) << correlations[0][i][j].imag() << " "
                        << std::setprecision(10) << correlations[1][i][j].real() << " "
                        << std::setprecision(10) << correlations[1][i][j].imag() << std::endl;
                }
            }
            out.close();
            std::cout << "Spin correlations saved to " << output_file << std::endl;
        }
    }
    
    return correlations;
}

// Compute spin expectations for a specific eigenstate loaded from a file
std::vector<std::vector<Complex>> compute_eigenstate_spin_expectations_from_file(
    const std::string& eigenstate_file, // File containing the eigenstate
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file,  // Optional: output file path
    bool print_output,           // Whether to print the results to console
    bool classical,               // Whether to load a classical eigenstate
    uint64_t nth_state                   // Select the nth most probable state (default: most probable)
) {
    // Calculate expected dimension
    uint64_t expected_dimension = 1ULL << num_sites;  // 2^num_sites
    
    // Load eigenstate from file
    ComplexVector eigenstate;
    if (classical) {
        eigenstate = load_classical_eigenstate_from_file(eigenstate_file, expected_dimension, nth_state);
    } else {
        eigenstate = load_eigenstate_from_file(eigenstate_file, expected_dimension);
    }

    if (eigenstate.empty()) {
        std::cerr << "Error: Failed to load eigenstate from " << eigenstate_file << std::endl;
        return {};
    }
    
    std::cout << "Loaded eigenstate from " << eigenstate_file << " with dimension " << eigenstate.size() << std::endl;
    // Calculate spin expectations
    return compute_eigenstate_spin_expectations(eigenstate, num_sites, spin_l, output_file, print_output);
}

std::vector<std::vector<std::vector<Complex>>> compute_eigenstate_spin_correlations_from_file(
    const std::string& eigenstate_file, // File containing the eigenstate
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file,  // Optional: output file path
    bool print_output,           // Whether to print the results to console
    bool classical,               // Whether to load a classical eigenstate
    uint64_t nth_state                 // Select the nth most probable state (default: most probable)
) {
    // Calculate expected dimension
    uint64_t expected_dimension = 1ULL << num_sites;  // 2^num_sites
    
    // Load eigenstate from file
    ComplexVector eigenstate;
    if (classical) {
        eigenstate = load_classical_eigenstate_from_file(eigenstate_file, expected_dimension, nth_state);
    } else {
        eigenstate = load_eigenstate_from_file(eigenstate_file, expected_dimension);
    }

    if (eigenstate.empty()) {
        std::cerr << "Error: Failed to load eigenstate from " << eigenstate_file << std::endl;
        return {};
    }
    
    std::cout << "Loaded eigenstate from " << eigenstate_file << " with dimension " << eigenstate.size() << std::endl;
    // Calculate spin correlations
    return compute_eigenstate_spin_correlations(eigenstate, num_sites, spin_l, output_file, print_output);
}


// Compute thermal expectation values of two-site correlators (Sz*Sz and S+*S-)
void compute_spin_correlations(
    const std::string& eigdir,  // Directory with eigenvalues and eigenvectors
    const std::string output_dir, // Directory for output files
    uint64_t num_sites,              // Number of sites
    float spin_l,              // Spin length (e.g., 0.5 for spin-1/2)
    double temperature,         // Temperature T (in energy units)
    bool print_output    // Whether to print the results to console
) {
    // Calculate the dimension of the Hilbert space
    uint64_t N = 1ULL << num_sites;  // 2^num_sites
    
    // Initialize correlations tensor: 2 types (Sz*Sz, S+*S-) x num_sites x num_sites
    std::vector<std::vector<std::vector<Complex>>> correlations(
        2, std::vector<std::vector<Complex>>(
            num_sites, std::vector<Complex>(num_sites, Complex(0.0, 0.0))
        )
    );
    
    std::cout << "Reading eigenvalues and eigenvectors from " << eigdir << std::endl;

    // Load eigenvalues
    std::vector<double> eigenvalues;
    std::ifstream eigenvalue_file(eigdir + "/eigenvalues.txt");
    if (!eigenvalue_file) {
        std::cerr << "Error: Cannot open eigenvalue file " << eigdir + "/eigenvalues.txt" << std::endl;
        return;
    }
    
    double eigenvalue;
    eigenvalues.clear();
    while (eigenvalue_file >> eigenvalue) {
        eigenvalues.push_back(eigenvalue);
    }
    size_t num_eigenvalues = eigenvalues.size();
    eigenvalue_file.close();
    
    std::cout << "Loaded " << num_eigenvalues << " eigenvalues from " << eigdir + "/eigenvalues.txt" << std::endl;
    
    // Calculate beta = 1/kT
    double beta = 1.0 / temperature;
    
    // Using the log-sum-exp trick for numerical stability
    double max_exp = -beta * eigenvalues[0];
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        max_exp = std::max(max_exp, -beta * eigenvalues[i]);
    }
    
    // Calculate partition function Z
    double Z = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        Z += std::exp(-beta * eigenvalues[i] - max_exp);
    }
    
    // Precompute all two-site operators
    std::vector<std::vector<DoubleSiteOperator>> SzSz_ops(num_sites, std::vector<DoubleSiteOperator>(num_sites));
    std::vector<std::vector<DoubleSiteOperator>> SpSm_ops(num_sites, std::vector<DoubleSiteOperator>(num_sites));
    
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            if (i != j) {
                SzSz_ops[i][j] = DoubleSiteOperator(num_sites, spin_l, 2, i, 2, j);
                SpSm_ops[i][j] = DoubleSiteOperator(num_sites, spin_l, 0, i, 1, j);
            }
        }
    }
    
    // Process each eigenvector
    for (size_t idx = 0; idx < num_eigenvalues; idx++) {
        // Load eigenvector
        std::string evec_file = eigdir + "/eigenvectors/eigenvector_" + std::to_string(idx) + ".dat";
        std::ifstream evec_stream(evec_file);
        if (!evec_stream) {
            std::cerr << "Error: Cannot open eigenvector file " << evec_file << std::endl;
            continue;
        }
        
        ComplexVector psi(N);
        // Read the file contents - first line is dimension
        uint64_t dimension;
        evec_stream >> dimension;
        if (dimension != N) {
            std::cerr << "Error: Eigenvector dimension " << dimension << " doesn't match expected size " << N << std::endl;
            evec_stream.close();
            continue;
        }

        // Initialize psi to zeros
        std::fill(psi.begin(), psi.end(), Complex(0.0, 0.0));

        // Read non-zero entries
        uint64_t index;
        double real_part, imag_part;
        while (evec_stream >> index >> real_part >> imag_part) {
            psi[index] = Complex(real_part, imag_part);
        }
        evec_stream.close();
        
        // Calculate Boltzmann factor
        double boltzmann = std::exp(-beta * eigenvalues[idx] - max_exp) / Z;
        
        // For each pair of sites, compute the correlation functions
        for (int i = 0; i < num_sites; i++) {
            for (int j = 0; j < num_sites; j++) {
                if (i == j) continue; // Skip same site
                
                // Apply operators
                std::vector<Complex> SzSz_psi(N);
                std::vector<Complex> SpSm_psi(N);
                SzSz_ops[i][j].apply(psi.data(), SzSz_psi.data(), N);
                SpSm_ops[i][j].apply(psi.data(), SpSm_psi.data(), N);
                
                // Calculate expectation values
                Complex SzSz_exp = Complex(0.0, 0.0);
                Complex SpSm_exp = Complex(0.0, 0.0);
                
                for (int k = 0; k < N; k++) {
                    SzSz_exp += std::conj(psi[k]) * SzSz_psi[k];
                    SpSm_exp += std::conj(psi[k]) * SpSm_psi[k];
                }
                
                // Add weighted contribution to thermal average
                correlations[0][i][j] += boltzmann * SzSz_exp; // Sz*Sz
                correlations[1][i][j] += boltzmann * SpSm_exp; // S+*S-
            }
        }
        
        // Progress reporting
        if ((idx + 1) % 10 == 0 || idx == num_eigenvalues - 1) {
            std::cout << "Processed " << idx + 1 << "/" << num_eigenvalues << " eigenvectors" << std::endl;
        }
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin-Spin Correlations at T = " << temperature << ":" << std::endl;
        
        // Print Sz*Sz correlations
        std::cout << "\nSz*Sz Correlations:" << std::endl;
        std::cout << std::setw(5) << "i\\j";
        for (int j = 0; j < num_sites; j++) {
            std::cout << std::setw(10) << j;
        }
        std::cout << std::endl;
        
        for (int i = 0; i < num_sites; i++) {
            std::cout << std::setw(5) << i;
            for (int j = 0; j < num_sites; j++) {
                if (i == j) {
                    std::cout << std::setw(10) << "---";
                } else {
                    std::cout << std::setw(10) << std::setprecision(4) << correlations[0][i][j].real();
                }
            }
            std::cout << std::endl;
        }
        
        // Print S+*S- correlations
        std::cout << "\nS+*S- Correlations:" << std::endl;
        std::cout << std::setw(5) << "i\\j";
        for (int j = 0; j < num_sites; j++) {
            std::cout << std::setw(10) << j;
        }
        std::cout << std::endl;
        
        for (int i = 0; i < num_sites; i++) {
            std::cout << std::setw(5) << i;
            for (int j = 0; j < num_sites; j++) {
                if (i == j) {
                    std::cout << std::setw(10) << "---";
                } else {
                    std::cout << std::setw(10) << std::setprecision(4) << correlations[1][i][j].real();
                }
            }
            std::cout << std::endl;
        }
    }
    
    // Save to file
    std::string outfile = output_dir + "/spin_correlations_T" + std::to_string(temperature) + ".dat";
    std::ofstream out(outfile);
    if (out.is_open()) {
        out << "# Site_i Site_j SzSz_real SzSz_imag SpSm_real SpSm_imag" << std::endl;
        for (int i = 0; i < num_sites; i++) {
            for (int j = 0; j < num_sites; j++) {
                if (i != j) {
                    out << i << " " << j << " "
                        << std::setprecision(10) << correlations[0][i][j].real() << " "
                        << std::setprecision(10) << correlations[0][i][j].imag() << " "
                        << std::setprecision(10) << correlations[1][i][j].real() << " "
                        << std::setprecision(10) << correlations[1][i][j].imag() << std::endl;
                }
            }
        }
        out.close();
        std::cout << "Spin correlations saved to " << outfile << std::endl;
    }
}
