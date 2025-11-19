#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <limits>
#include <cstring>
#include <cmath> // added for M_PI, std::sqrt
#include <iomanip> // added for std::setprecision and std::fixed
#include <algorithm> // for std::sort, std::max_element, std::min_element
#include <numeric> // for std::accumulate
#include "construct_ham.h"
#include "../cpu_solvers/TPQ.h"
#include "observables.h"
#include <mpi.h>
#include "../cpu_solvers/dynamics.h"
#include "../cpu_solvers/ftlm.h"

#ifdef WITH_CUDA
#include "../gpu/gpu_ed_wrapper.h"
#include "../gpu/gpu_operator.cuh"
#endif

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
namespace fs = std::filesystem;


// Helper function to read num_sites from positions.dat file
int read_num_sites_from_positions(const std::string& positions_file) {
    std::ifstream file(positions_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open positions.dat file: " + positions_file);
    }
    
    int num_sites = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Skip comment lines starting with #
        if (line.empty() || line[0] == '#') continue;
        num_sites++;
    }
    
    file.close();
    
    if (num_sites == 0) {
        throw std::runtime_error("No sites found in positions.dat file: " + positions_file);
    }
    
    return num_sites;
}

// Helper function to read ground state energy with multiple fallback methods
// Returns the MINIMUM energy across all available sources for robustness against corruption
double read_ground_state_energy(const std::string& directory) {
    std::vector<double> candidate_energies;
    std::vector<std::string> sources;
    
    // Method 1: Try eigenvalues.dat (binary format)
    std::string eigenvalues_dat = directory + "/output/eigenvectors/eigenvalues.dat";
    std::ifstream infile_dat(eigenvalues_dat, std::ios::binary);
    if (infile_dat.is_open()) {
        size_t num_eigenvalues;
        infile_dat.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
        
        if (num_eigenvalues > 0) {
            double ground_state_energy;
            infile_dat.read(reinterpret_cast<char*>(&ground_state_energy), sizeof(double));
            candidate_energies.push_back(ground_state_energy);
            sources.push_back("eigenvalues.dat");
        }
        infile_dat.close();
    }
    
    // Method 2: Try eigenvalues.txt (text format)
    std::string eigenvalues_txt = directory + "/output/eigenvalues.txt";
    std::ifstream infile_txt(eigenvalues_txt);
    if (infile_txt.is_open()) {
        double ground_state_energy;
        if (infile_txt >> ground_state_energy) {
            candidate_energies.push_back(ground_state_energy);
            sources.push_back("eigenvalues.txt");
        }
        infile_txt.close();
    }
    
    // Method 3: Try finding minimum energy in SS_rand0.dat
    std::string ss_file = directory + "/output/SS_rand0.dat";
    std::ifstream infile_ss(ss_file);
    if (infile_ss.is_open()) {
        std::string line;
        double min_energy = std::numeric_limits<double>::max();
        bool found_energy = false;
        
        while (std::getline(infile_ss, line)) {
            // Skip comment lines and empty lines
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            double inv_temp, energy;
            
            // Read first two columns: inv_temp and energy
            if (iss >> inv_temp >> energy) {
                if (energy < min_energy) {
                    min_energy = energy;
                    found_energy = true;
                }
            }
        }
        infile_ss.close();
        
        if (found_energy) {
            candidate_energies.push_back(min_energy);
            sources.push_back("SS_rand0.dat");
        }
    }
    
    // If no methods succeeded, throw an error
    if (candidate_energies.empty()) {
        throw std::runtime_error("Failed to read ground state energy from any available file: " 
                                 "eigenvalues.dat, eigenvalues.txt, or SS_rand0.dat");
    }
    
    // Return the minimum energy across all sources (most robust against corruption)
    auto min_it = std::min_element(candidate_energies.begin(), candidate_energies.end());
    size_t min_idx = std::distance(candidate_energies.begin(), min_it);
    double final_energy = *min_it;
    
    // Report all found energies and which one was selected
    std::cout << "Ground state energy candidates found:" << std::endl;
    for (size_t i = 0; i < candidate_energies.size(); i++) {
        std::cout << "  " << sources[i] << ": " 
                  << std::fixed << std::setprecision(10) << candidate_energies[i];
        if (i == min_idx) {
            std::cout << " ← SELECTED (minimum)";
        }
        std::cout << std::endl;
    }
    
    return final_energy;
}

void printSpinConfiguration(ComplexVector &state, int num_sites, float spin_length, const std::string &dir) {
    // Compute and print <S_i> for all sites
    std::vector<std::vector<Complex>> result(num_sites, std::vector<Complex>(3));

    for (int i = 0; i < num_sites; i++) {
        SingleSiteOperator S_plus(num_sites, spin_length, 0, i);
        SingleSiteOperator S_minus(num_sites, spin_length, 1, i);
        SingleSiteOperator S_z(num_sites, spin_length, 2, i);

        ComplexVector temp_plus(state.size(), Complex(0.0, 0.0));
        ComplexVector temp_minus(state.size(), Complex(0.0, 0.0));
        ComplexVector temp_z(state.size(), Complex(0.0, 0.0));

        S_plus.apply(state.data(), temp_plus.data(), state.size());
        S_minus.apply(state.data(), temp_minus.data(), state.size());
        S_z.apply(state.data(), temp_z.data(), state.size());

        Complex expectation_plus = 0.0;
        Complex expectation_minus = 0.0;
        Complex expectation_z = 0.0;
        for (size_t k = 0; k < state.size(); k++) {
            expectation_plus += std::conj(state[k]) * temp_plus[k];
            expectation_minus += std::conj(state[k]) * temp_minus[k];
            expectation_z += std::conj(state[k]) * temp_z[k];
        }
        result[i][0] = expectation_plus;
        result[i][1] = expectation_minus;
        result[i][2] = expectation_z;
    }
    // Write results to file
    std::ofstream outfile(dir + "/spin_configuration.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "Site S+ S- Sz\n";
    for (int i = 0; i < num_sites; i++) {
        outfile << i << " " << result[i][0] << " " << result[i][1] << " " << result[i][2] << "\n";
    }
    outfile.close();
}

void printSpinCorrelation(ComplexVector &state, int num_sites, float spin_length, const std::string &dir, int unit_cell_size) {
    // Compute and print <S_i . S_j> for all pairs (i,j)
    std::vector<std::vector<std::vector<Complex>>> result(2, std::vector<std::vector<Complex>>(num_sites, std::vector<Complex>(num_sites)));

    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            SingleSiteOperator S_plus_i(num_sites, spin_length, 0, i);
            SingleSiteOperator S_plus_j(num_sites, spin_length, 0, j);
            SingleSiteOperator S_z_i(num_sites, spin_length, 2, i);
            SingleSiteOperator S_z_j(num_sites, spin_length, 2, j);

            ComplexVector temp_plus_i(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_z_i(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_plus_j(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_z_j(state.size(), Complex(0.0, 0.0));

            S_plus_i.apply(state.data(), temp_plus_i.data(), state.size());
            S_z_i.apply(state.data(), temp_z_i.data(), state.size());
            S_plus_j.apply(state.data(), temp_plus_j.data(), state.size());
            S_z_j.apply(state.data(), temp_z_j.data(), state.size());

            Complex expectation_plus = 0.0;
            for (size_t k = 0; k < state.size(); k++) {
                expectation_plus += std::conj(temp_plus_i[k]) * temp_plus_j[k];
            }
            result[0][i][j] = expectation_plus;
            Complex expectation_z = 0.0;
            for (size_t k = 0; k < state.size(); k++) {
                expectation_z += std::conj(temp_z_i[k]) * temp_z_j[k];
            }
            result[1][i][j] = expectation_z;
        }
    }
    // Write results to file
    std::ofstream outfile(dir + "/spin_correlation.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "i j <S+_i S-_j> <Sz_i Sz_j>\n";
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            outfile << i << " " << j << " " << result[0][i][j] << " " << result[1][i][j] << "\n";
        }
    }
    outfile.close();

    // Print sublattice correlations
    std::ofstream subfile(dir + "/sublattice_correlation.txt");
    subfile << std::fixed << std::setprecision(6);
    subfile << "sub_i sub_j <S+_i S-_j>_sum <Sz_i Sz_j>_sum count\n";
    
    // Compute sublattice sums
    std::vector<std::vector<Complex>> sublattice_sums_plus(unit_cell_size, std::vector<Complex>(unit_cell_size, 0.0));
    std::vector<std::vector<Complex>> sublattice_sums_z(unit_cell_size, std::vector<Complex>(unit_cell_size, 0.0));
    std::vector<std::vector<int>> sublattice_counts(unit_cell_size, std::vector<int>(unit_cell_size, 0));
    
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            int sub_i = i % unit_cell_size;
            int sub_j = j % unit_cell_size;
            sublattice_sums_plus[sub_i][sub_j] += result[0][i][j];
            sublattice_sums_z[sub_i][sub_j] += result[1][i][j];
            sublattice_counts[sub_i][sub_j]++;
        }
    }
    
    // Write sublattice results
    for (int sub_i = 0; sub_i < unit_cell_size; sub_i++) {
        for (int sub_j = 0; sub_j < unit_cell_size; sub_j++) {
            subfile << sub_i << " " << sub_j << " " 
                   << sublattice_sums_plus[sub_i][sub_j] << " "
                   << sublattice_sums_z[sub_i][sub_j] << " "
                   << sublattice_counts[sub_i][sub_j] << "\n";
        }
    }
    subfile.close();

    // Print total sums for verification
    std::ofstream sumfile(dir + "/total_sums.txt");
    sumfile << std::fixed << std::setprecision(6);

    Complex total_plus_sum = 0.0;
    Complex total_z_sum = 0.0;
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            total_plus_sum += result[0][i][j];
            total_z_sum += result[1][i][j];
        }
    }

    sumfile << "Total <S+_i S-_j> sum: " << total_plus_sum << "\n";
    sumfile << "Total <Sz_i Sz_j> sum: " << total_z_sum << "\n";
    sumfile.close();

    std::cout << "Total correlation sums:" << std::endl;
    std::cout << "  <S+_i S-_j> sum: " << total_plus_sum << std::endl;
    std::cout << "  <Sz_i Sz_j> sum: " << total_z_sum << std::endl;

    std::cout << "Spin correlation data saved to spin_correlation.txt" << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 4 || argc > 15) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <directory> <krylov_dim_or_nmax> <spin_combinations> [method] [operator_type] [basis] [dt,t_end]/[omega_min,omega_max,omega_bins,broadening] [unit_cell_size] [momentum_points] [polarization] [theta] [use_gpu] [n_up] [T_min,T_max,T_steps]" << std::endl;
            std::cerr << "\nNote: num_sites is automatically detected from positions.dat, spin_length is fixed at 0.5" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "REQUIRED ARGUMENTS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "  directory: Path containing InterAll.dat, Trans.dat, and positions.dat files" << std::endl;
            std::cerr << "  krylov_dim_or_nmax: Dimension of Krylov subspace (krylov/taylor) or Lanczos order (spectral)" << std::endl;
            std::cerr << "  spin_combinations: Format \"op1,op2;op3,op4;...\" where op is:" << std::endl;
            std::cerr << "    - ladder basis: 0=Sp, 1=Sm, 2=Sz" << std::endl;
            std::cerr << "    - xyz basis: 0=Sx, 1=Sy, 2=Sz" << std::endl;
            std::cerr << "    - Example: \"0,1;2,2\" for SpSm/SxSy, SzSz combinations" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "OPTIONAL ARGUMENTS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "  method (default: krylov): krylov | taylor | spectral | spectral_thermal" << std::endl;
            std::cerr << "    - krylov: Time-domain correlation C(t) using Krylov time evolution" << std::endl;
            std::cerr << "    - taylor: Time-domain correlation C(t) using Taylor expansion" << std::endl;
            std::cerr << "    - spectral: Frequency-domain spectral function S(ω) via FTLM (single state)" << std::endl;
            std::cerr << "    - spectral_thermal: Frequency-domain S(ω) with thermal averaging (FTLM random sampling)" << std::endl;
            std::cerr << "\n  operator_type (default: sum): sum | transverse | sublattice | experimental | transverse_experimental" << std::endl;
            std::cerr << "    - sum: Standard momentum-resolved sum operators S^{op1}(Q) S^{op2}(-Q)" << std::endl;
            std::cerr << "    - transverse: Polarization-dependent operators for magnetic scattering" << std::endl;
            std::cerr << "    - sublattice: Sublattice-resolved correlations (use unit_cell_size)" << std::endl;
            std::cerr << "    - experimental: General form cos(θ)Sz + sin(θ)Sx (use theta)" << std::endl;
            std::cerr << "    - transverse_experimental: Transverse version with experimental angle" << std::endl;
            std::cerr << "\n  basis (default: ladder): ladder | xyz" << std::endl;
            std::cerr << "    - ladder: Use Sp/Sm/Sz operators (raising/lowering operators)" << std::endl;
            std::cerr << "    - xyz: Use Sx/Sy/Sz operators (Cartesian components)" << std::endl;
            std::cerr << "    - Note: experimental operator type always uses xyz basis internally" << std::endl;
            std::cerr << "\n  dt,t_end (format depends on method):" << std::endl;
            std::cerr << "    - For krylov/taylor: \"dt,t_end\" e.g., \"0.01,50.0\"" << std::endl;
            std::cerr << "      * dt: time step for evolution" << std::endl;
            std::cerr << "      * t_end: maximum evolution time" << std::endl;
            std::cerr << "    - For spectral: \"omega_min,omega_max,num_omega_bins,broadening\" e.g., \"-5.0,5.0,200,0.1\"" << std::endl;
            std::cerr << "      * omega_min: minimum frequency" << std::endl;
            std::cerr << "      * omega_max: maximum frequency" << std::endl;
            std::cerr << "      * num_omega_bins: number of frequency points (resolution)" << std::endl;
            std::cerr << "      * broadening: Lorentzian broadening parameter (eta)" << std::endl;
            std::cerr << "\n  unit_cell_size (for sublattice operators): number of sublattices (default: 4)" << std::endl;
            std::cerr << "  momentum_points (default: (0,0,0);(0,0,2π)): \"Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;...\"" << std::endl;
            std::cerr << "  polarization (for transverse operators): \"px,py,pz\" normalized vector (default: (1/√2,-1/√2,0))" << std::endl;
            std::cerr << "  theta (for experimental operators): angle in radians (default: 0.0)" << std::endl;
            std::cerr << "  use_gpu (optional): 1 for GPU acceleration, 0 for CPU only (default: 0)" << std::endl;
            std::cerr << "  n_up (optional): number of up spins for fixed-Sz sector (default: -1 = use full Hilbert space)" << std::endl;
            std::cerr << "    - When n_up >= 0: restricts to fixed total Sz = n_up - n_down = n_up - (num_sites - n_up)" << std::endl;
            std::cerr << "    - Reduces Hilbert space dimension from 2^N to C(N, n_up)" << std::endl;
            std::cerr << "    - Example: for 16 sites with n_up=8, dimension reduces from 65536 to 12870" << std::endl;
            std::cerr << "  T_min,T_max,T_steps (for spectral_thermal only): Temperature scan parameters" << std::endl;
            std::cerr << "    - Format: \"T_min,T_max,T_steps\" e.g., \"0.1,10.0,10\"" << std::endl;
            std::cerr << "    - T_min: Minimum temperature" << std::endl;
            std::cerr << "    - T_max: Maximum temperature" << std::endl;
            std::cerr << "    - T_steps: Number of temperature points (logarithmic spacing)" << std::endl;
            std::cerr << "    - If not specified, uses temperature from TPQ state (T = 1/β)" << std::endl;
            std::cerr << "    - When specified, computes S(ω,T) for each temperature in the range" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "SPECTRAL METHOD (method=spectral or method=spectral_thermal) DETAILS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "The spectral methods compute the dynamical structure factor (DSF):" << std::endl;
            std::cerr << "  S(q,ω) = (1/π) × Im[⟨ψ|O₁†(q) G(ω) O₂(q)|ψ⟩]" << std::endl;
            std::cerr << "where G(ω) = 1/(ω - H + iη) is the Green's function." << std::endl;
            std::cerr << "\nTwo variants available:" << std::endl;
            std::cerr << "  1. spectral: Single-state calculation (for T=0 or specific TPQ states)" << std::endl;
            std::cerr << "     - Uses the given state |ψ⟩ directly" << std::endl;
            std::cerr << "     - Fastest option for ground state calculations" << std::endl;
            std::cerr << "  2. spectral_thermal: Finite-temperature with thermal averaging" << std::endl;
            std::cerr << "     - Uses FTLM with random sampling for thermal ensemble averaging" << std::endl;
            std::cerr << "     - Computes S(q,ω,T) = Tr[ρ(T) O₁†(q) δ(ω-H) O₂(q)]" << std::endl;
            std::cerr << "     - Number of samples: 40 (configurable in code)" << std::endl;
            std::cerr << "     - Temperature: Can scan over T range or use TPQ state temperature" << std::endl;
            std::cerr << "     - More accurate for finite temperature but slower" << std::endl;
            std::cerr << "\nParameters:" << std::endl;
            std::cerr << "  krylov_dim_or_nmax: Lanczos order (typical values: 30-100)" << std::endl;
            std::cerr << "    - Higher values = better convergence but more computational cost" << std::endl;
            std::cerr << "    - Recommended: 50-100 for most systems" << std::endl;
            std::cerr << "\n  Broadening (eta) in spectral parameters:" << std::endl;
            std::cerr << "    - Controls smoothing of spectral features (peak width)" << std::endl;
            std::cerr << "    - Small values (0.01-0.05): High resolution, shows sharp features" << std::endl;
            std::cerr << "    - Medium values (0.1): Balanced (default), good for most cases" << std::endl;
            std::cerr << "    - Large values (0.2-0.5): Smoothed, reduces noise artifacts" << std::endl;
            std::cerr << "\n  Frequency resolution:" << std::endl;
            std::cerr << "    - num_omega_bins: Number of frequency points" << std::endl;
            std::cerr << "    - Larger values give finer frequency resolution" << std::endl;
            std::cerr << "    - Typical values: 100-500 depending on energy scale" << std::endl;
            std::cerr << "\nOutput files:" << std::endl;
            std::cerr << "  - spectral method: <operator>_spectral_sample_<idx>_beta_<beta>.txt" << std::endl;
            std::cerr << "  - spectral_thermal method: <operator>_spectral_thermal_sample_<idx>_beta_<beta>_T_<T>_nsamples_<N>.txt" << std::endl;
            std::cerr << "  - Columns: frequency(ω) | spectral_intensity S(ω) | error (if applicable)" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "TIME-DOMAIN METHODS (method=krylov or method=taylor) DETAILS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "Compute time correlation function C(t) = ⟨ψ|O₁†(0) O₂(t)|ψ⟩" << std::endl;
            std::cerr << "\nKrylov method:" << std::endl;
            std::cerr << "  - Most accurate and stable approach" << std::endl;
            std::cerr << "  - Uses Lanczos algorithm for exponential evolution" << std::endl;
            std::cerr << "  - Parameter: krylov_dim = subspace dimension (typical: 20-50)" << std::endl;
            std::cerr << "\nTaylor method:" << std::endl;
            std::cerr << "  - Fast but less stable for long times" << std::endl;
            std::cerr << "  - Uses Taylor series: exp(-iHt) ≈ Σ (-iHt)ⁿ/n!" << std::endl;
            std::cerr << "  - Parameter: krylov_dim_or_nmax = max order of Taylor expansion" << std::endl;
            std::cerr << "\nOutput files (time-domain methods):" << std::endl;
            std::cerr << "  - Format: <operator>_<method>_sample_<idx>_beta_<beta>.txt" << std::endl;
            std::cerr << "  - Columns: step_index | time_value | Re[C(t)] | Im[C(t)]" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "EXAMPLES:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "1. Spectral function with momentum resolution (single state):" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 50 \"2,2\" spectral sum ladder \"-5,5,200,0.1\" 4 \"0,0,0;0,0,1\"" << std::endl;
            std::cerr << "\n2. Thermal spectral function with FTLM averaging:" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 50 \"2,2\" spectral_thermal sum ladder \"-5,5,200,0.1\" 4 \"0,0,0;0,0,1\"" << std::endl;
            std::cerr << "\n3. Time-domain using Krylov (recommended):" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 30 \"0,1;2,2\" krylov sum ladder \"0.01,50.0\"" << std::endl;
            std::cerr << "\n4. Transverse scattering with custom polarization:" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 40 \"2,2\" spectral transverse xyz \"-10,10,300,0.2\" 4 \"0,0,0\" \"1,0,0\"" << std::endl;
            std::cerr << "\n5. Experimental geometry with angle:" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 50 \"2,2\" spectral experimental xyz \"-5,5,250,0.1\" 4 \"0,0,0\" \"0,0,0\" 0.785" << std::endl;
            std::cerr << "\n6. Fixed-Sz sector calculation (Sz = 0 for 16 sites):" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 50 \"2,2\" spectral sum ladder \"-5,5,200,0.1\" 4 \"0,0,0\" \"0,0,0\" 0 0 8" << std::endl;
            std::cerr << "\n7. Thermal spectral with fixed-Sz:" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 50 \"2,2\" spectral_thermal sum ladder \"-5,5,200,0.1\" 4 \"0,0,0\" \"0,0,0\" 0 0 8" << std::endl;
            std::cerr << "\n8. Temperature scan with spectral_thermal:" << std::endl;
            std::cerr << "   " << argv[0] << " ./data 50 \"2,2\" spectral_thermal sum ladder \"-5,5,200,0.1\" 4 \"0,0,0\" \"0,0,0\" 0 0 -1 \"0.1,10.0,10\"" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string directory = argv[1];
    int krylov_dim_or_nmax = std::stoi(argv[2]);
    std::string spin_combinations_str = argv[3];
    std::string method = (argc >= 5) ? std::string(argv[4]) : std::string("krylov");
    std::string operator_type = (argc >= 6) ? std::string(argv[5]) : std::string("sum");
    std::string basis = (argc >= 7) ? std::string(argv[6]) : std::string("ladder");
    
    // Read num_sites from positions.dat and set spin_length to 0.5
    std::string positions_file = directory + "/positions.dat";
    int num_sites;
    try {
        num_sites = read_num_sites_from_positions(positions_file);
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    float spin_length = 0.5f;
    
    // Experimental operators always use XYZ basis internally
    if ((operator_type == "experimental" || operator_type == "transverse_experimental") && basis != "xyz") {
        if (rank == 0) {
            std::cout << "Note: " << operator_type << " operator type requires xyz basis, setting basis=xyz" << std::endl;
        }
        basis = "xyz";
    }
    
    double dt_opt = 0.01;
    double t_end_opt = 50.0;
    double omega_min = -5.0;
    double omega_max = 5.0;
    int num_omega_bins = 200;
    double broadening = 0.1;
    
    if (argc >= 8) {
        std::string param_str = argv[7];
        
        if (method == "spectral") {
            // Parse omega_min,omega_max,num_omega_bins,broadening for spectral method
            std::stringstream ss(param_str);
            std::string val;
            std::vector<std::string> tokens;
            while (std::getline(ss, val, ',')) {
                tokens.push_back(val);
            }
            
            try {
                if (tokens.size() >= 1) omega_min = std::stod(tokens[0]);
                if (tokens.size() >= 2) omega_max = std::stod(tokens[1]);
                if (tokens.size() >= 3) num_omega_bins = std::stoi(tokens[2]);
                if (tokens.size() >= 4) broadening = std::stod(tokens[3]);
                if (rank == 0) {
                    std::cout << "Spectral method parameters: omega=[" << omega_min << "," << omega_max 
                              << "], bins=" << num_omega_bins << ", broadening=" << broadening << std::endl;
                }
            } catch (...) {
                if (rank == 0) {
                    std::cerr << "Warning: failed to parse spectral parameters. Using defaults." << std::endl;
                }
            }
        } else {
            // Parse dt,t_end for time-domain methods
            auto comma_pos = param_str.find(',');
            if (comma_pos != std::string::npos) {
                try {
                    dt_opt = std::stod(param_str.substr(0, comma_pos));
                    t_end_opt = std::stod(param_str.substr(comma_pos + 1));
                } catch (...) {
                    if (rank == 0) {
                        std::cerr << "Warning: failed to parse dt,t_end argument. Using defaults 0.01,50.0" << std::endl;
                    }
                }
            }
        }
    }
    
    int unit_cell_size = 4; // Default for pyrochlore
    if (argc >= 9) {
        try { unit_cell_size = std::stoi(argv[8]); } catch (...) { unit_cell_size = 4; }
    }

    // Parse momentum points
    std::vector<std::vector<double>> momentum_points;
    if (argc >= 10) {
        std::string momentum_str = argv[9];
        std::stringstream mom_ss(momentum_str);
        std::string point_str;
        
        while (std::getline(mom_ss, point_str, ';')) {
            std::stringstream point_ss(point_str);
            std::string coord_str;
            std::vector<double> point;
            
            while (std::getline(point_ss, coord_str, ',')) {
                try {
                    double coord = std::stod(coord_str);
                    coord *= M_PI;  // Scale to π
                    point.push_back(coord);
                } catch (...) {
                    if (rank == 0) {
                        std::cerr << "Warning: Failed to parse momentum coordinate: " << coord_str << std::endl;
                    }
                }
            }
            
            if (point.size() == 3) {
                momentum_points.push_back(point);
            } else if (rank == 0) {
                std::cerr << "Warning: Momentum point must have 3 coordinates, got " << point.size() << std::endl;
            }
        }
    }
    
    // Use default momentum points if none provided or parsing failed
    if (momentum_points.empty()) {
        momentum_points = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 2.0 * M_PI}
        };
        if (rank == 0) {
            std::cout << "Using default momentum points: (0,0,0) and (0,0,2π)" << std::endl;
        }
    }

    // Parse polarization vector for transverse operators
    std::vector<double> polarization = {1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0};
    if (argc >= 11) {
        std::string pol_str = argv[10];
        std::stringstream pol_ss(pol_str);
        std::string coord_str;
        std::vector<double> pol_temp;
        
        while (std::getline(pol_ss, coord_str, ',')) {
            try {
                double coord = std::stod(coord_str);
                pol_temp.push_back(coord);
            } catch (...) {
                if (rank == 0) {
                    std::cerr << "Warning: Failed to parse polarization coordinate: " << coord_str << std::endl;
                }
            }
        }
        
        if (pol_temp.size() == 3) {
            // Normalize the polarization vector
            double norm = std::sqrt(pol_temp[0]*pol_temp[0] + pol_temp[1]*pol_temp[1] + pol_temp[2]*pol_temp[2]);
            if (norm > 1e-10) {
                polarization = {pol_temp[0]/norm, pol_temp[1]/norm, pol_temp[2]/norm};
                if (rank == 0) {
                    std::cout << "Using custom polarization: (" << polarization[0] << "," 
                              << polarization[1] << "," << polarization[2] << ")" << std::endl;
                }
            } else if (rank == 0) {
                std::cerr << "Warning: Polarization vector has zero norm, using default" << std::endl;
            }
        } else if (rank == 0) {
            std::cerr << "Warning: Polarization must have 3 coordinates, got " << pol_temp.size() << std::endl;
        }
    }

    // Parse theta for experimental operators
    double theta = 0.0;  // Default to 0
    if (argc >= 12) {
        try {
            theta = std::stod(argv[11]);
            theta *= M_PI;
            if (rank == 0) {
                std::cout << "Using theta = " << theta << " radians" << std::endl;
            }
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to parse theta, using default 0.0" << std::endl;
            }
            theta = 0.0;
        }
    }

    // Parse GPU flag
    bool use_gpu = false;
    if (argc >= 13) {
        try {
            int gpu_flag = std::stoi(argv[12]);
            use_gpu = (gpu_flag != 0);
#ifndef ENABLE_GPU
            if (use_gpu && rank == 0) {
                std::cerr << "Warning: GPU requested but code not compiled with GPU support. Using CPU." << std::endl;
                use_gpu = false;
            }
#endif
            if (rank == 0 && use_gpu) {
                std::cout << "GPU acceleration enabled for time evolution" << std::endl;
            }
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to parse GPU flag, using CPU" << std::endl;
            }
            use_gpu = false;
        }
    }

    // Parse n_up for fixed-Sz sector (optional)
    int n_up = -1;  // Default: -1 means use full Hilbert space
    bool use_fixed_sz = false;
    if (argc >= 14) {
        try {
            n_up = std::stoi(argv[13]);
            if (n_up == -1) {
                // Explicitly using full Hilbert space
                use_fixed_sz = false;
            } else if (n_up >= 0 && n_up <= num_sites) {
                use_fixed_sz = true;
                if (rank == 0) {
                    std::cout << "Using fixed-Sz sector: n_up = " << n_up 
                              << ", Sz = " << (n_up - (num_sites - n_up)) * 0.5 << std::endl;
                    // Calculate and display dimension
                    size_t fixed_sz_dim = 1;
                    for (int i = 0; i < n_up; ++i) {
                        fixed_sz_dim = fixed_sz_dim * (num_sites - i) / (i + 1);
                    }
                    std::cout << "Fixed-Sz dimension: " << fixed_sz_dim 
                              << " (reduced from " << (1ULL << num_sites) << ")" << std::endl;
                }
            } else {
                if (rank == 0) {
                    std::cerr << "Warning: Invalid n_up value " << n_up 
                              << " (must be -1 or 0 <= n_up <= " << num_sites << "), using full Hilbert space" << std::endl;
                }
                n_up = -1;
                use_fixed_sz = false;
            }
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to parse n_up, using full Hilbert space" << std::endl;
            }
            n_up = -1;
            use_fixed_sz = false;
        }
    }

    // Parse temperature scan parameters for spectral_thermal (optional)
    double T_min = 1e-3;  // Negative means use TPQ state temperature
    double T_max = 1.0;
    int T_steps = 20;
    bool use_temperature_scan = true;
    
    if (argc >= 15) {
        std::string temp_str = argv[14];
        std::stringstream temp_ss(temp_str);
        std::string val;
        std::vector<std::string> tokens;
        
        while (std::getline(temp_ss, val, ',')) {
            tokens.push_back(val);
        }
        
        try {
            if (tokens.size() >= 3) {
                T_min = std::stod(tokens[0]);
                T_max = std::stod(tokens[1]);
                T_steps = std::stoi(tokens[2]);
                
                if (T_min > 0 && T_max > T_min && T_steps > 0) {
                    use_temperature_scan = true;
                    if (rank == 0) {
                        std::cout << "Temperature scan enabled: T ∈ [" << T_min << ", " << T_max 
                                  << "], " << T_steps << " points (log scale)" << std::endl;
                    }
                } else {
                    if (rank == 0) {
                        std::cerr << "Warning: Invalid temperature parameters (need T_min > 0, T_max > T_min, T_steps > 0)" << std::endl;
                        std::cerr << "Using TPQ state temperature instead" << std::endl;
                    }
                    use_temperature_scan = false;
                }
            } else {
                if (rank == 0) {
                    std::cerr << "Warning: Temperature parameter needs 3 values (T_min,T_max,T_steps)" << std::endl;
                    std::cerr << "Using TPQ state temperature instead" << std::endl;
                }
            }
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to parse temperature parameters" << std::endl;
                std::cerr << "Using TPQ state temperature instead" << std::endl;
            }
            use_temperature_scan = false;
        }
    }

    // Parse spin combinations
    std::vector<std::pair<int, int>> spin_combinations;
    std::stringstream ss(spin_combinations_str);
    std::string pair_str;
    
    while (std::getline(ss, pair_str, ';')) {
        std::stringstream pair_ss(pair_str);
        std::string op1_str, op2_str;
        
        if (std::getline(pair_ss, op1_str, ',') && std::getline(pair_ss, op2_str)) {
            try {
                int op1 = std::stoi(op1_str);
                int op2 = std::stoi(op2_str);
                
                if (op1 >= 0 && op1 <= 2 && op2 >= 0 && op2 <= 2) {
                    spin_combinations.push_back({op1, op2});
                } else {
                    if (rank == 0) {
                        std::cerr << "Warning: Invalid spin operator " << op1 << "," << op2 
                                  << ". Operators must be 0, 1, or 2." << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                if (rank == 0) {
                    std::cerr << "Warning: Failed to parse spin combination: " << pair_str << std::endl;
                }
            }
        }
    }
    
    if (spin_combinations.empty()) {
        if (rank == 0) {
            std::cerr << "Error: No valid spin combinations provided. Using default SzSz." << std::endl;
        }
        spin_combinations = {{2, 2}};
    }

    // Determine if using XYZ basis (Sx, Sy, Sz) vs ladder basis (Sp, Sm, Sz)
    bool use_xyz_basis = (basis == "xyz");

    auto spin_combination_name = [use_xyz_basis](int op) {
        if (use_xyz_basis) {
            // XYZ basis: Sx, Sy, Sz
            switch (op) {
                case 0:
                    return "Sx";
                case 1:
                    return "Sy";
                case 2:
                    return "Sz";
                default:
                    return "Unknown";
            }
        } else {
            // Ladder basis: Sp, Sm, Sz
            switch (op) {
                case 2:
                    return "Sz";
                case 0:
                    return "Sp";
                case 1:
                    return "Sm";
                default:
                    return "Unknown";
            }
        }
    };

    std::vector<const char*> spin_combination_names;
    for (const auto& pair : spin_combinations) {
        int first = pair.first;
        int second = pair.second;
        
        if (!use_xyz_basis) {
            // For ladder basis: Convert 0->1(Sp), 1->0(Sm) for first operator
            first = first == 2 ? 2 : 1 - first;
        }
        // For XYZ basis, use operators as-is (0=Sx, 1=Sy, 2=Sz)
        std::string combined_name = std::string(spin_combination_name(first)) + std::string(spin_combination_name(second));
        char* name = new char[combined_name.size() + 1];
        std::strcpy(name, combined_name.c_str());
        spin_combination_names.push_back(name);
    }

    // Regex to match tpq_state files - support both new format (with step) and legacy format
    // New format: tpq_state_i_beta=*_step=*.dat
    // Legacy format: tpq_state_i_beta=*.dat
    std::regex state_pattern_new("tpq_state_([0-9]+)_beta=([0-9.]+)_step=([0-9]+)\\.dat");
    std::regex state_pattern_legacy("tpq_state_([0-9]+)_beta=([0-9.]+)\\.dat");

    // Load Hamiltonian (all processes need this)
    if (rank == 0) {
        std::cout << "Loading Hamiltonian..." << std::endl;
    }
    
    Operator ham_op(num_sites, spin_length);
    
    std::string interall_file = directory + "/InterAll.dat";
    std::string trans_file = directory + "/Trans.dat";
    std::string counterterm_file = directory + "/CounterTerm.dat";
    std::string three_body_file = directory + "/ThreeBodyG.dat";
    // Check if files exist
    if (!fs::exists(interall_file) || !fs::exists(trans_file)) {
        if (rank == 0) {
            std::cerr << "Error: Hamiltonian files not found in " << directory << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Load Hamiltonian
    ham_op.loadFromInterAllFile(interall_file);
    ham_op.loadFromFile(trans_file);
    if (fs::exists(three_body_file)) {
        if (rank == 0) {
            std::cout << "Loading three-body interactions from " << three_body_file << std::endl;
        }
        ham_op.loadThreeBodyTerm(three_body_file);
    }
    
    // COUNTERTERM DISABLED
    // if (fs::exists(counterterm_file)) {
    //     ham_op.loadCounterTerm(counterterm_file);
    // }
    
    auto H = [&ham_op](const Complex* in, Complex* out, int size) {
        ham_op.apply(in, out, size);
    };
    
    // Read ground state energy for energy shift in spectral functions
    double ground_state_energy = 0.0;
    bool has_ground_state_energy = false;
    if (method == "spectral") {
        try {
            if (rank == 0) {
                std::cout << "Reading ground state energy (minimum across all sources)..." << std::endl;
            }
            ground_state_energy = read_ground_state_energy(directory);
            has_ground_state_energy = true;
            if (rank == 0) {
                std::cout << "Final ground state energy (for spectral shift): " 
                          << std::fixed << std::setprecision(10) << ground_state_energy << std::endl;
            }
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "Warning: Could not read ground state energy: " << e.what() << std::endl;
                std::cerr << "Proceeding without energy shift" << std::endl;
            }
        }
    }
    
    // Use 64-bit to compute Hilbert space dimension and guard against int overflow
    size_t N64 = 1ULL << num_sites;
    if (N64 > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::cerr << "Error: 2^num_sites exceeds 32-bit int range (num_sites=" << num_sites
                      << "). Refactor APIs to use size_t for N or reduce num_sites." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    int N = static_cast<int>(N64);
    
    // Helper function for cross product
    auto cross_product = [](const std::vector<double>& a, const std::vector<double>& b) -> std::array<double, 3> {
        return {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    };
    
    // Helper function to normalize a vector
    auto normalize = [](const std::array<double, 3>& v) -> std::array<double, 3> {
        double norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (norm < 1e-10) {
            return {0.0, 0.0, 0.0};
        }
        return {v[0]/norm, v[1]/norm, v[2]/norm};
    };
    
    if (rank == 0) {
        std::cout << "\nMomentum points:" << std::endl;
        for (size_t i = 0; i < momentum_points.size(); i++) {
            std::cout << "  Q[" << i << "] = (" << momentum_points[i][0] << ", " 
                      << momentum_points[i][1] << ", " << momentum_points[i][2] << ")" << std::endl;
        }
        std::cout << "\nPolarization vector (transverse_basis_1): (" 
                  << polarization[0] << ", " << polarization[1] << ", " 
                  << polarization[2] << ")" << std::endl;
    }
    
    // Pre-compute time evolution operator and transverse bases if needed
    std::string output_base_dir = directory + "/structure_factor_results";
    if (rank == 0) {
        ensureDirectoryExists(output_base_dir);
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Pre-compute time evolution operator if using taylor method
    std::function<void(const Complex*, Complex*, int)> U_t;
    
    if (method == "taylor") {
        if (rank == 0) {
            std::cout << "Pre-computing time evolution operator (n_max=" << krylov_dim_or_nmax
                      << ", dt=" << dt_opt << ", t_end=" << t_end_opt << ")" << std::endl;
        }
        U_t = create_time_evolution_operator(H, dt_opt, krylov_dim_or_nmax, true);
    }
    
    // Pre-compute transverse bases for transverse operators (needed for both krylov and taylor methods)
    std::vector<std::array<double,3>> transverse_basis_1, transverse_basis_2;
    
    if (operator_type == "transverse" || operator_type == "transverse_experimental") {
        int num_momentum = momentum_points.size();
        transverse_basis_1.resize(num_momentum);
        transverse_basis_2.resize(num_momentum);
        
        // transverse_basis_1 is the same for all momentum points (the polarization vector)
        std::array<double, 3> pol_array = {polarization[0], polarization[1], polarization[2]};
        
        for (int qi = 0; qi < num_momentum; ++qi) {
            transverse_basis_1[qi] = pol_array;
            
            // transverse_basis_2 is Q × polarization (cross product)
            auto cross = cross_product(momentum_points[qi], polarization);
            transverse_basis_2[qi] = normalize(cross);
            
            // Handle special case: if Q is parallel to polarization, cross product is zero
            double cross_norm = std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
            if (cross_norm < 1e-10) {
                // Find an orthogonal vector to polarization
                if (std::abs(pol_array[0]) > 0.5) {
                    // polarization has significant x component, use y-axis as reference
                    auto alt_cross = cross_product({0.0, 1.0, 0.0}, polarization);
                    transverse_basis_2[qi] = normalize(alt_cross);
                } else {
                    // use x-axis as reference
                    auto alt_cross = cross_product({1.0, 0.0, 0.0}, polarization);
                    transverse_basis_2[qi] = normalize(alt_cross);
                }
                if (rank == 0) {
                    std::cout << "Warning: Q[" << qi << "] parallel to polarization, using alternative basis" << std::endl;
                }
            }
        }
        
        if (rank == 0) {
            std::cout << "\nTransverse bases for momentum points:" << std::endl;
            for (int qi = 0; qi < num_momentum; ++qi) {
                const auto &Q = momentum_points[qi];
                const auto &b1 = transverse_basis_1[qi];
                const auto &b2 = transverse_basis_2[qi];
                std::cout << "  Q[" << qi << "] = (" << Q[0] << "," << Q[1] << "," << Q[2] 
                          << "), e1=(" << b1[0] << "," << b1[1] << "," << b1[2] 
                          << "), e2=(" << b2[0] << "," << b2[1] << "," << b2[2] << ")" << std::endl;
            }
        }
    }
    
    // Collect all tpq_state files from the output subdirectory (only rank 0)
    std::vector<std::string> tpq_files;
    std::vector<int> sample_indices;
    std::vector<double> beta_values;
    std::vector<std::string> beta_strings;
    
    if (rank == 0) {
        std::string tpq_directory = directory + "/output";
        for (const auto& entry : fs::directory_iterator(tpq_directory)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            std::smatch match;
            
            // Try new format first (with step)
            if (std::regex_match(filename, match, state_pattern_new)) {
                tpq_files.push_back(entry.path().string());
                sample_indices.push_back(std::stoi(match[1]));
                beta_strings.push_back(match[2]);
                beta_values.push_back(std::stod(match[2]));
            }
            // Fall back to legacy format (without step)
            else if (std::regex_match(filename, match, state_pattern_legacy)) {
                tpq_files.push_back(entry.path().string());
                sample_indices.push_back(std::stoi(match[1]));
                beta_strings.push_back(match[2]);
                beta_values.push_back(std::stod(match[2]));
            }
        }

        // Optionally include zero-temperature ground-state eigenvector
        const std::string gs_file = tpq_directory + "/eigenvectors/eigenvector_0.dat";
        if (fs::exists(gs_file)) {
            tpq_files.push_back(gs_file);
            sample_indices.push_back(0); // use 0 as a conventional index for ground state
            beta_strings.push_back("inf");
            beta_values.push_back(std::numeric_limits<double>::infinity());
        }
        
        std::cout << "Found " << tpq_files.size() << " state file(s) to process (including ground state if present)" << std::endl;
        std::cout << "Using " << size << " MPI processes" << std::endl;
    }
    
    // Broadcast the number of files to all processes
    int num_files = tpq_files.size();
    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (num_files == 0) {
        if (rank == 0) {
            std::cout << "No TPQ state files found." << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    
    // Get file sizes for workload estimation (only rank 0)
    std::vector<size_t> file_sizes(num_files, 0);
    if (rank == 0) {
        for (int i = 0; i < num_files; i++) {
            try {
                file_sizes[i] = fs::file_size(tpq_files[i]);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not get size of " << tpq_files[i] << ": " << e.what() << std::endl;
                file_sizes[i] = 0;
            }
        }
    }
    MPI_Bcast(file_sizes.data(), num_files, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    // Build fine-grained task list following (operators) × (method) structure
    // Each task is (state_idx, momentum_idx, combo_idx, sublattice_i, sublattice_j)
    struct Task {
        int state_idx;
        int momentum_idx;
        int combo_idx;
        int sublattice_i;
        int sublattice_j;
        size_t weight;  // file_size as proxy for cost
    };
    
    std::vector<Task> all_tasks;
    int num_momentum = momentum_points.size();
    int num_combos = spin_combinations.size();
    
    if (rank == 0) {
        // Special handling for spin_correlation, spin_configuration (process entire states atomically)
        if (method == "spin_correlation" || method == "spin_configuration") {
            // These methods process entire states atomically
            for (int s = 0; s < num_files; s++) {
                all_tasks.push_back({s, -1, -1, -1, -1, file_sizes[s]});
            }
            std::cout << "Parallelization: per-state (" << num_files << " tasks)" << std::endl;
        } else {
            // For krylov and taylor methods: parallelize based on operator type
            if (operator_type == "sublattice") {
                // Sublattice operators: parallelize across (state, momentum, combo, sublattice pairs)
                // Only compute upper triangle: sublattice_i <= sublattice_j (symmetry)
                int num_sublattice_pairs = unit_cell_size * (unit_cell_size + 1) / 2;
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        for (int c = 0; c < num_combos; c++) {
                            for (int sub_i = 0; sub_i < unit_cell_size; sub_i++) {
                                for (int sub_j = sub_i; sub_j < unit_cell_size; sub_j++) {
                                    size_t task_weight = file_sizes[s] / (num_momentum * num_combos * num_sublattice_pairs);
                                    all_tasks.push_back({s, q, c, sub_i, sub_j, task_weight});
                                }
                            }
                        }
                    }
                }
                std::cout << "Parallelization: per-sublattice-pair (upper triangle, " << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta × "
                          << num_combos << " combos × " << num_sublattice_pairs << " unique sublattice pairs)" << std::endl;
            } else if (operator_type == "transverse") {
                // Transverse operators: create 2 tasks per (state, momentum, combo) for SF/NSF
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        for (int c = 0; c < num_combos; c++) {
                            // Use sublattice_i as a flag: 0=SF, 1=NSF
                            size_t task_weight = file_sizes[s] / (num_momentum * num_combos * 2);
                            all_tasks.push_back({s, q, c, 0, -1, task_weight}); // SF component
                            all_tasks.push_back({s, q, c, 1, -1, task_weight}); // NSF component
                        }
                    }
                }
                std::cout << "Parallelization: per-transverse-component (" << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta × "
                          << num_combos << " combos × 2 components)" << std::endl;
            } else if (operator_type == "transverse_experimental") {
                // Transverse experimental operators: create 2 tasks per (state, momentum) for SF/NSF
                // Does NOT depend on combo (only uses theta parameter)
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        // Use sublattice_i as a flag: 0=SF, 1=NSF
                        // Set combo_idx to 0 (dummy value, not used)
                        size_t task_weight = file_sizes[s] / (num_momentum * 2);
                        all_tasks.push_back({s, q, 0, 0, -1, task_weight}); // SF component
                        all_tasks.push_back({s, q, 0, 1, -1, task_weight}); // NSF component
                    }
                }
                std::cout << "Parallelization: per-transverse-component (" << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta × 2 components)" << std::endl;
            } else if (operator_type == "experimental") {
                // Experimental operators: parallelize across (state, momentum) only
                // Does NOT depend on combo (only uses theta parameter)
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        // Set combo_idx to 0 (dummy value, not used)
                        size_t task_weight = file_sizes[s] / num_momentum;
                        all_tasks.push_back({s, q, 0, -1, -1, task_weight});
                    }
                }
                std::cout << "Parallelization: per-operator (" << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta)" << std::endl;
            } else {
                // Sum operators: parallelize across (state, momentum, combo)
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        for (int c = 0; c < num_combos; c++) {
                            size_t task_weight = file_sizes[s] / (num_momentum * num_combos);
                            all_tasks.push_back({s, q, c, -1, -1, task_weight});
                        }
                    }
                }
                std::cout << "Parallelization: per-operator (" << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta × "
                          << num_combos << " combos)" << std::endl;
            }
        }
        
        // Sort by weight (descending) for better load balance
        std::sort(all_tasks.begin(), all_tasks.end(), 
                  [](const Task& a, const Task& b) { return a.weight > b.weight; });
    }
    
    // Broadcast task list
    int num_tasks = all_tasks.size();
    MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        all_tasks.resize(num_tasks);
    }
    
    // Broadcast all tasks (struct is POD-like)
    for (int i = 0; i < num_tasks; i++) {
        int buf[5] = {all_tasks[i].state_idx, all_tasks[i].momentum_idx, all_tasks[i].combo_idx, 
                      all_tasks[i].sublattice_i, all_tasks[i].sublattice_j};
        size_t w = all_tasks[i].weight;
        MPI_Bcast(buf, 5, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            all_tasks[i] = {buf[0], buf[1], buf[2], buf[3], buf[4], w};
        }
    }
    
    // Resize vectors on non-root processes
    if (rank != 0) {
        tpq_files.resize(num_files);
        sample_indices.resize(num_files);
        beta_values.resize(num_files);
        beta_strings.resize(num_files);
    }
    
    // Optimized broadcast: use single buffer for all strings
    if (rank == 0) {
        // Pack all string data into a single buffer
        std::vector<char> string_buffer;
        std::vector<int> offsets;
        std::vector<int> lengths;
        
        for (int i = 0; i < num_files; i++) {
            offsets.push_back(string_buffer.size());
            lengths.push_back(tpq_files[i].size());
            string_buffer.insert(string_buffer.end(), tpq_files[i].begin(), tpq_files[i].end());
            
            offsets.push_back(string_buffer.size());
            lengths.push_back(beta_strings[i].size());
            string_buffer.insert(string_buffer.end(), beta_strings[i].begin(), beta_strings[i].end());
        }
        
        int buffer_size = string_buffer.size();
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(string_buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(offsets.data(), offsets.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lengths.data(), lengths.size(), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int buffer_size;
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<char> string_buffer(buffer_size);
        std::vector<int> offsets(num_files * 2);
        std::vector<int> lengths(num_files * 2);
        
        MPI_Bcast(string_buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(offsets.data(), offsets.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lengths.data(), lengths.size(), MPI_INT, 0, MPI_COMM_WORLD);
        
        // Unpack strings
        for (int i = 0; i < num_files; i++) {
            int file_offset = offsets[i * 2];
            int file_length = lengths[i * 2];
            tpq_files[i].assign(string_buffer.begin() + file_offset, 
                               string_buffer.begin() + file_offset + file_length);
            
            int beta_offset = offsets[i * 2 + 1];
            int beta_length = lengths[i * 2 + 1];
            beta_strings[i].assign(string_buffer.begin() + beta_offset, 
                                  string_buffer.begin() + beta_offset + beta_length);
        }
    }
    
    MPI_Bcast(sample_indices.data(), num_files, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(beta_values.data(), num_files, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Lambda to process a single task following (operators) × (method) structure
    auto process_task = [&](const Task& task) -> bool {
        int state_idx = task.state_idx;
        int momentum_idx = task.momentum_idx;
        int combo_idx = task.combo_idx;
        int sublattice_i = task.sublattice_i;
        int sublattice_j = task.sublattice_j;
        
        int sample_index = sample_indices[state_idx];
        double beta = beta_values[state_idx];
        std::string beta_str = beta_strings[state_idx];
        std::string filename = fs::path(tpq_files[state_idx]).filename().string();
        std::string output_dir = output_base_dir + "/beta_" + beta_str;
        
        // Load state (or reuse from cache - TODO: implement caching for efficiency)
        ComplexVector tpq_state;
        bool loaded_ok = false;
        if (filename.find("eigenvector") != std::string::npos) {
            loaded_ok = load_raw_data(tpq_state, tpq_files[state_idx], N64);
        } else {
            loaded_ok = load_tpq_state(tpq_state, tpq_files[state_idx]);
        }
        
        if (!loaded_ok || (int)tpq_state.size() != N) {
            std::cerr << "Rank " << rank << " failed to load/validate state from " << filename << std::endl;
            return false;
        }
        
        ensureDirectoryExists(output_dir);
        
        // Special methods that don't follow (operators) × (method) structure
        if (method == "spin_correlation") {
            printSpinCorrelation(tpq_state, num_sites, spin_length, output_dir, unit_cell_size);
            return true;
        } else if (method == "spin_configuration") {
            printSpinConfiguration(tpq_state, num_sites, spin_length, output_dir);
            return true;
        }
        
        // ============================================================
        // Main (operators) × (method) structure
        // ============================================================
        
        // STEP 1: Construct operators based on operator_type
        std::vector<Operator> obs_1, obs_2;
        std::vector<std::string> obs_names;
        std::string method_dir = output_dir + "/" + operator_type;
        
        // Add suffix to directory if using fixed-Sz
        if (use_fixed_sz) {
            method_dir += "_fixed_sz_nup" + std::to_string(n_up);
        }
        
        const auto &Q = momentum_points[momentum_idx];
        int op_type_1 = spin_combinations[combo_idx].first;
        int op_type_2 = spin_combinations[combo_idx].second;
        std::string base_name = std::string(spin_combination_names[combo_idx]);
        
        try {
            if (operator_type == "sum") {
                // Standard sum operators: S^{op1}(Q) S^{op2}(-Q)
                std::stringstream name_ss;
                name_ss << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
                obs_names.push_back(name_ss.str());
                
                if (use_fixed_sz) {
                    // Fixed-Sz operators
                    if (use_xyz_basis) {
                        FixedSzSumOperatorXYZ sum_op_1(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], positions_file);
                        FixedSzSumOperatorXYZ sum_op_2(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    } else {
                        FixedSzSumOperator sum_op_1(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], positions_file);
                        FixedSzSumOperator sum_op_2(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    }
                } else {
                    // Full Hilbert space operators
                    if (use_xyz_basis) {
                        SumOperatorXYZ sum_op_1(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], positions_file);
                        SumOperatorXYZ sum_op_2(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    } else {
                        SumOperator sum_op_1(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], positions_file);
                        SumOperator sum_op_2(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    }
                }
                
            } else if (operator_type == "transverse") {
                // Transverse operators for SF/NSF separation
                const auto &b1 = transverse_basis_1[momentum_idx];
                const auto &b2 = transverse_basis_2[momentum_idx];
                std::vector<double> e1_vec = {b1[0], b1[1], b1[2]};
                std::vector<double> e2_vec = {b2[0], b2[1], b2[2]};
                
                // SF component (transverse_basis_1)
                std::stringstream name_sf;
                name_sf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_NSF";
                
                // NSF component (transverse_basis_2)
                std::stringstream name_nsf;
                name_nsf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_SF";
                
                if (use_fixed_sz) {
                    // Fixed-Sz transverse operators
                    if (use_xyz_basis) {
                        FixedSzTransverseOperatorXYZ op1_sf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op2_sf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op1_nsf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op2_nsf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    } else {
                        FixedSzTransverseOperator op1_sf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperator op2_sf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperator op1_nsf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        FixedSzTransverseOperator op2_nsf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    }
                } else {
                    // Full Hilbert space transverse operators
                    if (use_xyz_basis) {
                        TransverseOperatorXYZ op1_sf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperatorXYZ op2_sf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperatorXYZ op1_nsf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        TransverseOperatorXYZ op2_nsf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    } else {
                        TransverseOperator op1_sf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperator op2_sf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperator op1_nsf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        TransverseOperator op2_nsf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    }
                }
                
                obs_names.push_back(name_sf.str());
                obs_names.push_back(name_nsf.str());
                
            } else if (operator_type == "sublattice") {
                // Sublattice-resolved operators
                std::stringstream name_ss;
                name_ss << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2]
                        << "_sub" << sublattice_i << "_sub" << sublattice_j;
                obs_names.push_back(name_ss.str());
                
                std::vector<double> Q_vec = {Q[0], Q[1], Q[2]};
                
                if (use_fixed_sz) {
                    // Fixed-Sz sublattice operators
                    FixedSzSublatticeOperator sub_op_1(sublattice_i, unit_cell_size, num_sites, spin_length, n_up, op_type_1, Q_vec, positions_file);
                    FixedSzSublatticeOperator sub_op_2(sublattice_j, unit_cell_size, num_sites, spin_length, n_up, op_type_2, Q_vec, positions_file);
                    
                    obs_1.push_back(Operator(sub_op_1));
                    obs_2.push_back(Operator(sub_op_2));
                } else {
                    // Full Hilbert space sublattice operators
                    SublatticeOperator sub_op_1(sublattice_i, unit_cell_size, num_sites, spin_length, op_type_1, Q_vec, positions_file);
                    SublatticeOperator sub_op_2(sublattice_j, unit_cell_size, num_sites, spin_length, op_type_2, Q_vec, positions_file);
                    
                    obs_1.push_back(Operator(sub_op_1));
                    obs_2.push_back(Operator(sub_op_2));
                }
                
            } else if (operator_type == "experimental") {
                // Experimental operators: cos(θ)Sz + sin(θ)Sx
                std::stringstream name_ss;
                name_ss << "Experimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_theta" << theta;
                obs_names.push_back(name_ss.str());
                
                if (use_fixed_sz) {
                    // Fixed-Sz experimental operators
                    FixedSzExperimentalOperator exp_op_1(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], positions_file);
                    FixedSzExperimentalOperator exp_op_2(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], positions_file);
                    
                    obs_1.push_back(Operator(exp_op_1));
                    obs_2.push_back(Operator(exp_op_2));
                } else {
                    // Full Hilbert space experimental operators
                    ExperimentalOperator exp_op_1(num_sites, spin_length, theta, momentum_points[momentum_idx], positions_file);
                    ExperimentalOperator exp_op_2(num_sites, spin_length, theta, momentum_points[momentum_idx], positions_file);
                    
                    obs_1.push_back(Operator(exp_op_1));
                    obs_2.push_back(Operator(exp_op_2));
                }
                
            } else if (operator_type == "transverse_experimental") {
                // Transverse experimental operators with SF/NSF separation: cos(θ)Sz + sin(θ)Sx
                const auto &b1 = transverse_basis_1[momentum_idx];
                const auto &b2 = transverse_basis_2[momentum_idx];
                std::vector<double> e1_vec = {b1[0], b1[1], b1[2]};
                std::vector<double> e2_vec = {b2[0], b2[1], b2[2]};
                
                // SF component (transverse_basis_1)
                std::stringstream name_sf;
                name_sf << "TransverseExperimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] 
                        << "_theta" << theta << "_NSF";
                
                // NSF component (transverse_basis_2)
                std::stringstream name_nsf;
                name_nsf << "TransverseExperimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] 
                         << "_theta" << theta << "_SF";
                
                if (use_fixed_sz) {
                    // Fixed-Sz transverse experimental operators
                    FixedSzTransverseExperimentalOperator op1_sf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    FixedSzTransverseExperimentalOperator op2_sf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    FixedSzTransverseExperimentalOperator op1_nsf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    FixedSzTransverseExperimentalOperator op2_nsf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    
                    obs_1.push_back(Operator(op1_sf));
                    obs_2.push_back(Operator(op2_sf));
                    obs_1.push_back(Operator(op1_nsf));
                    obs_2.push_back(Operator(op2_nsf));
                } else {
                    // Full Hilbert space transverse experimental operators
                    TransverseExperimentalOperator op1_sf(num_sites, spin_length, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    TransverseExperimentalOperator op2_sf(num_sites, spin_length, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    TransverseExperimentalOperator op1_nsf(num_sites, spin_length, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    TransverseExperimentalOperator op2_nsf(num_sites, spin_length, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    
                    obs_1.push_back(Operator(op1_sf));
                    obs_2.push_back(Operator(op2_sf));
                    obs_1.push_back(Operator(op1_nsf));
                    obs_2.push_back(Operator(op2_nsf));
                }
                
                obs_names.push_back(name_sf.str());
                obs_names.push_back(name_nsf.str());
            }
            
        } catch (const std::exception &e) {
            std::cerr << "Rank " << rank << " failed operator construction: " << e.what() << std::endl;
            return false;
        }
        
        // STEP 2: Apply time evolution method to the constructed operators
        ensureDirectoryExists(method_dir);
        
        try {
            // CPU implementation
            if (method == "taylor") {
                computeObservableDynamics_U_t(
                    U_t, tpq_state, obs_1, obs_2, obs_names, N,
                    method_dir, sample_index, beta, t_end_opt, dt_opt
                );
            } else if (method == "krylov") {
                // Use Krylov method with Operator objects directly
                int krylov_dim = krylov_dim_or_nmax;
                
                computeDynamicCorrelationsKrylov(
                    H, tpq_state, obs_1, obs_2, obs_names,
                    N, method_dir, sample_index, beta, t_end_opt, dt_opt, krylov_dim
                );
            } else if (method == "spectral") {
                    // Use spectral method with FTLM approach
                    int krylov_dim = krylov_dim_or_nmax;
                    
#ifdef WITH_CUDA
                    if (use_gpu) {
                        std::cout << "Using GPU-accelerated FTLM spectral calculation..." << std::endl;
                        
                        // Convert CPU Hamiltonian to GPU
                        GPUOperator gpu_ham(num_sites, spin_length);
                        if (!convertOperatorToGPU(ham_op, gpu_ham)) {
                            std::cerr << "Failed to convert Hamiltonian to GPU, falling back to CPU" << std::endl;
                            use_gpu = false;
                        } else {
                            // Process each operator pair on GPU
                            for (size_t i = 0; i < obs_1.size(); i++) {
                                std::cout << "  Processing operator pair " << (i+1) << "/" << obs_1.size() 
                                          << ": " << obs_names[i] << std::endl;
                                
                                // Convert observable operators to GPU
                                GPUOperator gpu_obs1(num_sites, spin_length);
                                GPUOperator gpu_obs2(num_sites, spin_length);
                                
                                if (!convertOperatorToGPU(obs_1[i], gpu_obs1) || 
                                    !convertOperatorToGPU(obs_2[i], gpu_obs2)) {
                                    std::cerr << "  Failed to convert operators to GPU, skipping..." << std::endl;
                                    continue;
                                }
                                
                                // Allocate device memory for TPQ state
                                cuDoubleComplex* d_psi;
                                cudaMalloc(&d_psi, N * sizeof(cuDoubleComplex));
                                cudaMemcpy(d_psi, tpq_state.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
                                
                                // Call GPU wrapper for dynamical correlation on a single state
                                auto [frequencies, S_real, S_imag] = GPUEDWrapper::runGPUDynamicalCorrelationState(
                                    &gpu_ham, &gpu_obs1, &gpu_obs2,
                                    d_psi,  // Device pointer to TPQ state
                                    N,
                                    krylov_dim,
                                    omega_min, omega_max, num_omega_bins,
                                    broadening,
                                    0.0,  // temperature (not used for single-state)
                                    ground_state_energy
                                );
                                
                                // Package results
                                DynamicalResponseResults results;
                                results.frequencies = frequencies;
                                results.spectral_function = S_real;
                                results.spectral_function_imag = S_imag;
                                // Initialize error vectors to zero (single-state, no error bars)
                                results.spectral_error.resize(frequencies.size(), 0.0);
                                results.spectral_error_imag.resize(frequencies.size(), 0.0);
                                results.total_samples = 1;
                                
                                // Save results
                                std::stringstream filename_ss;
                                filename_ss << method_dir << "/" << obs_names[i] 
                                            << "_spectral_sample_" << sample_index 
                                            << "_beta_" << beta << ".txt";
                                
                                save_dynamical_response_results(results, filename_ss.str());
                                
                                if (rank == 0) {
                                    std::cout << "  Saved GPU spectral function: " << filename_ss.str() << std::endl;
                                }
                                
                                // Cleanup device memory
                                cudaFree(d_psi);
                            }
                        }
                    }
#endif
                    
                    if (!use_gpu) {
                        // CPU spectral calculation
                        // Set up FTLM parameters
                        DynamicalResponseParameters params;
                        params.krylov_dim = krylov_dim;
                        params.broadening = broadening;
                        params.tolerance = 1e-10;
                        params.full_reorthogonalization = true;
                        
                        // Process each operator pair
                        for (size_t i = 0; i < obs_1.size(); i++) {
                            // Create function wrappers for operators
                            auto O1_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                                obs_1[i].apply(in, out, size);
                            };
                            
                            auto O2_func = [&obs_2, i](const Complex* in, Complex* out, int size) {
                                obs_2[i].apply(in, out, size);
                            };
                            
                            // Compute spectral function
                            auto results = compute_dynamical_correlation_state(
                                H, O1_func, O2_func, tpq_state, N, params,
                                omega_min, omega_max, num_omega_bins, 0.0, ground_state_energy
                            );
                            
                            // Save results
                            std::stringstream filename_ss;
                            filename_ss << method_dir << "/" << obs_names[i] 
                                        << "_spectral_sample_" << sample_index 
                                        << "_beta_" << beta << ".txt";
                            
                            save_dynamical_response_results(results, filename_ss.str());
                            
                            if (rank == 0) {
                                std::cout << "  Saved spectral function: " << filename_ss.str() << std::endl;
                            }
                        }
                    }
                } else if (method == "spectral_thermal") {
                    // Use spectral method with finite-temperature FTLM (thermal averaging)
                    // This uses random sampling to compute thermal averages at finite temperature
                    int krylov_dim = krylov_dim_or_nmax;
                    int num_samples = 40;  // Number of random samples for FTLM thermal averaging
                    unsigned int random_seed = state_idx * 1000 + momentum_idx * 100 + combo_idx;
                    
                    // Determine temperature(s) to compute
                    std::vector<double> temperatures;
                    
                    if (use_temperature_scan) {
                        // Use user-specified temperature range (log spacing)
                        double log_T_min = std::log(T_min);
                        double log_T_max = std::log(T_max);
                        double log_step = (log_T_max - log_T_min) / std::max(1, T_steps - 1);
                        
                        for (int i = 0; i < T_steps; i++) {
                            double log_T = log_T_min + i * log_step;
                            temperatures.push_back(std::exp(log_T));
                        }
                        
                        if (rank == 0) {
                            std::cout << "  Computing for " << T_steps << " temperature points:" << std::endl;
                            for (size_t i = 0; i < temperatures.size(); i++) {
                                std::cout << "    T[" << i << "] = " << temperatures[i] << std::endl;
                            }
                        }
                    } else {
                        // Use temperature from TPQ state beta
                        double temperature = 0.0;
                        if (std::isfinite(beta) && beta > 1e-10) {
                            temperature = 1.0 / beta;
                        }
                        temperatures.push_back(temperature);
                        
                        if (rank == 0) {
                            std::cout << "  Using TPQ state temperature: T = " << temperature 
                                      << " (β = " << beta << ")" << std::endl;
                        }
                    }
                    
                    // GPU/CPU thermal spectral calculation
#ifdef WITH_CUDA
                    if (use_gpu) {
                        std::cout << "Using GPU-accelerated FTLM thermal spectral calculation..." << std::endl;
                        
                        // Convert CPU Hamiltonian to GPU
                        GPUOperator gpu_ham(num_sites, spin_length);
                        if (!convertOperatorToGPU(ham_op, gpu_ham)) {
                            std::cerr << "Failed to convert Hamiltonian to GPU, falling back to CPU" << std::endl;
                            use_gpu = false;
                        } else {
                            // Process each operator pair on GPU
                            for (size_t i = 0; i < obs_1.size(); i++) {
                                std::cout << "  Processing operator pair " << (i+1) << "/" << obs_1.size() 
                                          << ": " << obs_names[i] << std::endl;
                                
                                // Convert observable operators to GPU
                                GPUOperator gpu_obs1(num_sites, spin_length);
                                GPUOperator gpu_obs2(num_sites, spin_length);
                                
                                if (!convertOperatorToGPU(obs_1[i], gpu_obs1) || 
                                    !convertOperatorToGPU(obs_2[i], gpu_obs2)) {
                                    std::cerr << "  Failed to convert operators to GPU, skipping..." << std::endl;
                                    continue;
                                }
                                
                                // Allocate device memory for TPQ state (reused for all temperatures)
                                cuDoubleComplex* d_psi;
                                cudaMalloc(&d_psi, N * sizeof(cuDoubleComplex));
                                cudaMemcpy(d_psi, tpq_state.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
                                
                                // Call GPU wrapper for thermal spectral function
                                // Note: Currently processes temperatures sequentially
                                // TODO: Implement multi-temperature optimization on GPU
                                for (double temperature : temperatures) {
                                    std::cout << "    Computing at T = " << temperature << std::endl;
                                    
                                    auto [frequencies, S_real, S_imag] = GPUEDWrapper::runGPUDynamicalCorrelationState(
                                        &gpu_ham, &gpu_obs1, &gpu_obs2,
                                        d_psi,  // Device pointer to state
                                        N,
                                        krylov_dim,
                                        omega_min, omega_max, num_omega_bins,
                                        broadening,
                                        temperature,
                                        ground_state_energy
                                    );
                                    
                                    // Package results
                                    DynamicalResponseResults results;
                                    results.frequencies = frequencies;
                                    results.spectral_function = S_real;
                                    results.spectral_function_imag = S_imag;
                                    // Initialize error vectors to zero (single-state, no error bars)
                                    results.spectral_error.resize(frequencies.size(), 0.0);
                                    results.spectral_error_imag.resize(frequencies.size(), 0.0);
                                    results.total_samples = 1;
                                    
                                    // Save results
                                    std::stringstream filename_ss;
                                    filename_ss << method_dir << "/" << obs_names[i] 
                                                << "_spectral_thermal_sample_" << sample_index 
                                                << "_T_" << temperature << ".txt";
                                    
                                    save_dynamical_response_results(results, filename_ss.str());
                                    
                                    if (rank == 0) {
                                        std::cout << "    Saved GPU thermal spectral: " << filename_ss.str() << std::endl;
                                    }
                                }
                                
                                // Cleanup device memory
                                cudaFree(d_psi);
                            }
                        }
                    }
#endif
                    
                    if (!use_gpu) {
                        // CPU thermal spectral calculation
                        // Set up FTLM parameters for thermal averaging
                        DynamicalResponseParameters params;
                        params.krylov_dim = krylov_dim;
                        params.broadening = broadening;
                        params.tolerance = 1e-10;
                        params.full_reorthogonalization = true;
                        params.num_samples = num_samples;
                        params.random_seed = random_seed;
                        params.store_intermediate = false;
                        
                        if (rank == 0) {
                            std::cout << "Observables to compute thermal spectral functions for:" << std::endl;
                            std::cout << "  Number of operators: " << obs_1.size() << std::endl;
                        }
                        
                        // Process each operator pair
                        for (size_t i = 0; i < obs_1.size(); i++) {
                            std::cout << "  Processing operator " << obs_names[i] << std::endl;
                            // Create function wrappers for operators
                            auto O1_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                                obs_1[i].apply(in, out, size);
                            };
                            
                            auto O2_func = [&obs_2, i](const Complex* in, Complex* out, int size) {
                                obs_2[i].apply(in, out, size);
                            };
                            
                            // OPTIMIZATION: Compute spectral function for ALL temperatures at once
                            // This runs Lanczos ONCE and reuses the spectral decomposition for all temperatures
                            // Much more efficient than running Lanczos separately for each temperature!
                            if (rank == 0) {
                                std::cout << "  *** OPTIMIZED MODE: Computing " << temperatures.size() 
                                          << " temperature points with SINGLE Lanczos run ***" << std::endl;
                            }
                            
                            // Use the optimized multi-temperature function
                            auto results_map = compute_dynamical_correlation_state_multi_temperature(
                                H, O1_func, O2_func, tpq_state, N, params,
                                omega_min, omega_max, num_omega_bins, 
                                temperatures, ground_state_energy
                            );
                            
                            // Save results for each temperature
                            for (const auto& [temperature, results] : results_map) {
                                std::stringstream filename_ss;
                                filename_ss << method_dir << "/" << obs_names[i] 
                                            << "_spectral_thermal_sample_" << sample_index 
                                            << "_beta_" << std::fixed << std::setprecision(6) << (1.0/temperature)
                                            << "_T_" << temperature << "_nsamples_1.txt";
                                
                                save_dynamical_response_results(results, filename_ss.str());
                            
                                if (rank == 0) {
                                    std::cout << "  Saved thermal spectral function: " << filename_ss.str() << std::endl;
                                    std::cout << "    Temperature: " << temperature << ", Beta: " << (1.0/temperature) << std::endl;
                                }
                            }
                            
                            if (rank == 0) {
                                std::cout << "  *** Optimization saved ~" << (temperatures.size() - 1) 
                                          << " Lanczos iterations! ***" << std::endl;
                            }
                        }
                    }
                } else {
                    std::cerr << "Rank " << rank << " unknown method: " << method << std::endl;
                    return false;
                }
        } catch (const std::exception &e) {
            std::cerr << "Rank " << rank << " failed time evolution: " << e.what() << std::endl;
            return false;
        }
        
        return true;
    };
    
    // Dynamic work distribution using master-worker pattern
    int local_processed_count = 0;
    double start_time = MPI_Wtime();
    
    if (size == 1) {
        // Serial execution
        for (const auto& task : all_tasks) {
            if (process_task(task)) {
                local_processed_count++;
            }
        }
    } else {
        // Master-worker dynamic scheduling with rank 0 also processing tasks
        const int TASK_TAG = 1;
        const int STOP_TAG = 2;
        const int DONE_TAG = 3;
        const int REQUEST_TAG = 4;
        
        if (rank == 0) {
            // Master: dispatch tasks and also process tasks itself
            int next_task = 0;
            int active_workers = std::min(size - 1, num_tasks);
            
            // Send initial batch to other workers
            for (int r = 1; r <= active_workers; ++r) {
                MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
                next_task++;
            }
            
            // Idle remaining workers
            for (int r = active_workers + 1; r < size; ++r) {
                int dummy = -1;
                MPI_Send(&dummy, 1, MPI_INT, r, STOP_TAG, MPI_COMM_WORLD);
            }
            
            // Process tasks on rank 0 while managing other workers
            int completed = 0;
            while (completed < num_tasks) {
                // Check if rank 0 can grab a task
                if (next_task < num_tasks) {
                    int my_task = next_task;
                    next_task++;
                    
                    std::cout << "Rank 0 processing task " << my_task << "/" << num_tasks;
                    if (all_tasks[my_task].momentum_idx >= 0) {
                        std::cout << " (state=" << all_tasks[my_task].state_idx 
                                  << ", Q=" << all_tasks[my_task].momentum_idx
                                  << ", combo=" << all_tasks[my_task].combo_idx;
                        if (all_tasks[my_task].sublattice_i >= 0) {
                            std::cout << ", sub_i=" << all_tasks[my_task].sublattice_i 
                                      << ", sub_j=" << all_tasks[my_task].sublattice_j;
                        }
                        std::cout << ")";
                    }
                    std::cout << std::endl;
                    
                    if (process_task(all_tasks[my_task])) {
                        local_processed_count++;
                    }
                    completed++;
                }
                
                // Check for completed tasks from other workers (non-blocking)
                int flag;
                MPI_Status status;
                MPI_Iprobe(MPI_ANY_SOURCE, DONE_TAG, MPI_COMM_WORLD, &flag, &status);
                
                if (flag) {
                    int done_task;
                    MPI_Recv(&done_task, 1, MPI_INT, status.MPI_SOURCE, DONE_TAG, MPI_COMM_WORLD, &status);
                    completed++;
                    
                    if (next_task < num_tasks) {
                        MPI_Send(&next_task, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
                        next_task++;
                    } else {
                        int dummy = -1;
                        MPI_Send(&dummy, 1, MPI_INT, status.MPI_SOURCE, STOP_TAG, MPI_COMM_WORLD);
                    }
                }
            }
        } else {
            // Worker: request and process tasks
            while (true) {
                int task_id;
                MPI_Status status;
                MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                
                if (status.MPI_TAG == STOP_TAG) {
                    break;
                }
                
                std::cout << "Rank " << rank << " processing task " << task_id << "/" << num_tasks;
                if (all_tasks[task_id].momentum_idx >= 0) {
                    std::cout << " (state=" << all_tasks[task_id].state_idx 
                              << ", Q=" << all_tasks[task_id].momentum_idx
                              << ", combo=" << all_tasks[task_id].combo_idx;
                    if (all_tasks[task_id].sublattice_i >= 0) {
                        std::cout << ", sub_i=" << all_tasks[task_id].sublattice_i 
                                  << ", sub_j=" << all_tasks[task_id].sublattice_j;
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;
                
                if (process_task(all_tasks[task_id])) {
                    local_processed_count++;
                }
                
                MPI_Send(&task_id, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // Gather timing and count statistics
    std::vector<double> all_times;
    if (rank == 0) {
        all_times.resize(size);
    }
    MPI_Gather(&elapsed_time, 1, MPI_DOUBLE, rank == 0 ? all_times.data() : nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int total_processed_count;
    MPI_Reduce(&local_processed_count, &total_processed_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Processing complete!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Processed " << total_processed_count << "/" << num_tasks << " tasks successfully." << std::endl;
        std::cout << "Results saved in: " << output_base_dir << std::endl;
        
        // Print timing statistics
        std::cout << "\nTiming statistics:" << std::endl;
        auto max_it = std::max_element(all_times.begin(), all_times.end());
        auto min_it = std::min_element(all_times.begin(), all_times.end());
        double max_time = *max_it;
        double min_time = *min_it;
        double avg_time = std::accumulate(all_times.begin(), all_times.end(), 0.0) / size;
        double load_imbalance = 0.0;
        if (max_time > 0.0) {
            load_imbalance = (max_time - min_time) / max_time * 100.0;
        }
        
        std::cout << "  Max time: " << max_time << " seconds (Rank " 
                  << std::distance(all_times.begin(), max_it) << ")" << std::endl;
        std::cout << "  Min time: " << min_time << " seconds (Rank " 
                  << std::distance(all_times.begin(), min_it) << ")" << std::endl;
        std::cout << "  Avg time: " << avg_time << " seconds" << std::endl;
        std::cout << "  Load imbalance: " << std::fixed << std::setprecision(2) << load_imbalance << "%" << std::endl;
        
        std::cout << "\nPer-rank timing:" << std::endl;
        for (int r = 0; r < size; r++) {
            std::cout << "  Rank " << r << ": " << all_times[r] << " seconds" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
