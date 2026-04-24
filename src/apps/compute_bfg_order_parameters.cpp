/**
 * @file compute_bfg_order_parameters.cpp
 * @brief Fast C++ computation of BFG order parameters from wavefunctions
 * 
 * Computes:
 * 1. S(q) - Spin structure factor using S^-S^+ correlations at ALL k-points + 2D grid
 * 2. Nematic order - Bond orientation anisotropy (C3 → C1 breaking)
 *    - Variants: XY (S+S- + S-S+), S+S-, SzSz, Heisenberg (S·S)
 * 3. VBS (Valence Bond Solid) order - S_D(q) with proper 4-site correlations
 *    - Computes actual ⟨D_b D_b'⟩ dimer-dimer correlations
 * 
 * Output includes:
 * - Order parameters at special k-points (Γ, K, M, etc.)
 * - Full 2D structure factor grids for visualization
 * - Per-bond expectations for spatial visualization
 * - Detailed HDF5 output compatible with Python plotting
 * 
 * Usage:
 *   ./compute_bfg_order_parameters <wavefunction.h5> <cluster_dir> [output.h5]
 * 
 * Compile with:
 *   g++ -O3 -march=native -fopenmp -std=c++17 compute_bfg_order_parameters.cpp -o compute_bfg_order_parameters -lhdf5 -lhdf5_cpp
 */

#include <atomic>
#include <chrono>
#include <complex>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "ed/bfg/cluster.h"
#include "ed/bfg/correlations.h"
#include "ed/bfg/order_parameters.h"
#include "ed/bfg/results_io.h"
#include "ed/bfg/spin_structure_factor.h"
#include "ed/bfg/structure_factor.h"
#include "ed/bfg/wavefunction_io.h"

namespace fs = std::filesystem;

using Complex = std::complex<double>;

// P2.1 (slices 1-8): every BFG physics kernel and POD lives in the ed_bfg
// static library. The orchestration code below (process_all_temperatures /
// scan_jpm_directories / int main()) is the only thing that still lives
// here; pull the symbols it touches into the local TU so the call sites
// stay unqualified.
using ed::bfg::Cluster;
using ed::bfg::compute_all_order_parameters;
using ed::bfg::compute_heisenberg_bond_expectations;
using ed::bfg::compute_nematic_order;
using ed::bfg::compute_nematic_order_real;
using ed::bfg::compute_plaquette_order;
using ed::bfg::compute_smsp_correlations;
using ed::bfg::compute_spin_structure_factor;
using ed::bfg::compute_spsm_bond_expectations;
using ed::bfg::compute_sq_2d_grid;
using ed::bfg::compute_szsz_bond_expectations;
using ed::bfg::compute_szsz_correlations;
using ed::bfg::compute_vbs_order;
using ed::bfg::compute_xy_bond_expectations;
using ed::bfg::load_all_tpq_states;
using ed::bfg::load_cluster;
using ed::bfg::load_tpq_state;
using ed::bfg::load_wavefunction;
using ed::bfg::OrderParameterResults;
using ed::bfg::save_results;
using ed::bfg::save_scan_results;
using ed::bfg::save_temperature_scan_results;
using ed::bfg::set_memory_efficient_mode;

// -----------------------------------------------------------------------------
// All physics kernels (Cluster, wavefunction loaders, two-body correlations,
// bond expectations, structure factors, ring observables, spin structure
// factor, nematic / VBS / plaquette / 2D-grid order parameters,
// compute_all_order_parameters, and the HDF5 results writers) now live in
// the ed_bfg static library (include/ed/bfg/*.h, src/bfg/*.cpp). The
// using-declarations at the top of this TU pull every name into the local
// translation unit so the orchestration code below (process_all_temperatures
// / scan_jpm_directories / int main()) reads identically to the historical
// monolithic driver. P2.1 (slices 1-8).
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Process all temperatures for a single Jpm directory (for tpq_all_temps mode)
// Returns a vector of results, one per temperature
// -----------------------------------------------------------------------------

std::vector<OrderParameterResults> process_all_temperatures(
    const std::string& wf_file,
    const Cluster& cluster,
    double jpm,
    int n_q_grid,
    bool save_full
) {
    std::vector<OrderParameterResults> results_list;
    
    // Load all TPQ states (all temperatures)
    auto tpq_states = load_all_tpq_states(wf_file);
    
    if (tpq_states.empty()) {
        std::cerr << "No TPQ states found in " << wf_file << std::endl;
        return results_list;
    }
    
    std::cout << "  Processing " << tpq_states.size() << " temperatures for Jpm=" << jpm << std::endl;
    
    // Process each temperature (using quick scalar computation; full mode not yet implemented for T-scan)
    (void)n_q_grid;
    (void)save_full;
    
    for (size_t t_idx = 0; t_idx < tpq_states.size(); ++t_idx) {
        const auto& tpq_state = tpq_states[t_idx];
        
        OrderParameterResults results = compute_all_order_parameters(tpq_state.psi, cluster, jpm);
        results.temperature = tpq_state.temperature;
        
        results_list.push_back(results);
    }
    
    return results_list;
}

// `save_temperature_scan_results` now lives in `ed_bfg::results_io`. The
// using-declaration at the top of this TU keeps the call site below
// unchanged. P2.1 (7th slice).

// -----------------------------------------------------------------------------
// Scan directory mode
// -----------------------------------------------------------------------------

std::vector<OrderParameterResults> scan_jpm_directories(
    const std::string& scan_dir,
    const std::string& output_dir,
    int n_workers,
    int n_q_grid,
    bool save_full,
    bool use_tpq,
    bool tpq_all_temps = false
) {
    // Get MPI rank and size
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    
    // Find all Jpm=* directories (all ranks do this)
    std::vector<std::pair<double, std::string>> jpm_dirs;
    std::regex jpm_regex("Jpm=([+-]?[0-9]*\\.?[0-9]+)");
    
    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        std::smatch match;
        if (std::regex_search(dirname, match, jpm_regex)) {
            double jpm = std::stod(match[1].str());
            jpm_dirs.push_back({jpm, entry.path().string()});
        }
    }
    
    std::sort(jpm_dirs.begin(), jpm_dirs.end());
    
    if (mpi_rank == 0) {
        std::cout << "Found " << jpm_dirs.size() << " Jpm directories" << std::endl;
#ifdef USE_MPI
        std::cout << "MPI enabled: " << mpi_size << " processes" << std::endl;
#endif
        if (save_full) {
            std::cout << "Full output mode: saving S(q) and S_D(q) 2D grids per directory" << std::endl;
        }
        if (tpq_all_temps) {
            std::cout << "TPQ all-temps mode: processing ALL temperatures for each Jpm" << std::endl;
        }
    }
    
    if (jpm_dirs.empty()) {
        return {};
    }
    
    // Load cluster from first directory
    Cluster cluster = load_cluster(jpm_dirs[0].second);
    if (mpi_rank == 0) {
        std::cout << "Cluster: " << cluster.n_sites << " sites, " 
                  << cluster.edges_nn.size() << " bonds" << std::endl;
    }
    
    // Process directories - each MPI rank processes a subset
    std::vector<OrderParameterResults> all_results(jpm_dirs.size());
    std::mutex print_mutex;
    std::atomic<int> completed(0);
    
    // Distribute work: rank processes indices where (i % mpi_size == mpi_rank)
    #pragma omp parallel for schedule(dynamic) num_threads(n_workers)
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
        // MPI work distribution: only process if this index belongs to this rank
#ifdef USE_MPI
        if (static_cast<int>(i % mpi_size) != mpi_rank) {
            continue;
        }
#endif
        double jpm = jpm_dirs[i].first;
        const std::string& dir = jpm_dirs[i].second;
        
        try {
            // Find wavefunction file
            std::string wf_file;
            
            // First check output/ subdirectory for ed_results.h5
            std::string output_subdir = dir + "/output";
            if (fs::exists(output_subdir)) {
                std::string ed_results = output_subdir + "/ed_results.h5";
                if (fs::exists(ed_results)) {
                    wf_file = ed_results;
                } else {
                    // Search for any .h5 file in output/
                    for (const auto& entry : fs::directory_iterator(output_subdir)) {
                        if (entry.path().extension() == ".h5") {
                            wf_file = entry.path().string();
                            break;
                        }
                    }
                }
            }
            
            // If not found in output/, try main directory
            if (wf_file.empty()) {
                for (const auto& entry : fs::directory_iterator(dir)) {
                    std::string fname = entry.path().filename().string();
                    if (fname.find(".h5") != std::string::npos && 
                        fname.find("eigenvector") != std::string::npos) {
                        wf_file = entry.path().string();
                        break;
                    }
                }
            }
            
            if (wf_file.empty()) {
                // Try other patterns in main directory
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".h5") {
                        wf_file = entry.path().string();
                        break;
                    }
                }
            }
            
            if (wf_file.empty()) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "No wavefunction file found in " << dir << std::endl;
                continue;
            }
            
            // ============================================================
            // TPQ ALL TEMPERATURES MODE: process all beta values and save
            // per-Jpm temperature-dependent results
            // ============================================================
            if (tpq_all_temps) {
                auto temp_results = process_all_temperatures(wf_file, cluster, jpm, n_q_grid, save_full);
                
                if (temp_results.empty()) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cerr << "No TPQ states found in " << wf_file << std::endl;
                    continue;
                }
                
                // Save per-Jpm temperature scan file
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << jpm;
                std::string out_file = output_dir + "/order_params_Jpm=" + oss.str() + "_T_scan.h5";
                save_temperature_scan_results(temp_results, out_file, jpm);
                
                // Use lowest temperature result for summary
                all_results[i] = temp_results.back();  // Sorted by T descending, so last is lowest
                
                int done = ++completed;
                {
                    std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
                    std::cout << "[Rank " << mpi_rank << "] ";
#endif
                    std::cout << "[" << done << "/" << jpm_dirs.size() << "] "
                              << "Jpm=" << std::fixed << std::setprecision(4) << jpm
                              << " | " << temp_results.size() << " temperatures processed"
                              << " | T_range=[" << temp_results.back().temperature 
                              << ", " << temp_results.front().temperature << "]"
                              << std::endl;
                }
                continue;  // Skip the normal single-temperature processing
            }
            
            // Load wavefunction (ground state or TPQ lowest temperature)
            std::vector<Complex> psi;
            double temperature = 0.0;
            
            if (use_tpq) {
                try {
                    auto [tpq_psi, T] = load_tpq_state(wf_file);
                    psi = std::move(tpq_psi);
                    temperature = T;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cerr << "TPQ load failed for " << dir << ": " << e.what() 
                              << ", falling back to ground state" << std::endl;
                    psi = load_wavefunction(wf_file);
                }
            } else {
                psi = load_wavefunction(wf_file);
            }
            
            // Set memory-efficient mode on first file processed by this thread
            // (all files should have same Hilbert space size)
            #pragma omp critical(set_memory_mode)
            {
                static bool memory_mode_set = false;
                if (!memory_mode_set) {
                    set_memory_efficient_mode(psi.size());
                    memory_mode_set = true;
                }
            }
            
            OrderParameterResults results;
            results.jpm = jpm;
            results.temperature = temperature;
            
            if (save_full) {
                // Full computation with 2D grids
                auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
                auto szsz_corr = compute_szsz_correlations(psi, cluster.n_sites);
                auto sf_result = compute_spin_structure_factor(smsp_corr, szsz_corr, cluster);
                auto s_q_2d = compute_sq_2d_grid(smsp_corr, szsz_corr, cluster, n_q_grid);
                
                auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
                auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
                auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
                auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
                
                auto nem_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
                auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
                auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
                auto nem_heisenberg_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
                
                auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster, n_q_grid);
                auto plaq_result = compute_plaquette_order(psi, cluster, n_q_grid);
                
                // Save full results to per-Jpm file
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << jpm;
                std::string out_file = output_dir + "/order_params_Jpm=" + oss.str() + ".h5";
                save_results(out_file, sf_result, nem_result, nem_spsm_result, nem_szsz_result,
                            nem_heisenberg_result, vbs_result, plaq_result,
                            cluster, s_q_2d, n_q_grid,
                            spsm_bond_exp, szsz_bond_exp, heisenberg_bond_exp);
                
                // Fill scalar results for summary
                results.m_translation = sf_result.m_translation;
                results.m_nematic = nem_result.m_nem;
                results.m_nematic_spsm = nem_spsm_result.m_nem;
                results.m_nematic_szsz = nem_szsz_result.m_nem;
                results.m_nematic_heisenberg = nem_heisenberg_result.m_nem;
                results.anisotropy = nem_result.anisotropy;
                results.m_vbs = vbs_result.m_vbs_xy;
                results.m_vbs_xy = vbs_result.m_vbs_xy;
                results.m_vbs_heis = vbs_result.m_vbs_heis;
                results.D_mean = vbs_result.D_mean_xy;
                results.D_mean_xy = vbs_result.D_mean_xy;
                results.D_mean_heis = vbs_result.D_mean_heis;
                results.m_plaquette = plaq_result.m_plaquette;
                results.P_mean = plaq_result.P_mean;
                results.resonance_strength = plaq_result.resonance_strength;
                results.chi_mean = plaq_result.chi_mean;
                results.n_plaquettes = plaq_result.n_plaquettes;
                results.n_triangles = plaq_result.n_triangles;
                results.D_mean_xy = vbs_result.D_mean_xy;
                results.D_mean_heis = vbs_result.D_mean_heis;
            } else {
                // Quick scalar-only computation
                results = compute_all_order_parameters(psi, cluster, jpm);
                results.temperature = temperature;  // Preserve temperature from TPQ
            }
            
            all_results[i] = results;
            
            int done = ++completed;
            {
                std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
                std::cout << "[Rank " << mpi_rank << "] ";
#endif
                std::cout << "[" << done << "/" << jpm_dirs.size() << "] "
                          << "Jpm=" << std::fixed << std::setprecision(4) << jpm;
                if (use_tpq) {
                    std::cout << " T=" << std::setprecision(6) << results.temperature;
                }
                std::cout << " | m_trans=" << std::setprecision(6) << results.m_translation
                          << " | m_nem=" << results.m_nematic
                          << " | m_vbs_xy=" << results.m_vbs_xy
                          << " | m_vbs_heis=" << results.m_vbs_heis
                          << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
            std::cerr << "[Rank " << mpi_rank << "] ";
#endif
            std::cerr << "Error processing " << dir << ": " << e.what() << std::endl;
        }
    }
    
#ifdef USE_MPI
    // Gather results from all ranks to rank 0
    if (mpi_rank == 0) {
        std::cout << "\nGathering results from all MPI ranks..." << std::endl;
    }
    
    // Create a flat buffer for MPI communication (just the scalar values)
    std::vector<double> local_buffer;
    std::vector<int> local_indices;
    
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
        if (static_cast<int>(i % mpi_size) == mpi_rank && all_results[i].is_valid()) {
            local_indices.push_back(i);
            local_buffer.push_back(all_results[i].jpm);
            local_buffer.push_back(all_results[i].temperature);
            local_buffer.push_back(all_results[i].m_translation);
            local_buffer.push_back(all_results[i].m_nematic);
            local_buffer.push_back(all_results[i].m_nematic_spsm);
            local_buffer.push_back(all_results[i].m_nematic_szsz);
            local_buffer.push_back(all_results[i].m_nematic_heisenberg);
            local_buffer.push_back(all_results[i].m_vbs);
            local_buffer.push_back(all_results[i].m_vbs_xy);
            local_buffer.push_back(all_results[i].m_vbs_heis);
            local_buffer.push_back(all_results[i].anisotropy);
            local_buffer.push_back(all_results[i].D_mean);
            local_buffer.push_back(all_results[i].D_mean_xy);
            local_buffer.push_back(all_results[i].D_mean_heis);
        }
    }
    
    // Gather all results to rank 0
    std::vector<int> recv_counts(mpi_size);
    int local_count = local_buffer.size();
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> displs;
    std::vector<double> recv_buffer;
    if (mpi_rank == 0) {
        displs.resize(mpi_size);
        int total = 0;
        for (int i = 0; i < mpi_size; ++i) {
            displs[i] = total;
            total += recv_counts[i];
        }
        recv_buffer.resize(total);
    }
    
    MPI_Gatherv(local_buffer.data(), local_count, MPI_DOUBLE,
                recv_buffer.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Gather indices
    std::vector<int> all_indices;
    std::vector<int> index_counts(mpi_size);
    int local_idx_count = local_indices.size();
    MPI_Gather(&local_idx_count, 1, MPI_INT, index_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> idx_displs;
    if (mpi_rank == 0) {
        idx_displs.resize(mpi_size);
        int total = 0;
        for (int i = 0; i < mpi_size; ++i) {
            idx_displs[i] = total;
            total += index_counts[i];
        }
        all_indices.resize(total);
    }
    
    MPI_Gatherv(local_indices.data(), local_idx_count, MPI_INT,
                all_indices.data(), index_counts.data(), idx_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Rank 0 unpacks the results
    if (mpi_rank == 0) {
        const int vals_per_result = 14;
        for (size_t i = 0; i < all_indices.size(); ++i) {
            int idx = all_indices[i];
            size_t offset = i * vals_per_result;
            all_results[idx].jpm = recv_buffer[offset + 0];
            all_results[idx].temperature = recv_buffer[offset + 1];
            all_results[idx].m_translation = recv_buffer[offset + 2];
            all_results[idx].m_nematic = recv_buffer[offset + 3];
            all_results[idx].m_nematic_spsm = recv_buffer[offset + 4];
            all_results[idx].m_nematic_szsz = recv_buffer[offset + 5];
            all_results[idx].m_nematic_heisenberg = recv_buffer[offset + 6];
            all_results[idx].m_vbs = recv_buffer[offset + 7];
            all_results[idx].m_vbs_xy = recv_buffer[offset + 8];
            all_results[idx].m_vbs_heis = recv_buffer[offset + 9];
            all_results[idx].anisotropy = recv_buffer[offset + 10];
            all_results[idx].D_mean = recv_buffer[offset + 11];
            all_results[idx].D_mean_xy = recv_buffer[offset + 12];
            all_results[idx].D_mean_heis = recv_buffer[offset + 13];
        }
    }
#endif
    
    return all_results;
}

// `save_scan_results` now lives in `ed_bfg::results_io`. The
// using-declaration at the top of this TU keeps the call site below
// unchanged. P2.1 (7th slice).

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  Single file mode:\n"
              << "    " << prog << " <wavefunction.h5> <cluster_dir> [output.h5]\n\n"
              << "  Scan directory mode:\n"
              << "    " << prog << " --scan-dir <dir> [options]\n\n"
              << "Options:\n"
              << "  --scan-dir <dir>     Directory containing Jpm=* subdirectories\n"
              << "  --output-dir <dir>   Output directory for results\n"
              << "  --n-workers <n>      Number of parallel workers (default: 4)\n"
              << "  --n-q-grid <n>       2D q-grid size for visualization (default: 50)\n"
              << "  --save-full          Save full S(q), S_D(q), S_P(q) 2D grids and bond-resolved data per Jpm\n"
              << "  --tpq                Use TPQ states (lowest temperature) instead of ground state\n"
              << "  --tpq-all-temps      Process ALL temperatures from TPQ data (creates T-dependent output)\n"
              << "\nComputes BFG order parameters from ground state or TPQ wavefunction:\n"
              << "  1. S(q) - Spin structure factor (translation order)\n"
              << "  2. Nematic order - Bond orientation anisotropy (C3 breaking)\n"
              << "     - Variants: XY, S+S-, SzSz, Heisenberg\n"
              << "  3. VBS order - Valence bond solid with proper 4-site dimer correlations\n"
              << "     - Variants: XY dimer (S+S- + S-S+), Heisenberg dimer (S·S)\n"
              << "     - Bond-resolved: full dimer-dimer correlation matrices\n"
              << "  4. Plaquette/bowtie resonance order - BFG-native 4-spin ring flip\n"
              << "     - S_P(q) structure factor for bowtie plaquettes\n"
              << "     - Triangle chiral order\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif
    
    if (argc < 2) {
        if (mpi_rank == 0) {
            print_usage(argv[0]);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    std::string scan_dir, output_dir;
    std::string wf_file, cluster_dir, output_file = "bfg_order_parameters.h5";
    int n_workers = 4;
    int n_q_grid = 50;
    bool scan_mode = false;
    bool save_full = false;
    bool use_tpq = false;
    bool tpq_all_temps = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scan-dir" && i + 1 < argc) {
            scan_dir = argv[++i];
            scan_mode = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--n-workers" && i + 1 < argc) {
            n_workers = std::stoi(argv[++i]);
        } else if (arg == "--n-q-grid" && i + 1 < argc) {
            n_q_grid = std::stoi(argv[++i]);
        } else if (arg == "--save-full") {
            save_full = true;
        } else if (arg == "--tpq") {
            use_tpq = true;
        } else if (arg == "--tpq-all-temps") {
            use_tpq = true;
            tpq_all_temps = true;
        } else if (arg == "-h" || arg == "--help") {
            if (mpi_rank == 0) {
                print_usage(argv[0]);
            }
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else if (wf_file.empty()) {
            wf_file = arg;
        } else if (cluster_dir.empty()) {
            cluster_dir = arg;
        } else {
            output_file = arg;
        }
    }
    
    #ifdef _OPENMP
    if (mpi_rank == 0) {
        std::cout << "OpenMP enabled with " << omp_get_max_threads() << " max threads" << std::endl;
    }
    #endif
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (scan_mode) {
            // Scan directory mode
            if (scan_dir.empty()) {
                if (mpi_rank == 0) {
                    std::cerr << "Error: --scan-dir required in scan mode" << std::endl;
                }
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
            if (output_dir.empty()) {
                output_dir = scan_dir + "/order_parameter_results";
            }
            
            // Only rank 0 creates output directory
            if (mpi_rank == 0) {
                fs::create_directories(output_dir);
                
                std::string mode_str;
                if (tpq_all_temps) {
                    mode_str = "TPQ (all temperatures)";
                } else if (use_tpq) {
                    mode_str = "TPQ (lowest temperature)";
                } else {
                    mode_str = "Ground state";
                }
                
                std::cout << "========================================\n"
                          << "BFG ORDER PARAMETER SCAN (CPU)\n"
                          << "========================================\n"
                          << "Scan directory: " << scan_dir << "\n"
                          << "Output directory: " << output_dir << "\n"
                          << "Workers: " << n_workers << "\n"
                          << "Mode: " << mode_str << "\n"
                          << "Save full: " << (save_full ? "yes (2D grids)" : "no (scalars only)") << "\n"
                          << "========================================" << std::endl;
            }
            
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);  // Wait for directory creation
#endif
            
            auto results = scan_jpm_directories(scan_dir, output_dir, n_workers, n_q_grid, save_full, use_tpq, tpq_all_temps);
            
            // Only rank 0 saves the combined results
            if (mpi_rank == 0 && !results.empty()) {
                std::string results_file = output_dir + "/scan_results.h5";
                save_scan_results(results, results_file);
            }
            
        } else {
            // Single file mode - only run on rank 0
#ifdef USE_MPI
            if (mpi_rank != 0) {
                MPI_Finalize();
                return 0;  // Other ranks exit
            }
#endif
            
            if (wf_file.empty() || cluster_dir.empty()) {
                print_usage(argv[0]);
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
            
            std::cout << "========================================\n"
                      << "BFG ORDER PARAMETER COMPUTATION (CPU)\n"
                      << "========================================" << std::endl;
            
            Cluster cluster = load_cluster(cluster_dir);
            
            std::vector<Complex> psi;
            double temperature = 0.0;
            
            if (use_tpq) {
                auto [tpq_psi, tpq_temp] = load_tpq_state(wf_file);
                psi = std::move(tpq_psi);
                temperature = tpq_temp;
                std::cout << "Using TPQ state (T=" << temperature << ")" << std::endl;
            } else {
                psi = load_wavefunction(wf_file);
                std::cout << "Using ground state wavefunction" << std::endl;
            }
            
            // Set memory-efficient mode based on Hilbert space size
            set_memory_efficient_mode(psi.size());
            
            // Verify size
            uint64_t expected_size = 1ULL << cluster.n_sites;
            if (psi.size() != expected_size) {
                std::cerr << "Warning: wavefunction size " << psi.size() 
                          << " != expected 2^" << cluster.n_sites << " = " << expected_size << std::endl;
            }
            
            // Compute correlations
            auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
            auto szsz_corr = compute_szsz_correlations(psi, cluster.n_sites);
            
            // Compute S(q) at k-points (full Heisenberg = SzSz + (1/2)(S+S- + S-S+))
            auto sf_result = compute_spin_structure_factor(smsp_corr, szsz_corr, cluster);
            
            // Compute 2D S(q) grid for visualization
            auto s_q_2d = compute_sq_2d_grid(smsp_corr, szsz_corr, cluster, n_q_grid);
            
            // Compute bond expectations for nematic and VBS
            auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
            
            // Compute additional bond operators for visualization
            auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
            auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
            auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
            
            // Compute nematic order for all three bond types
            auto nem_xy_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
            auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
            auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
            auto nem_heisenberg_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
            
            // Use XY nematic as primary (for backwards compatibility)
            auto nem_result = nem_xy_result;
            
            // VBS order with proper 4-site correlations (with 2D grid)
            auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster, n_q_grid);
            
            // Plaquette/bowtie resonance order
            auto plaq_result = compute_plaquette_order(psi, cluster, n_q_grid);
            
            // Save results with full 2D grids and bond data
            save_results(output_file, sf_result, nem_result, nem_spsm_result, nem_szsz_result, 
                        nem_heisenberg_result, vbs_result, plaq_result,
                        cluster, s_q_2d, n_q_grid,
                        spsm_bond_exp, szsz_bond_exp, heisenberg_bond_exp);
            
            // Print summary
            std::cout << "\n========== ORDER PARAMETER SUMMARY ==========" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Translation order:  m = " << sf_result.m_translation 
                      << " at q = (" << sf_result.q_max[0] << ", " << sf_result.q_max[1] << ")" << std::endl;
            std::cout << "Nematic order (XY):         m = " << nem_result.m_nem 
                      << ", anisotropy = " << nem_result.anisotropy << std::endl;
            std::cout << "Nematic order (S+S-):       m = " << nem_spsm_result.m_nem 
                      << ", anisotropy = " << nem_spsm_result.anisotropy << std::endl;
            std::cout << "Nematic order (SzSz):       m = " << nem_szsz_result.m_nem 
                      << ", anisotropy = " << nem_szsz_result.anisotropy << std::endl;
            std::cout << "Nematic order (Heisenberg): m = " << nem_heisenberg_result.m_nem 
                      << ", anisotropy = " << nem_heisenberg_result.anisotropy << std::endl;
            std::cout << "VBS order (XY, 4-site):         m = " << vbs_result.m_vbs_xy 
                      << " at q = (" << vbs_result.q_max_xy[0] << ", " << vbs_result.q_max_xy[1] << ")" << std::endl;
            std::cout << "VBS order (Heisenberg, 4-site): m = " << vbs_result.m_vbs_heis 
                      << " at q = (" << vbs_result.q_max_heis[0] << ", " << vbs_result.q_max_heis[1] << ")" << std::endl;
            std::cout << "Plaquette order (bowtie):   m = " << plaq_result.m_plaquette 
                      << " at q = (" << plaq_result.q_max[0] << ", " << plaq_result.q_max[1] << ")" << std::endl;
            std::cout << "  Resonance: <P> = " << plaq_result.P_mean 
                      << ", strength = " << plaq_result.resonance_strength << std::endl;
            std::cout << "  Triangle chiral: <χ> = " << plaq_result.chi_mean << std::endl;
            std::cout << "==============================================" << std::endl;
        }
        
    } catch (const std::exception& e) {
        if (mpi_rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    if (mpi_rank == 0) {
        std::cout << "\nTotal runtime: " << total_duration.count() << " seconds" << std::endl;
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}
