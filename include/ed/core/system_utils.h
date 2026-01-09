#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <cstdlib>
#include <string>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <functional>

#ifdef WITH_MPI
#include <mpi.h>
#endif

// ============================================================================
// FILE HASHING UTILITIES
// ============================================================================

/**
 * @brief Compute a simple hash of a file's contents and modification time
 * 
 * This is used for cache invalidation - if the hash changes, cached data
 * based on this file should be regenerated.
 * 
 * @param filepath Path to the file
 * @return Hash string, or empty string if file doesn't exist
 */
inline std::string compute_file_hash(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        return "";
    }
    
    // Use file size + modification time as a quick hash
    // This is fast and sufficient for detecting changes
    auto file_size = std::filesystem::file_size(filepath);
    auto mod_time = std::filesystem::last_write_time(filepath);
    auto mod_time_t = mod_time.time_since_epoch().count();
    
    std::ostringstream oss;
    oss << file_size << "_" << mod_time_t;
    return oss.str();
}

/**
 * @brief Compute combined hash of multiple files
 * 
 * Used for validating that cached symmetry data matches current Hamiltonian files.
 * 
 * @param filepaths Vector of file paths to hash
 * @return Combined hash string
 */
inline std::string compute_combined_hash(const std::vector<std::string>& filepaths) {
    std::ostringstream oss;
    for (const auto& fp : filepaths) {
        oss << compute_file_hash(fp) << ";";
    }
    return oss.str();
}

/**
 * @brief Read a hash from a cache marker file
 * @param marker_file Path to the marker file
 * @return Hash string, or empty string if file doesn't exist
 */
inline std::string read_cache_marker(const std::string& marker_file) {
    if (!std::filesystem::exists(marker_file)) {
        return "";
    }
    std::ifstream ifs(marker_file);
    std::string hash;
    std::getline(ifs, hash);
    return hash;
}

/**
 * @brief Write a hash to a cache marker file
 * @param marker_file Path to the marker file
 * @param hash Hash string to write
 */
inline void write_cache_marker(const std::string& marker_file, const std::string& hash) {
    std::ofstream ofs(marker_file);
    ofs << hash << std::endl;
}

/**
 * @brief Check if symmetry cache is valid for given Hamiltonian files
 * 
 * Validates that:
 * 1. The symmetry data file exists
 * 2. The automorphism results exist
 * 3. The cache marker matches current Hamiltonian file hashes
 * 
 * @param directory Hamiltonian directory
 * @param symmetry_file Name of symmetry HDF5 file (e.g., "symmetry_data.h5")
 * @return true if cache is valid, false if regeneration is needed
 */
inline bool is_symmetry_cache_valid(const std::string& directory, 
                                    const std::string& symmetry_file = "symmetry_data.h5") {
    std::string hdf5_path = directory + "/" + symmetry_file;
    std::string automorphism_file = directory + "/automorphism_results/automorphisms.json";
    std::string cache_marker = directory + "/" + symmetry_file + ".marker";
    
    // Check if all required files exist
    if (!std::filesystem::exists(hdf5_path)) {
        return false;
    }
    if (!std::filesystem::exists(automorphism_file)) {
        return false;
    }
    
    // Compute hash of source files (Hamiltonian + automorphisms)
    std::vector<std::string> source_files = {
        directory + "/InterAll.dat",
        directory + "/Trans.dat",
        automorphism_file
    };
    
    std::string current_hash = compute_combined_hash(source_files);
    std::string cached_hash = read_cache_marker(cache_marker);
    
    if (current_hash != cached_hash) {
        std::cout << "Cache invalidated: source files have changed" << std::endl;
        return false;
    }
    
    return true;
}

/**
 * @brief Mark symmetry cache as valid for current Hamiltonian files
 * 
 * Call this after successfully generating symmetry data.
 * 
 * @param directory Hamiltonian directory
 * @param symmetry_file Name of symmetry HDF5 file
 */
inline void mark_symmetry_cache_valid(const std::string& directory,
                                      const std::string& symmetry_file = "symmetry_data.h5") {
    std::string automorphism_file = directory + "/automorphism_results/automorphisms.json";
    std::string cache_marker = directory + "/" + symmetry_file + ".marker";
    
    std::vector<std::string> source_files = {
        directory + "/InterAll.dat",
        directory + "/Trans.dat",
        automorphism_file
    };
    
    std::string hash = compute_combined_hash(source_files);
    write_cache_marker(cache_marker, hash);
}

/**
 * @brief Invalidate symmetry cache (forces regeneration on next run)
 * 
 * @param directory Hamiltonian directory
 * @param symmetry_file Name of symmetry HDF5 file
 */
inline void invalidate_symmetry_cache(const std::string& directory,
                                      const std::string& symmetry_file = "symmetry_data.h5") {
    std::string hdf5_path = directory + "/" + symmetry_file;
    std::string cache_marker = directory + "/" + symmetry_file + ".marker";
    
    // Remove both the data and marker files
    if (std::filesystem::exists(hdf5_path)) {
        std::filesystem::remove(hdf5_path);
    }
    if (std::filesystem::exists(cache_marker)) {
        std::filesystem::remove(cache_marker);
    }
}

// ============================================================================
// SYSTEM UTILITIES  
// ============================================================================

/**
 * @brief Execute a system command and check for errors
 * @param cmd Command to execute
 * @return true if command succeeded, false otherwise
 */
inline bool safe_system_call(const std::string& cmd) {
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Warning: System command failed with code " << result << ": " << cmd << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Create a directory, throwing on failure
 * @param path Directory path to create
 * @throws std::runtime_error if directory creation fails
 */
inline void create_directory_or_throw(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::create_directories(path, ec) && !std::filesystem::exists(path)) {
        throw std::runtime_error("Failed to create directory: " + path + " (" + ec.message() + ")");
    }
}

/**
 * @brief Create a directory with MPI-safe synchronization
 * Only rank 0 creates the directory, then all ranks synchronize.
 * @param path Directory path to create
 * @throws std::runtime_error if directory creation fails
 */
inline void create_directory_mpi_safe(const std::string& path) {
    int rank = 0;
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    bool success = true;
    std::string error_msg;
    
    if (rank == 0) {
        std::error_code ec;
        if (!std::filesystem::create_directories(path, ec) && !std::filesystem::exists(path)) {
            success = false;
            error_msg = "Failed to create directory: " + path + " (" + ec.message() + ")";
        }
    }
    
#ifdef WITH_MPI
    // Broadcast success status to all ranks
    int success_int = success ? 1 : 0;
    MPI_Bcast(&success_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    success = (success_int != 0);
    
    // If failed, broadcast error message length and content
    if (!success) {
        int msg_len = static_cast<int>(error_msg.size());
        MPI_Bcast(&msg_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            error_msg.resize(msg_len);
        }
        MPI_Bcast(&error_msg[0], msg_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    
    // Synchronize before continuing
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    if (!success) {
        throw std::runtime_error(error_msg);
    }
}

/**
 * @brief Ensure a directory exists, creating it if necessary (legacy compatibility wrapper)
 * @param path Directory path to ensure exists
 * @return true if directory exists or was created, false otherwise
 */
inline bool ensure_directory_exists(const std::string& path) {
    try {
        create_directory_or_throw(path);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/**
 * @brief Find the automorphism finder Python script
 * 
 * Searches for the script in common locations relative to the source tree.
 * The script is typically located at python/edlib/automorphism_finder.py
 * 
 * @return Path to the automorphism finder script, or empty string if not found
 */
inline std::string find_automorphism_finder() {
    // Get path relative to this header file (system_utils.h is in include/ed/core/)
    std::string header_path = __FILE__;
    
    // Navigate up from include/ed/core to project root
    std::filesystem::path header_fs_path(header_path);
    std::filesystem::path project_root = header_fs_path.parent_path()   // ed/core
                                                       .parent_path()   // ed
                                                       .parent_path()   // include
                                                       .parent_path();  // project root
    
    // Primary location: python/edlib/automorphism_finder.py
    std::filesystem::path primary_path = project_root / "python" / "edlib" / "automorphism_finder.py";
    if (std::filesystem::exists(primary_path)) {
        return primary_path.string();
    }
    
    // Legacy location: include/ed/core/automorphism_finder.py
    std::filesystem::path legacy_path = project_root / "include" / "ed" / "core" / "automorphism_finder.py";
    if (std::filesystem::exists(legacy_path)) {
        return legacy_path.string();
    }
    
    // Fallback: check if it's in PATH or current directory
    if (std::filesystem::exists("automorphism_finder.py")) {
        return "automorphism_finder.py";
    }
    
    return "";
}

/**
 * @brief Generate automorphisms for a Hamiltonian directory
 * 
 * Runs the automorphism finder script to generate symmetry information
 * required for symmetrized diagonalization.
 * 
 * @param directory Directory containing Hamiltonian files (InterAll.dat, Trans.dat)
 * @return true if automorphisms were generated successfully, false otherwise
 */
inline bool generate_automorphisms(const std::string& directory) {
    // Check if automorphisms already exist
    std::string automorphism_file = directory + "/automorphism_results/automorphisms.json";
    if (std::filesystem::exists(automorphism_file)) {
        std::cout << "Using existing automorphisms from: " << automorphism_file << std::endl;
        return true;
    }
    
    // Find the script
    std::string finder_path = find_automorphism_finder();
    if (finder_path.empty()) {
        std::cerr << "Error: Could not find automorphism_finder.py" << std::endl;
        std::cerr << "Expected locations:" << std::endl;
        std::cerr << "  - python/edlib/automorphism_finder.py (relative to project root)" << std::endl;
        return false;
    }
    
    // Run the script
    std::cout << "Generating automorphisms..." << std::endl;
    std::string cmd = "python3 \"" + finder_path + "\" --data_dir=\"" + directory + "\"";
    std::cout << "Running: " << cmd << std::endl;
    
    if (!safe_system_call(cmd)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return false;
    }
    
    // Verify that the output was created
    if (!std::filesystem::exists(automorphism_file)) {
        std::cerr << "Warning: Automorphism file was not created: " << automorphism_file << std::endl;
        return false;
    }
    
    return true;
}

#endif // SYSTEM_UTILS_H
