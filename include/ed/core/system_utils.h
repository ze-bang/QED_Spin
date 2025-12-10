#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <cstdlib>
#include <string>
#include <iostream>
#include <filesystem>
#include <stdexcept>

#ifdef WITH_MPI
#include <mpi.h>
#endif

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

#endif // SYSTEM_UTILS_H
