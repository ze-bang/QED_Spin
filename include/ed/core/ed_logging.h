#ifndef ED_LOGGING_H
#define ED_LOGGING_H

/**
 * @file ed_logging.h
 * @brief Centralized, clean logging for ED calculations
 * 
 * Provides consistent, professional output formatting across all ED operations.
 * All output goes through this module for uniform appearance.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cmath>

namespace ed_log {

// ============================================================================
// CONFIGURATION
// ============================================================================

// Verbosity levels
enum class Verbosity {
    SILENT = 0,    // No output
    MINIMAL = 1,   // Only final results
    NORMAL = 2,    // Key progress updates
    VERBOSE = 3    // Detailed debug info
};

// Global verbosity setting (can be modified at runtime)
inline Verbosity& verbosity() {
    static Verbosity v = Verbosity::NORMAL;
    return v;
}

inline void setVerbosity(Verbosity v) { verbosity() = v; }
inline bool isVerbose() { return verbosity() >= Verbosity::VERBOSE; }
inline bool isNormal() { return verbosity() >= Verbosity::NORMAL; }
inline bool isMinimal() { return verbosity() >= Verbosity::MINIMAL; }

// ============================================================================
// FORMATTING UTILITIES
// ============================================================================

// Standard width for headers
constexpr int HEADER_WIDTH = 60;

/**
 * @brief Print a section header
 */
inline void header(const std::string& title) {
    if (!isMinimal()) return;
    std::cout << "\n" << std::string(HEADER_WIDTH, '=') << "\n";
    // Center the title
    int padding = (HEADER_WIDTH - title.length()) / 2;
    if (padding > 0) std::cout << std::string(padding, ' ');
    std::cout << title << "\n";
    std::cout << std::string(HEADER_WIDTH, '=') << "\n";
}

/**
 * @brief Print a subsection header
 */
inline void subheader(const std::string& title) {
    if (!isNormal()) return;
    std::cout << "\n--- " << title << " ---\n";
}

/**
 * @brief Print a key-value pair
 */
template<typename T>
inline void info(const std::string& key, const T& value) {
    if (!isNormal()) return;
    std::cout << "  " << std::left << std::setw(24) << (key + ":") << value << "\n";
}

/**
 * @brief Print a key-value pair with units
 */
template<typename T>
inline void info(const std::string& key, const T& value, const std::string& units) {
    if (!isNormal()) return;
    std::cout << "  " << std::left << std::setw(24) << (key + ":") << value << " " << units << "\n";
}

/**
 * @brief Print a progress message
 */
inline void progress(const std::string& message) {
    if (!isNormal()) return;
    std::cout << "  → " << message << "\n";
}

/**
 * @brief Print a success message
 */
inline void success(const std::string& message) {
    if (!isMinimal()) return;
    std::cout << "  ✓ " << message << "\n";
}

/**
 * @brief Print a warning message
 */
inline void warning(const std::string& message) {
    std::cerr << "  ⚠ Warning: " << message << "\n";
}

/**
 * @brief Print an error message
 */
inline void error(const std::string& message) {
    std::cerr << "  ✗ Error: " << message << "\n";
}

/**
 * @brief Print verbose debug info
 */
inline void debug(const std::string& message) {
    if (!isVerbose()) return;
    std::cout << "  [DEBUG] " << message << "\n";
}

/**
 * @brief Print completion summary
 */
inline void complete(const std::string& title, double duration_sec) {
    if (!isMinimal()) return;
    std::cout << "\n" << std::string(HEADER_WIDTH, '-') << "\n";
    std::cout << title << " complete";
    if (duration_sec > 0) {
        if (duration_sec < 60) {
            std::cout << " (" << std::fixed << std::setprecision(2) << duration_sec << " s)";
        } else if (duration_sec < 3600) {
            std::cout << " (" << std::fixed << std::setprecision(1) << duration_sec/60 << " min)";
        } else {
            std::cout << " (" << std::fixed << std::setprecision(2) << duration_sec/3600 << " h)";
        }
    }
    std::cout << "\n" << std::string(HEADER_WIDTH, '-') << "\n";
}

// ============================================================================
// FORMATTED NUMBER PRINTING
// ============================================================================

/**
 * @brief Format a large number with commas (e.g., 1,234,567)
 */
inline std::string formatNumber(uint64_t n) {
    std::string s = std::to_string(n);
    int insertPosition = s.length() - 3;
    while (insertPosition > 0) {
        s.insert(insertPosition, ",");
        insertPosition -= 3;
    }
    return s;
}

/**
 * @brief Format bytes as human-readable (KB, MB, GB)
 */
inline std::string formatBytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = bytes;
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(unit > 0 ? 1 : 0) << size << " " << units[unit];
    return oss.str();
}

/**
 * @brief Format scientific notation
 */
inline std::string formatSci(double val, int precision = 6) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << val;
    return oss.str();
}

// ============================================================================
// TIMER UTILITY
// ============================================================================

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }
    
    std::string elapsedStr() const {
        double t = elapsed();
        std::ostringstream oss;
        if (t < 60) {
            oss << std::fixed << std::setprecision(2) << t << " s";
        } else if (t < 3600) {
            oss << std::fixed << std::setprecision(1) << t/60 << " min";
        } else {
            oss << std::fixed << std::setprecision(2) << t/3600 << " h";
        }
        return oss.str();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// DIAGONALIZATION OUTPUT HELPERS
// ============================================================================

/**
 * @brief Print eigenvalue summary
 */
inline void printEigenvalues(const std::vector<double>& eigenvalues, uint64_t max_show = 5) {
    if (!isNormal() || eigenvalues.empty()) return;
    
    subheader("Eigenvalues");
    
    size_t show = std::min(eigenvalues.size(), (size_t)max_show);
    for (size_t i = 0; i < show; i++) {
        std::cout << "  E[" << std::setw(3) << i << "] = " 
                  << std::fixed << std::setprecision(10) << eigenvalues[i] << "\n";
    }
    if (eigenvalues.size() > max_show) {
        std::cout << "  ... (" << (eigenvalues.size() - max_show) << " more eigenvalues)\n";
    }
    
    // Gap to ground state
    if (eigenvalues.size() > 1) {
        double gap = eigenvalues[1] - eigenvalues[0];
        std::cout << "  Δ (gap) = " << std::fixed << std::setprecision(10) << gap << "\n";
    }
}

/**
 * @brief Print system info at start of calculation
 */
inline void printSystemInfo(uint64_t num_sites, float spin, uint64_t hilbert_dim, 
                            bool fixed_sz = false, int64_t n_up = -1) {
    if (!isNormal()) return;
    
    subheader("System Configuration");
    info("Sites", num_sites);
    info("Spin", spin);
    
    std::string dim_str = formatNumber(hilbert_dim);
    double mem_per_vec = hilbert_dim * 16.0;  // Complex double
    info("Hilbert dimension", dim_str);
    info("Memory per vector", formatBytes(mem_per_vec));
    
    if (fixed_sz) {
        info("Fixed Sz sector", "enabled");
        if (n_up >= 0) {
            double sz = n_up - num_sites / 2.0;
            std::ostringstream oss;
            oss << n_up << " up (Sz = " << sz << ")";
            info("Sector", oss.str());
        }
    }
}

/**
 * @brief Print method info
 */
inline void printMethodInfo(const std::string& method, uint64_t num_eigenvalues,
                            double tolerance, uint64_t max_iter) {
    if (!isNormal()) return;
    
    subheader("Diagonalization Method");
    info("Method", method);
    info("Target eigenvalues", num_eigenvalues);
    info("Tolerance", formatSci(tolerance));
    info("Max iterations", max_iter);
}

/**
 * @brief Print convergence info
 */
inline void printConvergence(uint64_t iterations, double residual, bool converged) {
    if (!isNormal()) return;
    
    if (converged) {
        success("Converged in " + std::to_string(iterations) + " iterations (residual = " + formatSci(residual) + ")");
    } else {
        warning("Did not converge after " + std::to_string(iterations) + " iterations (residual = " + formatSci(residual) + ")");
    }
}

} // namespace ed_log

#endif // ED_LOGGING_H
