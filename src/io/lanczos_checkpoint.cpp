// =============================================================================
// src/io/lanczos_checkpoint.cpp
//
// HDF5-backed Krylov-state checkpoint / restart for the default Lanczos
// solver. See include/ed/io/lanczos_checkpoint.h for the schema and the
// rationale (Phase 3a #1 from docs/architecture/SCALING.md / docs/history/MODERNIZATION_AUDIT.md §9).
// =============================================================================

#include "ed/io/lanczos_checkpoint.h"

#include <H5Cpp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace lanczos_io {

namespace {

constexpr uint32_t kSchemaVersion = 1;
constexpr const char* kSolverTag = "lanczos";
constexpr const char* kCheckpointBaseName = "lanczos_checkpoint.h5";

// ---------------------------------------------------------------------------
// Small env helpers
// ---------------------------------------------------------------------------

std::string env_string(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// HDF5 compound type for Complex == std::complex<double>, member names
// (real, imag) to match the legacy ED writer convention used elsewhere
// in src/bfg/wavefunction_io.cpp and include/ed/core/hdf5_io.h.
H5::CompType make_complex_compound() {
    H5::CompType ct(sizeof(Complex));
    ct.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
    ct.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);
    return ct;
}

// ---------------------------------------------------------------------------
// Scalar / 1-D dataset helpers
// ---------------------------------------------------------------------------

template <typename T>
void write_scalar(H5::Group& g, const char* name, const T& value,
                  const H5::DataType& dt) {
    H5::DataSpace space(H5S_SCALAR);
    H5::DataSet ds = g.createDataSet(name, dt, space);
    ds.write(&value, dt);
}

void write_string_scalar(H5::Group& g, const char* name,
                         const std::string& value) {
    H5::StrType st(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace space(H5S_SCALAR);
    H5::DataSet ds = g.createDataSet(name, st, space);
    const char* cstr = value.c_str();
    ds.write(&cstr, st);
}

template <typename T>
T read_scalar(H5::Group& g, const char* name, const H5::DataType& dt) {
    T value{};
    H5::DataSet ds = g.openDataSet(name);
    ds.read(&value, dt);
    return value;
}

std::string read_string_scalar(H5::Group& g, const char* name) {
    H5::DataSet ds = g.openDataSet(name);
    H5::StrType st = ds.getStrType();
    std::string out;
    if (st.isVariableStr()) {
        char* tmp = nullptr;
        ds.read(&tmp, st);
        if (tmp) {
            out.assign(tmp);
            // HDF5 allocated; free via the standard C runtime free() that
            // libhdf5 uses for its variable-length string buffers.
            std::free(tmp);
        }
    } else {
        // Fixed-length string: size known.
        const std::size_t len = st.getSize();
        out.assign(len, '\0');
        ds.read(&out[0], st);
        // Trim trailing NULs / spaces.
        while (!out.empty() && (out.back() == '\0' || out.back() == ' ')) {
            out.pop_back();
        }
    }
    return out;
}

template <typename T>
void write_1d(H5::Group& g, const char* name, const std::vector<T>& v,
              const H5::DataType& dt) {
    hsize_t dim = static_cast<hsize_t>(v.size());
    H5::DataSpace space(1, &dim);
    H5::DataSet ds = g.createDataSet(name, dt, space);
    if (dim > 0) {
        ds.write(v.data(), dt);
    }
}

template <typename T>
std::vector<T> read_1d(H5::Group& g, const char* name, const H5::DataType& dt) {
    H5::DataSet ds = g.openDataSet(name);
    H5::DataSpace space = ds.getSpace();
    int rank = space.getSimpleExtentNdims();
    if (rank != 1) {
        throw std::runtime_error(std::string("read_1d(\"") + name +
                                 "\"): expected rank 1, got rank " +
                                 std::to_string(rank));
    }
    hsize_t dim = 0;
    space.getSimpleExtentDims(&dim);
    std::vector<T> out(static_cast<std::size_t>(dim));
    if (dim > 0) {
        ds.read(out.data(), dt);
    }
    return out;
}

void write_complex_1d(H5::Group& g, const char* name, const ComplexVector& v) {
    H5::CompType ct = make_complex_compound();
    write_1d(g, name, v, ct);
}

ComplexVector read_complex_1d(H5::Group& g, const char* name) {
    H5::CompType ct = make_complex_compound();
    return read_1d<Complex>(g, name, ct);
}

void write_complex_2d(H5::Group& g, const char* name,
                      const std::vector<ComplexVector>& rows, hsize_t row_len) {
    H5::CompType ct = make_complex_compound();
    hsize_t dims[2] = {static_cast<hsize_t>(rows.size()), row_len};
    H5::DataSpace space(2, dims);
    H5::DataSet ds = g.createDataSet(name, ct, space);
    if (dims[0] == 0 || dims[1] == 0) {
        return;  // empty dataset is fine on its own
    }
    // Pack rows into a contiguous buffer.
    std::vector<Complex> flat(static_cast<std::size_t>(dims[0]) *
                              static_cast<std::size_t>(dims[1]));
    for (hsize_t r = 0; r < dims[0]; ++r) {
        if (rows[r].size() != static_cast<std::size_t>(row_len)) {
            throw std::runtime_error(std::string("write_complex_2d(\"") + name +
                                     "\"): row length mismatch");
        }
        std::memcpy(flat.data() + r * row_len, rows[r].data(),
                    sizeof(Complex) * row_len);
    }
    ds.write(flat.data(), ct);
}

std::vector<ComplexVector> read_complex_2d(H5::Group& g, const char* name,
                                           hsize_t expected_row_len) {
    H5::CompType ct = make_complex_compound();
    H5::DataSet ds = g.openDataSet(name);
    H5::DataSpace space = ds.getSpace();
    int rank = space.getSimpleExtentNdims();
    if (rank != 2) {
        throw std::runtime_error(std::string("read_complex_2d(\"") + name +
                                 "\"): expected rank 2");
    }
    hsize_t dims[2] = {0, 0};
    space.getSimpleExtentDims(dims);
    if (expected_row_len != 0 && dims[1] != expected_row_len) {
        throw std::runtime_error(std::string("read_complex_2d(\"") + name +
                                 "\"): row length " + std::to_string(dims[1]) +
                                 " != expected " +
                                 std::to_string(expected_row_len));
    }
    std::vector<ComplexVector> out(static_cast<std::size_t>(dims[0]));
    if (dims[0] == 0 || dims[1] == 0) {
        for (auto& row : out) row.resize(static_cast<std::size_t>(dims[1]));
        return out;
    }
    std::vector<Complex> flat(static_cast<std::size_t>(dims[0]) *
                              static_cast<std::size_t>(dims[1]));
    ds.read(flat.data(), ct);
    for (hsize_t r = 0; r < dims[0]; ++r) {
        out[r].assign(flat.begin() + r * dims[1],
                      flat.begin() + (r + 1) * dims[1]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

std::string utc_iso_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string short_hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) {
        buf[sizeof(buf) - 1] = '\0';
        return std::string(buf);
    }
    return std::string("<unknown>");
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool checkpoint_enabled() { return !checkpoint_dir().empty(); }

std::string checkpoint_dir() { return env_string("ED_LANCZOS_CHECKPOINT_DIR"); }

uint64_t checkpoint_interval() {
    const std::string s = env_string("ED_LANCZOS_CHECKPOINT_INTERVAL");
    if (s.empty()) return 100;
    try {
        long long v = std::stoll(s);
        if (v > 0) return static_cast<uint64_t>(v);
    } catch (...) {
        // Invalid values silently fall back to default.
    }
    return 100;
}

std::string checkpoint_filename(const std::string& dir) {
    if (dir.empty()) return std::string();
    std::filesystem::path p(dir);
    p /= kCheckpointBaseName;
    return p.string();
}

bool checkpoint_resume_requested() {
    const std::string flag = env_string("ED_LANCZOS_RESUME");
    if (flag.empty() || flag == "0" || flag == "false" || flag == "FALSE" ||
        flag == "no" || flag == "NO") {
        return false;
    }
    const std::string dir = checkpoint_dir();
    if (dir.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(checkpoint_filename(dir), ec) && !ec;
}

std::string capture_mt19937_state(const std::mt19937& gen) {
    std::ostringstream oss;
    oss << gen;
    return oss.str();
}

void restore_mt19937_state(std::mt19937& gen, const std::string& state_text) {
    if (state_text.empty()) return;  // tolerated; gen left as-is
    std::istringstream iss(state_text);
    iss >> gen;
}

void write_lanczos_checkpoint(const std::string& dir,
                              const LanczosCheckpoint& cp) {
    if (dir.empty()) {
        throw std::runtime_error("write_lanczos_checkpoint: empty dir");
    }
    if (cp.alpha.size() != cp.iteration) {
        throw std::runtime_error(
            "write_lanczos_checkpoint: alpha.size() != iteration");
    }
    if (cp.beta.size() != cp.iteration + 1) {
        throw std::runtime_error(
            "write_lanczos_checkpoint: beta.size() != iteration + 1");
    }
    if (cp.v_prev.size() != cp.N || cp.v_current.size() != cp.N) {
        throw std::runtime_error(
            "write_lanczos_checkpoint: v_prev / v_current size mismatch");
    }
    for (const auto& rv : cp.ring_vectors) {
        if (rv.size() != cp.N) {
            throw std::runtime_error(
                "write_lanczos_checkpoint: ring vector size mismatch");
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("write_lanczos_checkpoint: mkdir(" + dir +
                                 ") failed: " + ec.message());
    }

    const std::string final_path = checkpoint_filename(dir);
    const std::string tmp_path = final_path + ".tmp";

    // Serialize the HDF5 work inside a guard so concurrent threads in the
    // same process don't tear each other's writes (libhdf5 is not always
    // thread-safe depending on build flags).
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> io_lock(io_mtx);

    try {
        H5::Exception::dontPrint();
        // Always create / truncate the .tmp file.
        H5::H5File file(tmp_path, H5F_ACC_TRUNC);

        // /metadata
        {
            H5::Group g = file.createGroup("/metadata");
            const uint32_t schema = kSchemaVersion;
            write_scalar(g, "schema_version", schema, H5::PredType::NATIVE_UINT32);
            write_string_scalar(g, "solver", std::string(kSolverTag));
            write_scalar(g, "N", cp.N, H5::PredType::NATIVE_UINT64);
            write_scalar(g, "max_iter", cp.max_iter, H5::PredType::NATIVE_UINT64);
            write_scalar(g, "exct", cp.exct, H5::PredType::NATIVE_UINT64);
            write_scalar(g, "tol", cp.tol, H5::PredType::NATIVE_DOUBLE);
            write_scalar(g, "iteration", cp.iteration,
                         H5::PredType::NATIVE_UINT64);
            write_string_scalar(g, "timestamp_iso", utc_iso_timestamp());
            write_string_scalar(g, "host", short_hostname());
            const uint8_t cs = cp.complex_seed ? 1u : 0u;
            write_scalar(g, "complex_seed", cs, H5::PredType::NATIVE_UINT8);
            write_scalar(g, "last_w_norm", cp.last_w_norm,
                         H5::PredType::NATIVE_DOUBLE);
        }

        // /tridiag
        {
            H5::Group g = file.createGroup("/tridiag");
            write_1d(g, "alpha", cp.alpha, H5::PredType::NATIVE_DOUBLE);
            write_1d(g, "beta", cp.beta, H5::PredType::NATIVE_DOUBLE);
        }

        // /vectors
        {
            H5::Group g = file.createGroup("/vectors");
            write_complex_1d(g, "v_prev", cp.v_prev);
            write_complex_1d(g, "v_current", cp.v_current);
        }

        // /ring
        {
            H5::Group g = file.createGroup("/ring");
            const uint64_t count =
                static_cast<uint64_t>(cp.ring_vectors.size());
            write_scalar(g, "head", cp.ring_head, H5::PredType::NATIVE_UINT64);
            write_scalar(g, "count", count, H5::PredType::NATIVE_UINT64);
            // Always create the dataset, even if empty, for round-trip
            // reads. Use rows = count, row_len = N (zero-row case allowed).
            write_complex_2d(g, "vectors", cp.ring_vectors,
                             static_cast<hsize_t>(cp.N));
        }

        // /rng
        {
            H5::Group g = file.createGroup("/rng");
            write_string_scalar(g, "state", cp.rng_state_text);
        }

        // /counters
        {
            H5::Group g = file.createGroup("/counters");
            write_scalar(g, "total_reorth", cp.total_reorth_count,
                         H5::PredType::NATIVE_UINT64);
            write_scalar(g, "selective_reorth", cp.selective_reorth_count,
                         H5::PredType::NATIVE_UINT64);
        }

        // /convergence
        {
            H5::Group g = file.createGroup("/convergence");
            write_1d(g, "prev_eigenvalues", cp.prev_eigenvalues,
                     H5::PredType::NATIVE_DOUBLE);
            const uint8_t cf = cp.eigenvalues_converged ? 1u : 0u;
            write_scalar(g, "converged_flag", cf, H5::PredType::NATIVE_UINT8);
        }

        file.flush(H5F_SCOPE_GLOBAL);
        // RAII close happens at scope exit.
    } catch (H5::Exception& e) {
        // Best-effort cleanup of the half-written tmp file.
        std::error_code rec;
        std::filesystem::remove(tmp_path, rec);
        throw std::runtime_error(std::string("write_lanczos_checkpoint: ") +
                                 e.getCDetailMsg());
    }

    // Atomic rename onto the canonical path. std::filesystem::rename is
    // atomic on POSIX when source and destination are on the same
    // filesystem; we just created `tmp_path` in `dir` so this holds.
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        std::error_code rec;
        std::filesystem::remove(tmp_path, rec);
        throw std::runtime_error("write_lanczos_checkpoint: rename(" +
                                 tmp_path + " -> " + final_path +
                                 ") failed: " + ec.message());
    }
}

LanczosCheckpoint read_lanczos_checkpoint(const std::string& dir) {
    if (dir.empty()) {
        throw std::runtime_error("read_lanczos_checkpoint: empty dir");
    }
    const std::string path = checkpoint_filename(dir);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        throw std::runtime_error("read_lanczos_checkpoint: file not found: " +
                                 path);
    }

    static std::mutex io_mtx;
    std::lock_guard<std::mutex> io_lock(io_mtx);

    LanczosCheckpoint cp;
    try {
        H5::Exception::dontPrint();
        H5::H5File file(path, H5F_ACC_RDONLY);

        // /metadata
        {
            H5::Group g = file.openGroup("/metadata");
            const uint32_t schema = read_scalar<uint32_t>(
                g, "schema_version", H5::PredType::NATIVE_UINT32);
            if (schema != kSchemaVersion) {
                throw std::runtime_error(
                    "read_lanczos_checkpoint: unsupported schema_version " +
                    std::to_string(schema) + " (expected " +
                    std::to_string(kSchemaVersion) + ")");
            }
            const std::string solver = read_string_scalar(g, "solver");
            if (solver != kSolverTag) {
                throw std::runtime_error(
                    "read_lanczos_checkpoint: solver tag mismatch (got '" +
                    solver + "', expected '" + kSolverTag + "')");
            }
            cp.N = read_scalar<uint64_t>(g, "N", H5::PredType::NATIVE_UINT64);
            cp.max_iter = read_scalar<uint64_t>(g, "max_iter",
                                                H5::PredType::NATIVE_UINT64);
            cp.exct =
                read_scalar<uint64_t>(g, "exct", H5::PredType::NATIVE_UINT64);
            cp.tol =
                read_scalar<double>(g, "tol", H5::PredType::NATIVE_DOUBLE);
            cp.iteration = read_scalar<uint64_t>(g, "iteration",
                                                 H5::PredType::NATIVE_UINT64);
            const uint8_t cs = read_scalar<uint8_t>(
                g, "complex_seed", H5::PredType::NATIVE_UINT8);
            cp.complex_seed = (cs != 0);
            cp.last_w_norm = read_scalar<double>(g, "last_w_norm",
                                                 H5::PredType::NATIVE_DOUBLE);
        }

        // /tridiag
        {
            H5::Group g = file.openGroup("/tridiag");
            cp.alpha =
                read_1d<double>(g, "alpha", H5::PredType::NATIVE_DOUBLE);
            cp.beta = read_1d<double>(g, "beta", H5::PredType::NATIVE_DOUBLE);
        }

        // /vectors
        {
            H5::Group g = file.openGroup("/vectors");
            cp.v_prev = read_complex_1d(g, "v_prev");
            cp.v_current = read_complex_1d(g, "v_current");
        }

        // /ring
        {
            H5::Group g = file.openGroup("/ring");
            cp.ring_head =
                read_scalar<uint64_t>(g, "head", H5::PredType::NATIVE_UINT64);
            const uint64_t count = read_scalar<uint64_t>(
                g, "count", H5::PredType::NATIVE_UINT64);
            cp.ring_vectors = read_complex_2d(g, "vectors",
                                              static_cast<hsize_t>(cp.N));
            if (cp.ring_vectors.size() != count) {
                throw std::runtime_error(
                    "read_lanczos_checkpoint: ring count " +
                    std::to_string(count) + " != stored rows " +
                    std::to_string(cp.ring_vectors.size()));
            }
        }

        // /rng
        {
            H5::Group g = file.openGroup("/rng");
            cp.rng_state_text = read_string_scalar(g, "state");
        }

        // /counters
        {
            H5::Group g = file.openGroup("/counters");
            cp.total_reorth_count = read_scalar<uint64_t>(
                g, "total_reorth", H5::PredType::NATIVE_UINT64);
            cp.selective_reorth_count = read_scalar<uint64_t>(
                g, "selective_reorth", H5::PredType::NATIVE_UINT64);
        }

        // /convergence
        {
            H5::Group g = file.openGroup("/convergence");
            cp.prev_eigenvalues = read_1d<double>(g, "prev_eigenvalues",
                                                  H5::PredType::NATIVE_DOUBLE);
            const uint8_t cf = read_scalar<uint8_t>(g, "converged_flag",
                                                    H5::PredType::NATIVE_UINT8);
            cp.eigenvalues_converged = (cf != 0);
        }
    } catch (H5::Exception& e) {
        throw std::runtime_error(std::string("read_lanczos_checkpoint: ") +
                                 e.getCDetailMsg());
    }

    // Cross-field invariants (cheap, catches obviously corrupted files).
    if (cp.alpha.size() != cp.iteration) {
        throw std::runtime_error(
            "read_lanczos_checkpoint: alpha.size() != iteration");
    }
    if (cp.beta.size() != cp.iteration + 1) {
        throw std::runtime_error(
            "read_lanczos_checkpoint: beta.size() != iteration + 1");
    }
    if (cp.v_prev.size() != cp.N || cp.v_current.size() != cp.N) {
        throw std::runtime_error(
            "read_lanczos_checkpoint: v_prev / v_current dim mismatch");
    }

    return cp;
}

}  // namespace lanczos_io
