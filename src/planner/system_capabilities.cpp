// =============================================================================
// src/planner/system_capabilities.cpp
//
// Implementation of `ed::planner::probe_system()`. Best-effort runtime probe;
// see header for the env-override contract.
// =============================================================================

#include <ed/planner/system_capabilities.h>

// Build-capability flags come straight from the same preprocessor macros
// ed::has_*_build() uses (WITH_CUDA / WITH_MPI / ED_HAVE_NCCL). We replicate
// the three #ifdefs locally rather than #include <ed/api.h> so this leaf probe
// does not drag in the heavy operator/Eigen/json header stack.

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

#include <unistd.h>   // sysconf

#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#endif
#ifdef WITH_MPI
#  include <mpi.h>
#endif

namespace ed::planner {

namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

[[nodiscard]] std::optional<double> env_double(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return std::nullopt;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    if (end == v) return std::nullopt;
    return d;
}

[[nodiscard]] std::optional<long> env_long(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return std::nullopt;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v) return std::nullopt;
    return n;
}

// Total physical RAM via sysconf.
[[nodiscard]] std::uint64_t probe_ram_total_bytes() {
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_sz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_sz > 0) {
        return static_cast<std::uint64_t>(pages) *
               static_cast<std::uint64_t>(page_sz);
    }
    return 0;
}

// Available RAM via /proc/meminfo MemAvailable (kB). Returns 0 if unreadable.
[[nodiscard]] std::uint64_t probe_ram_avail_bytes() {
    std::ifstream f("/proc/meminfo");
    if (!f) return 0;
    std::string key;
    while (f >> key) {
        if (key == "MemAvailable:") {
            std::uint64_t kb = 0;
            if (f >> kb) return kb * std::uint64_t{1024};
            return 0;
        }
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

}  // namespace

std::string SystemCapabilities::summary() const {
    std::ostringstream os;
    os << "SystemCapabilities (best effort):\n"
       << "  ram_total   = " << ram_total_gb() << " GB\n"
       << "  ram_avail   = " << ram_avail_gb() << " GB\n"
       << "  n_cores     = " << n_cores << "\n"
       << "  gpus        = " << n_gpus << " (vram_avail "
       << vram_avail_gb() << " GB / device)\n"
       << "  mpi_ranks   = " << n_mpi_ranks << "\n"
       << "  build flags = WITH_CUDA=" << (has_cuda_build ? 1 : 0)
       << " WITH_MPI=" << (has_mpi_build ? 1 : 0)
       << " WITH_NCCL=" << (has_nccl_build ? 1 : 0) << "\n";
    for (const auto& n : notes) os << "  note: " << n << "\n";
    return os.str();
}

SystemCapabilities probe_system(bool refresh) {
    static std::mutex mtx;
    static std::optional<SystemCapabilities> cache;
    std::lock_guard<std::mutex> lock(mtx);
    if (cache.has_value() && !refresh) return *cache;

    SystemCapabilities c;
    c.notes.clear();
#ifdef WITH_CUDA
    c.has_cuda_build = true;
#endif
#ifdef WITH_MPI
    c.has_mpi_build = true;
#endif
#ifdef ED_HAVE_NCCL
    c.has_nccl_build = true;
#endif

    // ---- CPU cores ----------------------------------------------------------
    {
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n > 0) {
            c.n_cores = static_cast<int>(n);
        } else {
            const unsigned hc = std::thread::hardware_concurrency();
            c.n_cores = hc > 0 ? static_cast<int>(hc) : 1;
        }
    }

    // ---- RAM (env override > sysconf/proc) ----------------------------------
    if (auto gb = env_double("QED_HOST_MEMORY_GB")) {
        c.ram_total_bytes = static_cast<std::uint64_t>(*gb * kGiB);
        c.ram_avail_bytes = c.ram_total_bytes;
        c.notes.push_back("ram: from env QED_HOST_MEMORY_GB");
    } else {
        const std::uint64_t total = probe_ram_total_bytes();
        const std::uint64_t avail = probe_ram_avail_bytes();
        if (total > 0) {
            c.ram_total_bytes = total;
            c.ram_avail_bytes = (avail > 0) ? avail : total;
            c.notes.push_back("ram: via sysconf/proc");
        } else {
            c.notes.push_back("ram: undetected (defaulted to 16 GB; set "
                              "QED_HOST_MEMORY_GB to override)");
        }
    }

    // ---- GPU (env override > nvidia runtime) --------------------------------
    {
        auto env_n   = env_long("QED_HOST_N_GPUS");
        auto env_mem = env_double("QED_HOST_GPU_MEMORY_GB");
        if (env_n || env_mem) {
            c.n_gpus = env_n ? static_cast<int>(*env_n) : 0;
            const std::uint64_t b =
                env_mem ? static_cast<std::uint64_t>(*env_mem * kGiB) : 0;
            c.vram_total_bytes = b;
            c.vram_avail_bytes = b;
            c.notes.push_back("gpu: from env QED_HOST_N_GPUS / "
                              "QED_HOST_GPU_MEMORY_GB");
        } else {
#ifdef WITH_CUDA
            int dev_count = 0;
            if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0) {
                std::uint64_t smallest_avail = 0;
                std::uint64_t smallest_total = 0;
                bool any = false;
                for (int d = 0; d < dev_count; ++d) {
                    if (cudaSetDevice(d) != cudaSuccess) continue;
                    std::size_t free_b = 0, total_b = 0;
                    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) continue;
                    if (!any || total_b < smallest_total) {
                        smallest_avail = free_b;
                        smallest_total = total_b;
                    }
                    any = true;
                }
                if (any) {
                    c.n_gpus = dev_count;
                    c.vram_avail_bytes = smallest_avail;
                    c.vram_total_bytes = smallest_total;
                    c.notes.push_back("gpu: via cudaMemGetInfo");
                } else {
                    c.notes.push_back("gpu: CUDA build but no usable device");
                }
            } else {
                c.notes.push_back("gpu: no CUDA-visible device");
            }
#else
            c.notes.push_back("gpu: no CUDA build");
#endif
        }
    }

    // ---- MPI ranks (env/scheduler > MPI_Comm_size) --------------------------
    {
        int ranks = 0;
        std::string how;
#ifdef WITH_MPI
        int inited = 0;
        if (MPI_Initialized(&inited) == MPI_SUCCESS && inited) {
            int sz = 1;
            if (MPI_Comm_size(MPI_COMM_WORLD, &sz) == MPI_SUCCESS && sz > 0) {
                ranks = sz;
                how = "via MPI_Comm_size";
            }
        }
#endif
        if (ranks == 0) {
            if (auto n = env_long("QED_HOST_N_MPI_RANKS")) {
                ranks = static_cast<int>(*n);
                how = "from env QED_HOST_N_MPI_RANKS";
            } else if (auto s = env_long("SLURM_NTASKS")) {
                ranks = static_cast<int>(*s);
                how = "from env SLURM_NTASKS";
            } else {
                ranks = c.n_cores > 0 ? c.n_cores : 1;
                how = "defaulted to logical core count";
            }
        }
        c.n_mpi_ranks = ranks > 0 ? ranks : 1;
        c.notes.push_back("mpi_ranks: " + how);
    }

    cache = c;
    return c;
}

}  // namespace ed::planner
