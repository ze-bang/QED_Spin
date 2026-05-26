#include <ed/io/lanczos_basis_buffer.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <string>
#include <unordered_map>
#include <vector>

namespace lanczos_io {

namespace {

struct Buffer {
    uint64_t N = 0;
    std::vector<ComplexVector> vectors;
};

struct Registry {
    std::mutex mtx;
    std::unordered_map<std::string, Buffer> buffers;

    // Wave B5 (May 2026): per-N free-list of recently-released vector
    // pools. When a `BasisBufferScope` exits we move its `vectors`
    // (which still holds the per-vector heap blocks) into this pool;
    // the next `register_basis_buffer` for the same N pops the pool
    // and gets pre-allocated storage. Most workloads call Lanczos
    // many times for the same N (per-sample FTLM, per-sector
    // streaming symmetry), so this collapses the heap traffic down
    // to "warm pool" reuse after the first run. Cap the pool size at
    // a few entries per N to avoid pathological memory growth.
    static constexpr std::size_t kPoolPerN = 4;
    std::unordered_map<uint64_t, std::vector<std::vector<ComplexVector>>> pool;
};

Registry& registry() {
    static Registry r;
    return r;
}

} // anonymous namespace

bool force_disk_storage() {
    static const bool cached = []() {
        const char* env = std::getenv("ED_LANCZOS_DISK");
        if (!env) return false;
        // Accept "1", "true", "TRUE", "yes", "YES" as truthy.
        if (env[0] == '\0') return false;
        if (std::strcmp(env, "0") == 0) return false;
        if (std::strcmp(env, "false") == 0) return false;
        if (std::strcmp(env, "FALSE") == 0) return false;
        if (std::strcmp(env, "no") == 0) return false;
        if (std::strcmp(env, "NO") == 0) return false;
        return true;
    }();
    return cached;
}

void register_basis_buffer(const std::string& key,
                           uint64_t N,
                           uint64_t reserve_vectors) {
    if (force_disk_storage()) return;

    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto& buf = reg.buffers[key];
    buf = Buffer{};
    buf.N = N;

    // Wave B5: try to pop a warm pool entry sized for this N. The
    // pool holds the still-allocated per-vector heap blocks from the
    // last release call -- reusing them collapses the heap traffic
    // to "warm allocator" rates.
    auto pool_it = reg.pool.find(N);
    if (pool_it != reg.pool.end() && !pool_it->second.empty()) {
        buf.vectors = std::move(pool_it->second.back());
        pool_it->second.pop_back();
        // The pooled vectors still hold capacity for their dim-N
        // payloads; just clear the outer size so append() restarts
        // from index 0 without dropping the inner allocations.
        // (clear() preserves capacity of the outer vector AND each
        // inner ComplexVector remains intact, so future
        // ``vectors.push_back(...)`` slots reuse the popped storage.)
        buf.vectors.clear();
    }
    if (reserve_vectors > 0) {
        // Cap the reservation: for very large problems the caller's bound can
        // be wildly optimistic. We reserve at most ~4 GiB of pointer slots
        // (std::vector header is tiny; the real memory is allocated lazily per
        // vector on append()).
        constexpr uint64_t kMaxReserve = uint64_t(1) << 26; // 64M slots
        buf.vectors.reserve(std::min(reserve_vectors, kMaxReserve));
    }
}

void release_basis_buffer(const std::string& key) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);
    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return;

    // Wave B5: move the per-vector storage into the per-N pool so the
    // next ``register_basis_buffer`` for the same N can reuse it.
    // Capped at ``kPoolPerN`` entries to avoid unbounded memory
    // growth; pool slots beyond that are just dropped (free()'d).
    Buffer& b = it->second;
    if (b.N != 0 && !b.vectors.empty()) {
        auto& slot = reg.pool[b.N];
        if (slot.size() < Registry::kPoolPerN) {
            slot.push_back(std::move(b.vectors));
        }
    }
    reg.buffers.erase(it);
}

bool has_basis_buffer(const std::string& key) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);
    return reg.buffers.find(key) != reg.buffers.end();
}

bool append_basis_vector(const std::string& key, const ComplexVector& vec) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return false;

    Buffer& buf = it->second;
    if (buf.N != 0 && vec.size() != buf.N) return false;
    if (buf.N == 0) buf.N = vec.size();

    buf.vectors.push_back(vec);
    return true;
}

bool append_basis_vector(const std::string& key, ComplexVector&& vec) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return false;

    Buffer& buf = it->second;
    if (buf.N != 0 && vec.size() != buf.N) return false;
    if (buf.N == 0) buf.N = vec.size();

    buf.vectors.push_back(std::move(vec));
    return true;
}

bool set_basis_vector(const std::string& key,
                      uint64_t index,
                      const ComplexVector& vec) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return false;

    Buffer& buf = it->second;
    if (index >= buf.vectors.size()) return false;
    if (buf.N != 0 && vec.size() != buf.N) return false;

    buf.vectors[index] = vec;
    return true;
}

void truncate_basis_buffer(const std::string& key, uint64_t new_size) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return;

    Buffer& buf = it->second;
    if (new_size < buf.vectors.size()) {
        buf.vectors.resize(new_size);
    }
}

bool get_basis_vector(const std::string& key,
                      uint64_t index,
                      ComplexVector& out) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return false;

    const Buffer& buf = it->second;
    if (index >= buf.vectors.size()) return false;

    out = buf.vectors[index]; // copy out under the lock; safe for readers.
    return true;
}

const Complex* get_basis_vector_ptr(const std::string& key, uint64_t index) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return nullptr;

    const Buffer& buf = it->second;
    if (index >= buf.vectors.size()) return nullptr;

    return buf.vectors[index].data();
}

uint64_t basis_buffer_size(const std::string& key) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    auto it = reg.buffers.find(key);
    if (it == reg.buffers.end()) return 0;
    return static_cast<uint64_t>(it->second.vectors.size());
}

uint64_t total_basis_buffer_bytes() {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mtx);

    uint64_t total = 0;
    for (const auto& kv : reg.buffers) {
        const Buffer& buf = kv.second;
        for (const auto& v : buf.vectors) {
            total += v.size() * sizeof(Complex);
        }
    }
    return total;
}

} // namespace lanczos_io
