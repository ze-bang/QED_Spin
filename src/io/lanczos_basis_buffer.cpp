#include <ed/io/lanczos_basis_buffer.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
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
    reg.buffers.erase(key);
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
