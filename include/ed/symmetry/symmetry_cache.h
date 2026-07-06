#pragma once
// =============================================================================
// include/ed/symmetry/symmetry_cache.h
//
// Stage 3 of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md): content-addressed persistence
// + in-process sharing for the OrbitTable, finally implementing the
// long-plumbed-but-never-consumed ``basis_cache_dir`` contract.
//
// Two layers, both keyed by ``OrbitTable::content_hash`` (CompiledGroup
// element hash x subspace signature x engine version -- Hamiltonian
// couplings deliberately excluded, which is why parameter sweeps hit):
//
//   1. In-process registry: a small FIFO of shared_ptr<const OrbitTable>.
//      Repeated qed.solve/thermal/spectral calls in one process (and the
//      GeneratorSet temp-dir round-trip, whose PATH changes but whose
//      CONTENT does not) reuse the table without any rebuild. Always on;
//      correctness-neutral (tables are immutable once built).
//
//   2. Disk cache: ``<cache_dir>/sym_v2/<hash>.otab`` -- a raw
//      little-endian binary blob (header + reps + stab ids + flattened
//      stabilizer sets), written atomically (tmp + rename) so a killed
//      run never leaves a torn file. The stored hash must equal the
//      filename key AND the recomputed request key, or the file is
//      ignored and rebuilt (derived data: corruption costs one rebuild,
//      never wrong physics). Env:
//         ED_SYM_CACHE=0        disable the DISK layer
//         ED_SYM_CACHE_DIR=...  override the location for every lattice
//
// The acquire_* front-ends below are what the sector builders call; the
// raw build_orbit_table_* functions in orbit_table.h remain the compute
// kernels.
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <ed/symmetry/orbit_table.h>
#include <ed/symmetry/sym_profile.h>

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// Cache keys WITHOUT building the table (CompiledGroup compilation is
// microseconds). Must match the content_hash the builders stamp -- pinned
// by tests/unit/test_symmetry_cache.cpp.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::uint64_t
orbit_table_key_fixed_sz(std::uint64_t n_bits, int n_up,
                         const SymmetryGroupInfo& info) {
    const CompiledGroup cg = info.max_clique.empty()
        ? CompiledGroup{}
        : CompiledGroup::from_permutations(
              info.max_clique, static_cast<int>(info.max_clique[0].size()));
    return cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL)
        ^ (static_cast<std::uint64_t>(n_up + 1) * 0xD6E8FEB86659FD93ULL);
}

[[nodiscard]] inline std::uint64_t
orbit_table_key_full(std::uint64_t n_bits, const SymmetryGroupInfo& info) {
    const CompiledGroup cg = info.max_clique.empty()
        ? CompiledGroup{}
        : CompiledGroup::from_permutations(
              info.max_clique, static_cast<int>(info.max_clique[0].size()));
    return cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL);
}

namespace detail {

// ---------------------------------------------------------------------------
// Disk format. Little-endian POD header + arrays. Version bumps invalidate
// via kOrbitTableVersion inside the key (a mismatched file is simply never
// looked up) AND the explicit header field (defense in depth for hand-moved
// files).
// ---------------------------------------------------------------------------
struct OtabHeader {
    char          magic[8];        // "QEDOTAB\0"
    std::uint64_t version;         // kOrbitTableVersion
    std::uint64_t content_hash;
    std::uint64_t subspace_dim;
    std::uint64_t n_reps;
    std::uint64_t n_stab_sets;
    std::uint64_t stab_elems_total;
};
inline constexpr char kOtabMagic[8] = {'Q','E','D','O','T','A','B','\0'};

[[nodiscard]] inline std::string
otab_path(const std::string& cache_dir, std::uint64_t key) {
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(key));
    return cache_dir + "/sym_v2/" + hex + ".otab";
}

[[nodiscard]] inline bool disk_cache_enabled() noexcept {
    const char* v = std::getenv("ED_SYM_CACHE");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}

[[nodiscard]] inline std::string cache_dir_override() {
    const char* v = std::getenv("ED_SYM_CACHE_DIR");
    return (v != nullptr && v[0] != '\0') ? std::string(v) : std::string{};
}

}  // namespace detail

// ---------------------------------------------------------------------------
// save / load (public for tests + the precompute_basis_only CLI mode).
// Failures are soft everywhere: the cache is derived data, so any I/O or
// validation problem falls back to "no cache" and the caller rebuilds.
// ---------------------------------------------------------------------------
inline bool save_orbit_table(const OrbitTable& tab,
                             const std::string& cache_dir) {
    if (cache_dir.empty()) return false;
    namespace fs = std::filesystem;
    std::error_code ec;

    const std::string path = detail::otab_path(cache_dir, tab.content_hash);
    fs::create_directories(fs::path(path).parent_path(), ec);
    if (ec) return false;

    // Flatten the deduped stabilizer sets.
    std::vector<std::uint64_t> stab_offsets;
    std::vector<std::uint16_t> stab_flat;
    stab_offsets.reserve(tab.stab_elems.size() + 1);
    stab_offsets.push_back(0);
    for (const auto& st : tab.stab_elems) {
        stab_flat.insert(stab_flat.end(), st.begin(), st.end());
        stab_offsets.push_back(stab_flat.size());
    }

    detail::OtabHeader h{};
    std::memcpy(h.magic, detail::kOtabMagic, sizeof(h.magic));
    h.version          = detail::kOrbitTableVersion;
    h.content_hash     = tab.content_hash;
    h.subspace_dim     = tab.subspace_dim;
    h.n_reps           = tab.reps.size();
    h.n_stab_sets      = tab.stab_elems.size();
    h.stab_elems_total = stab_flat.size();

    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        auto put = [&f](const void* p, std::size_t n) {
            f.write(static_cast<const char*>(p),
                    static_cast<std::streamsize>(n));
        };
        put(&h, sizeof(h));
        put(tab.reps.data(),    tab.reps.size()    * sizeof(std::uint64_t));
        put(tab.stab_id.data(), tab.stab_id.size() * sizeof(std::uint16_t));
        put(stab_offsets.data(), stab_offsets.size() * sizeof(std::uint64_t));
        put(stab_flat.data(),   stab_flat.size()   * sizeof(std::uint16_t));
        if (!f.good()) {
            f.close();
            fs::remove(tmp, ec);
            return false;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) { fs::remove(tmp, ec); return false; }
    return true;
}

[[nodiscard]] inline std::shared_ptr<const OrbitTable>
load_orbit_table(std::uint64_t key, const std::string& cache_dir) {
    if (cache_dir.empty()) return nullptr;
    const std::string path = detail::otab_path(cache_dir, key);
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;

    detail::OtabHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f.good()
        || std::memcmp(h.magic, detail::kOtabMagic, sizeof(h.magic)) != 0
        || h.version != detail::kOrbitTableVersion
        || h.content_hash != key
        || h.n_stab_sets > 65536) {
        return nullptr;
    }

    auto tab = std::make_shared<OrbitTable>();
    tab->content_hash = h.content_hash;
    tab->subspace_dim = h.subspace_dim;
    tab->reps.resize(h.n_reps);
    tab->stab_id.resize(h.n_reps);
    std::vector<std::uint64_t> stab_offsets(h.n_stab_sets + 1);
    std::vector<std::uint16_t> stab_flat(h.stab_elems_total);

    auto get = [&f](void* p, std::size_t n) {
        f.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
    };
    get(tab->reps.data(),    h.n_reps * sizeof(std::uint64_t));
    get(tab->stab_id.data(), h.n_reps * sizeof(std::uint16_t));
    get(stab_offsets.data(), stab_offsets.size() * sizeof(std::uint64_t));
    get(stab_flat.data(),    stab_flat.size() * sizeof(std::uint16_t));
    if (!f.good()) return nullptr;

    tab->stab_elems.resize(h.n_stab_sets);
    for (std::size_t k = 0; k < h.n_stab_sets; ++k) {
        const std::uint64_t b = stab_offsets[k], e = stab_offsets[k + 1];
        if (e < b || e > stab_flat.size()) return nullptr;
        tab->stab_elems[k].assign(stab_flat.begin() + b, stab_flat.begin() + e);
    }
    // Sanity: every stab_id must index a stored set.
    for (std::uint16_t id : tab->stab_id)
        if (id >= h.n_stab_sets) return nullptr;
    return tab;
}

namespace detail {

// ---------------------------------------------------------------------------
// In-process registry: content-keyed FIFO of the most recent tables.
// Small (tables are 10 B/rep; 8 entries cover a full multi-Sz sweep loop),
// mutex-guarded, always on.
// ---------------------------------------------------------------------------
class OrbitTableRegistry {
public:
    static OrbitTableRegistry& instance() {
        static OrbitTableRegistry r;
        return r;
    }

    [[nodiscard]] std::shared_ptr<const OrbitTable> find(std::uint64_t key) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& e : entries_)
            if (e->content_hash == key) return e;
        return nullptr;
    }

    void insert(std::shared_ptr<const OrbitTable> tab) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& e : entries_)
            if (e->content_hash == tab->content_hash) return;
        entries_.push_back(std::move(tab));
        while (entries_.size() > kMaxEntries) entries_.pop_front();
    }

private:
    static constexpr std::size_t kMaxEntries = 8;
    std::mutex mu_;
    std::deque<std::shared_ptr<const OrbitTable>> entries_;
};

template <class BuildFn>
[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_impl(std::uint64_t key, const std::string& cache_dir, BuildFn&& build) {
    auto& reg = OrbitTableRegistry::instance();
    if (auto hit = reg.find(key)) {
        if (sym_profile_enabled())
            std::fprintf(stderr,
                         "[sym-profile] orbit-table registry HIT (%zu reps)\n",
                         hit->size());
        return hit;
    }
    const std::string dir = disk_cache_enabled() ? cache_dir : std::string{};
    if (auto disk = load_orbit_table(key, dir)) {
        if (sym_profile_enabled())
            std::fprintf(stderr,
                         "[sym-profile] orbit-table disk HIT (%zu reps)\n",
                         disk->size());
        reg.insert(disk);
        return disk;
    }
    auto tab = std::make_shared<OrbitTable>(build());
    if (!dir.empty()) {
        SymPhaseTimer prof("orbit-table disk save");
        save_orbit_table(*tab, dir);
    }
    reg.insert(tab);
    return tab;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Acquire front-ends. ``cache_dir == ""`` -> registry only (no disk).
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_fixed_sz(std::uint64_t n_bits, int n_up,
                             const SymmetryGroupInfo& info,
                             const std::string& cache_dir = {}) {
    return detail::acquire_impl(
        orbit_table_key_fixed_sz(n_bits, n_up, info), cache_dir,
        [&] { return build_orbit_table_fixed_sz_streaming(n_bits, n_up, info); });
}

/// Stage 5b: acquire with a caller-supplied CompiledGroup (flip-extended
/// lanes). Same key formula as the info-based fixed-Sz path, so the disk
/// cache + registry work identically (the flip elements change the group
/// hash and therefore the key).
[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_fixed_sz_compiled(std::uint64_t        n_bits,
                                      int                  n_up,
                                      const CompiledGroup& cg,
                                      const std::string&   cache_dir = {}) {
    const std::uint64_t key = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL)
        ^ (static_cast<std::uint64_t>(n_up + 1) * 0xD6E8FEB86659FD93ULL);
    return detail::acquire_impl(
        key, cache_dir,
        [&] { return build_orbit_table_fixed_sz_streaming(n_bits, n_up, cg); });
}

[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_full(std::uint64_t n_bits,
                         const SymmetryGroupInfo& info,
                         const std::string& cache_dir = {}) {
    return detail::acquire_impl(
        orbit_table_key_full(n_bits, info), cache_dir,
        [&] { return build_orbit_table_full(n_bits, info); });
}

[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_parity_compiled(std::uint64_t        n_bits,
                                    int                  parity,
                                    const CompiledGroup& cg,
                                    const std::string&   cache_dir = {}) {
    const std::uint64_t key = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL)
        ^ (static_cast<std::uint64_t>(parity + 7) * 0xA24BAED4963EE407ULL);
    return detail::acquire_impl(
        key, cache_dir,
        [&] { return build_orbit_table_parity_compiled(n_bits, parity, cg); });
}

[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_full_compiled(std::uint64_t        n_bits,
                                  const CompiledGroup& cg,
                                  const std::string&   cache_dir = {}) {
    const std::uint64_t key = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL);
    return detail::acquire_impl(
        key, cache_dir,
        [&] { return build_orbit_table_full_compiled(n_bits, cg); });
}

/// Resolve the effective disk-cache directory for a lattice fixture
/// directory (the documented ``basis_cache_dir`` contract):
///   explicit option > ED_SYM_CACHE_DIR > <lattice_dir>/basis_cache;
///   ED_SYM_CACHE=0 disables regardless. ``lattice_dir`` may be empty
///   (in-memory operators): registry-only unless an override names a dir.
[[nodiscard]] inline std::string
resolve_sym_cache_dir(const std::string& explicit_dir,
                      const std::string& lattice_dir) {
    if (!detail::disk_cache_enabled()) return {};
    if (!explicit_dir.empty()) return explicit_dir;
    if (std::string ovr = detail::cache_dir_override(); !ovr.empty()) return ovr;
    if (!lattice_dir.empty()) return lattice_dir + "/basis_cache";
    return {};
}

}  // namespace ed::symmetry
