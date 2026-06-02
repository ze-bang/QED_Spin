#pragma once

#include <ed/core/construct_ham.h>
#include <ed/core/linear_operator.h>
#include <ed/core/sorted_uint64_index.h>   // Phase 3a #5: compact uint64->size_t map
#include <ed/symmetry/projector_chain.h>   // Orthogonal symmetry composition (May 2026)
#include <H5Cpp.h>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <atomic>         // Wave A2: lazy-build state
#include <cstdlib>        // Wave A2: getenv for budget gate
#include <filesystem>     // P0.12
#include <numeric>
#include <mutex>
#include <system_error>   // P0.12

// ============================================================================
// ShardedOrbitCache -- striped lock map for state -> orbit-rep memoisation
// ----------------------------------------------------------------------------
// The streaming symmetry operator memoises basis -> orbit_representative
// across SpMV iterations. Hot SpMV is multi-threaded (OpenMP), and a single
// std::mutex around one std::unordered_map serialises every cache hit, which
// can cap parallel scaling at ~4 cores. This struct shards by (basis %
// kNumShards), so contention drops by ~kNumShards (16x by default).
//
// API mirrors the small subset that the call sites used (find / insert).
// ============================================================================
struct ShardedOrbitCache {
    static constexpr std::size_t kNumShards = 16;  // power of two for fast %
    struct Shard {
        std::unordered_map<uint64_t, uint64_t> map;
        mutable std::mutex mu;
    };
    mutable std::array<Shard, kNumShards> shards;

    static inline std::size_t shard_of(uint64_t key) noexcept {
        // splitmix64-style mix to avoid pathological clustering when keys
        // share low bits (typical for fixed-Sz / popcount-aligned states).
        uint64_t x = key + 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x ^= (x >> 31);
        return static_cast<std::size_t>(x) & (kNumShards - 1);
    }

    bool find(uint64_t key, uint64_t& out) const {
        const auto& s = shards[shard_of(key)];
        std::lock_guard<std::mutex> lock(s.mu);
        auto it = s.map.find(key);
        if (it == s.map.end()) return false;
        out = it->second;
        return true;
    }

    void insert(uint64_t key, uint64_t value) const {
        auto& s = shards[shard_of(key)];
        std::lock_guard<std::mutex> lock(s.mu);
        s.map.emplace(key, value);
    }

    void clear() {
        for (auto& s : shards) {
            std::lock_guard<std::mutex> lock(s.mu);
            s.map.clear();
        }
    }
};

// ============================================================================
// SectorLookupHandle  (Wave A2, May 2026)
// ----------------------------------------------------------------------------
// Bundles the existing ``SortedUint64Index`` fallback (O(log N) per find)
// with an optional dense O(1) side-table built per sector. Used inside
// the streaming-symmetry SpMV hot loop:
//
//   const std::size_t k = lookup.find(s_prime);  // <-- the hot lookup
//
// becomes
//
//   const std::size_t k = handle.find(s_prime);
//
// which inlines to a single branch on ``handle.dense != nullptr``. The
// dense table is an int32_t array of length ``index_dim`` (= fixed-Sz
// dim, or 2^N for the full-space variant); ``-1`` marks "not in this
// sector's orbits". Building it is O(orbit_total) and the memory cost
// is 4 bytes per slot per sector -- a 4x reduction over the
// ``SortedUint64Index`` storage when the sector covers the bulk of the
// fixed-Sz / full Hilbert space.
//
// Memory budget is gated by ``ED_SYM_DENSE_LOOKUP_BYTES_MAX``
// (default 512 MB total across all sectors); above that the operator
// stays on the SortedUint64Index path.
// ============================================================================
struct SectorLookupHandle {
    const ed::core::SortedUint64Index* fallback = nullptr;
    const std::int32_t* dense                   = nullptr;
    // For the fixed-Sz variant this points to the parent's LinIndexTable;
    // for the full-space variant it is nullptr and ``s`` itself is the
    // dense-table index (since the full Hilbert space is enumerable by
    // ``state``).
    const LinIndexTable* lin                    = nullptr;

    inline std::size_t find(std::uint64_t s) const {
        if (dense != nullptr) {
            std::int64_t li;
            if (lin != nullptr) {
                li = lin->lookup(s);
                if (li < 0) {
                    return ed::core::SortedUint64Index::kNotFound;
                }
            } else {
                li = static_cast<std::int64_t>(s);
            }
            const std::int32_t v = dense[li];
            return (v < 0) ? ed::core::SortedUint64Index::kNotFound
                           : static_cast<std::size_t>(v);
        }
        return fallback->find(s);
    }
};

/**
 * @file streaming_symmetry.h
 * @brief Memory-efficient streaming implementation of symmetry-adapted exact diagonalization
 * 
 * This implementation avoids storing the full symmetrized basis and block matrices on disk.
 * Instead, it uses on-the-fly computation of symmetrized matrix-vector products.
 * 
 * Key advantages:
 * - No disk storage required for basis vectors
 * - No disk storage required for block matrices
 * - Memory usage scales with O(sector_dim × orbit_size) instead of O(full_dim)
 * - Faster basis generation (no file I/O)
 * - Can handle much larger systems
 * 
 * Algorithm (Matrix-Free):
 * 1. Generate orbit representatives on-the-fly
 * 2. Pre-compute orbit elements and phase factors per basis state
 * 3. Build lookup table: computational_state -> (sector_basis_index, orbit_index)
 * 4. Apply H term-by-term on orbit elements, project to symmetrized basis
 * 
 * This is truly matrix-free: H|ψ⟩ computed without expanding to full Hilbert space.
 */

// ============================================================================
// Orbit-Based Symmetry Data Structures
// ============================================================================

/**
 * @brief Compact representation of a symmetrized basis state
 * 
 * Stores orbit elements and their phase coefficients for efficient H*v computation.
 * Memory: O(orbit_size) per basis state instead of O(full_dim).
 */
struct SymBasisState {
    uint64_t orbit_rep;                    // Representative element (smallest in orbit)
    std::vector<int> quantum_numbers;      // Quantum numbers for this sector
    std::vector<uint64_t> orbit_elements;  // All states in the orbit (sorted ascending after sortOrbit())
    std::vector<Complex> orbit_coefficients;  // Coefficient of each orbit element in symmetrized state (parallel to orbit_elements)
    double norm;                           // Normalization factor
    // Cached reciprocal of `norm` (0 when norm == 0). Derived, NOT persisted:
    // recomputed by sortOrbit() -- the universal finalize point invoked once
    // after `norm` is assigned on every construction / HDF5-load path. Lets the
    // matvec hot loop turn the per-emit `group_norm / norm_k` division into a
    // multiply (see SymmetryBasisPolicy::coeff_modifier / iter_orbit).
    double inv_norm;

    SymBasisState() : orbit_rep(0), norm(0.0), inv_norm(0.0) {}

    SymBasisState(uint64_t rep, const std::vector<int>& qn, double n = 1.0)
        : orbit_rep(rep), quantum_numbers(qn), norm(n),
          inv_norm(n != 0.0 ? 1.0 / n : 0.0) {}

    /**
     * Sort orbit_elements ascending and parallel-sort orbit_coefficients to
     * keep them aligned. After this, findCoeff() does an O(log |orbit|)
     * binary search instead of an O(|orbit|) linear scan -- a 5-50x speedup
     * for large symmetry groups (e.g. 768 = 48 * 16 cubic + translation).
     *
     * Idempotent and safe to call multiple times; cheap if already sorted.
     */
    void sortOrbit() {
        // Refresh the cached reciprocal norm at the canonical finalize point.
        // Done before the early-out so size-0/1 orbits also get a valid
        // inv_norm (their `norm` is still set by the construction path).
        inv_norm = (norm != 0.0) ? (1.0 / norm) : 0.0;
        const size_t n = orbit_elements.size();
        if (n < 2) return;
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return orbit_elements[a] < orbit_elements[b]; });
        // In-place permutation with one extra buffer.
        std::vector<uint64_t> e(n);
        std::vector<Complex> c(n);
        for (size_t k = 0; k < n; ++k) {
            e[k] = orbit_elements[idx[k]];
            c[k] = orbit_coefficients[idx[k]];
        }
        orbit_elements = std::move(e);
        orbit_coefficients = std::move(c);
    }

    /**
     * O(log |orbit|) lookup of the coefficient of |s_prime> in this
     * symmetrised basis state. Returns 0 if s_prime is not in the orbit.
     * Requires orbit_elements to be sorted ascending (call sortOrbit()
     * exactly once after the orbit is fully populated).
     *
     * NOTE: kept as std::lower_bound deliberately. Unlike
     * ed::core::SortedUint64Index::find() (which searches multi-GB,
     * cache-cold arrays where a branch-free + prefetch form wins ~2x),
     * an orbit is SMALL -- bounded by the group order |G| (typically ~N
     * up to ~768) and almost always L1/L2-resident. There the manual
     * prefetch hints are pure overhead and the branch-free cmov form is
     * no faster than the well-predicted std::lower_bound, so the simple
     * form is retained.
     */
    inline Complex findCoeff(uint64_t s_prime) const {
        auto it = std::lower_bound(orbit_elements.begin(), orbit_elements.end(), s_prime);
        if (it == orbit_elements.end() || *it != s_prime) return Complex(0.0, 0.0);
        return orbit_coefficients[static_cast<size_t>(it - orbit_elements.begin())];
    }
};

/**
 * @brief Sector information with orbit representatives
 */
struct SymmetrySector {
    uint64_t sector_id;
    std::vector<int> quantum_numbers;
    std::vector<Complex> phase_factors;
    std::vector<SymBasisState> basis_states;  // Compact basis representation
    
    SymmetrySector() : sector_id(0) {}
};

// ============================================================================
// Streaming Symmetry Operator
// ============================================================================

/**
 * @brief Operator class with streaming symmetry-adapted matrix-vector products
 * 
 * This class extends the Operator class to provide on-the-fly computation
 * of symmetrized matrix-vector products without storing basis vectors or matrices.
 * 
 * MATRIX-FREE APPROACH:
 * - Pre-computes orbit elements and coefficients for each symmetrized basis state
 * - Builds lookup table: computational_state -> (sector_idx, basis_idx)
 * - Apply H term-by-term on orbit elements, project using lookup (no 2^N expansion)
 * 
 * Memory: O(total_orbit_elements) instead of O(2^N)
 */
class StreamingSymmetryOperator : public Operator {
private:
    // Mutable so the lazy per-sector orbit materialization can (re)build
    // sectors_[k].basis_states from inside const matvec / bind entry points.
    mutable std::vector<SymmetrySector> sectors_;
    mutable ShardedOrbitCache state_to_orbit_cache_;  // Striped-lock cache (16 shards) -- lower contention than single-mutex map
    
    // Lookup table: computational_state -> basis_idx_in_sector (one per sector).
    // Phase 3a #5: replaced std::unordered_map with SortedUint64Index (sorted
    // vector + binary search) to halve per-entry footprint from ~32-40 B to
    // 16 B. Critical at N>=36 with full point-group + Sz where the dominant
    // sector has ~3 x 10^7 representatives. The class exposes the same
    // operator[] = idiom for the build phase, so call sites only change in
    // the lookup pattern (find() -> sentinel kNotFound, no iterator). MUST
    // call .finalize() once per sector after the build pass to sort the
    // backing keys array.
    mutable std::vector<ed::core::SortedUint64Index> state_to_sector_basis_;
    
    // Cached group size from HDF5 load (allows skipping symmetry_info loading)
    uint64_t cached_group_size_ = 0;

    // ------------------------------------------------------------------
    // Streaming sector materialization ("stream sym sectors" plan,
    // Jun 2026). Twin of the block in FixedSzStreamingSymmetryOperator;
    // see that class for the rationale. Keeps only the deduped orbit-rep
    // list resident and builds one sector's orbit CSR on demand (LRU-1),
    // with the CPU reverse index built lazily on first matvec.
    // ------------------------------------------------------------------
    std::vector<uint64_t>        unique_orbit_reps_;
    bool                         lazy_sectors_enabled_ = false;
    mutable std::atomic<std::ptrdiff_t> materialized_sector_{-1};
    mutable std::vector<char>    reverse_index_built_;
    mutable std::mutex           materialize_mu_;

public:
    StreamingSymmetryOperator(uint64_t n_bits, float spin_l) 
        : Operator(n_bits, spin_l) {
        if (n_bits >= 64) {
            throw std::runtime_error("StreamingSymmetryOperator: n_bits = " + std::to_string(n_bits)
                + " >= 64 is not supported (would cause undefined behavior in 1ULL << n_bits)");
        }
    }

    /**
     * @brief Get the symmetry group size.
     * Returns symmetry_info.max_clique.size() if loaded, else the value
     * cached from an HDF5 basis load.  This allows the matvec and
     * eigenvector expansion code to work without loading the full
     * automorphism / symmetry data.
     */
    uint64_t getGroupSize() const {
        if (!symmetry_info.max_clique.empty())
            return symmetry_info.max_clique.size();
        if (cached_group_size_ > 0)
            return cached_group_size_;
        throw std::runtime_error("getGroupSize: neither symmetry_info nor "
                                 "cached group size available");
    }
    
    /**
     * @brief Generate symmetry sectors with orbit representatives (streaming version)
     * 
     * This generates the sectors without saving basis vectors to disk.
     * Pre-computes orbit elements/coefficients and builds lookup table for
     * matrix-free H*v computation.
     * 
     * Memory usage: O(total_orbit_elements) - much smaller than O(2^N)
     */
    void generateSymmetrySectorsStreaming(const std::string& dir) {
        std::cout << "\n=== Generating Symmetry Sectors (Matrix-Free Streaming) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        const size_t dim = 1ULL << n_bits_;
        const size_t num_sectors = symmetry_info.sectors.size();
        sectors_.resize(num_sectors);
        symmetrized_block_ham_sizes.assign(num_sectors, 0);
        state_to_sector_basis_.resize(num_sectors);
        
        // =====================================================================
        // PASS 1 (parallelizable): Identify unique orbit representatives
        // =====================================================================
        std::cout << "Pass 1: Identifying unique orbits (" << dim
                  << " states, group size " << symmetry_info.max_clique.size()
                  << ")..." << std::flush;
        
        auto pass1_start = std::chrono::high_resolution_clock::now();
        
        // Collect canonical orbit representatives WITHOUT an O(dim) scratch array.
        // For the full Hilbert space, rep(b) = min_{g in G} g(b); a state is its
        // orbit's canonical rep iff b == rep(b). Same set as distinct values of
        // rep(i) over all i, but only O(|orbits|) storage. With OpenMP, each thread
        // appends canonical bases in a static chunk, then we sort to restore
        // deterministic ascending order (matches the old "first occurrence" scan).
        // Kept resident in ``unique_orbit_reps_`` for lazy materialization.
        unique_orbit_reps_.clear();
#ifdef _OPENMP
        {
            const int nthreads = omp_get_max_threads();
            std::vector<std::vector<uint64_t>> thread_reps(static_cast<size_t>(nthreads));
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                #pragma omp for schedule(static) nowait
                for (size_t basis = 0; basis < dim; ++basis) {
                    uint64_t rep = basis;
                    for (const auto& perm : symmetry_info.max_clique) {
                        uint64_t permuted = applyPermutation(basis, perm);
                        if (permuted < rep) rep = permuted;
                    }
                    if (basis == rep) {
                        thread_reps[static_cast<size_t>(tid)].push_back(basis);
                    }
                }
            }
            size_t total = 0;
            for (const auto& v : thread_reps) {
                total += v.size();
            }
            unique_orbit_reps_.reserve(total);
            for (const auto& v : thread_reps) {
                unique_orbit_reps_.insert(unique_orbit_reps_.end(), v.begin(), v.end());
            }
            std::sort(unique_orbit_reps_.begin(), unique_orbit_reps_.end());
        }
#else
        unique_orbit_reps_.reserve(dim / symmetry_info.max_clique.size() + 1);
        for (size_t basis = 0; basis < dim; ++basis) {
            uint64_t rep = basis;
            for (const auto& perm : symmetry_info.max_clique) {
                uint64_t permuted = applyPermutation(static_cast<uint64_t>(basis), perm);
                if (permuted < rep) rep = permuted;
            }
            if (basis == rep) {
                unique_orbit_reps_.push_back(static_cast<uint64_t>(basis));
            }
        }
#endif
        
        auto pass1_end = std::chrono::high_resolution_clock::now();
        double pass1_ms = std::chrono::duration<double, std::milli>(pass1_end - pass1_start).count();
        const size_t num_orbits = unique_orbit_reps_.size();
        std::cout << " found " << num_orbits << " unique orbits"
                  << " (" << std::fixed << std::setprecision(1) << pass1_ms << " ms)" << std::endl;

        // Copy per-sector metadata up front (cheap; needed by both modes).
        for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
            const auto& sector_meta = symmetry_info.sectors[sector_idx];
            auto& sector = sectors_[sector_idx];
            sector.sector_id       = sector_meta.sector_id;
            sector.quantum_numbers = sector_meta.quantum_numbers;
            sector.phase_factors   = sector_meta.phase_factors;
        }

        // =====================================================================
        // Decide eager vs lazy ("stream sym sectors", Jun 2026). See the twin
        // in generateSymmetrySectorsStreamingFixedSz for the rationale.
        // =====================================================================
        std::size_t lazy_budget_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4 GiB
        if (const char* env = std::getenv("ED_SYM_LAZY_SECTORS_BYTES_MAX")) {
            try { lazy_budget_bytes = std::stoull(env); } catch (...) {}
        }
        const std::size_t group_sz =
            std::max<std::size_t>(1, symmetry_info.max_clique.size());
        const long double est_elems =
            static_cast<long double>(num_orbits) * group_sz * num_sectors;
        const long double est_bytes = est_elems * (24.0L + 16.0L);
        bool force_lazy = false, force_eager = false;
        if (const char* env = std::getenv("ED_SYM_LAZY_SECTORS")) {
            if (env[0] == '1') force_lazy = true;
            else if (env[0] == '0') force_eager = true;
        }
        lazy_sectors_enabled_ = !force_eager
            && (force_lazy
                || est_bytes > static_cast<long double>(lazy_budget_bytes));
        reverse_index_built_.assign(num_sectors,
                                    lazy_sectors_enabled_ ? char(0) : char(1));

        if (lazy_sectors_enabled_) {
            // LAZY MODE: Pass 1.5 -- norm-only dimension scan.
            std::cout << "Pass 1.5 (lazy streaming; est. eager footprint "
                      << std::fixed << std::setprecision(1)
                      << static_cast<double>(est_bytes
                             / (1024.0L * 1024.0L * 1024.0L))
                      << " GiB > "
                      << (lazy_budget_bytes / (1024.0 * 1024.0 * 1024.0))
                      << " GiB budget): scanning " << num_sectors
                      << " sector dimensions..." << std::endl;
            auto p15_start = std::chrono::high_resolution_clock::now();
            for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
                const auto& sector = sectors_[sector_idx];
                std::atomic<size_t> valid_count{0};
                #pragma omp parallel for schedule(dynamic, 64)
                for (size_t oi = 0; oi < num_orbits; ++oi) {
                    std::vector<uint64_t> elements;
                    std::vector<Complex> coefficients;
                    double norm_sq = 0.0;
                    computeOrbitData(unique_orbit_reps_[oi], sector.phase_factors,
                                     elements, coefficients, norm_sq);
                    if (norm_sq > 1e-10) {
                        valid_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                symmetrized_block_ham_sizes[sector_idx] =
                    static_cast<int>(valid_count.load());
            }
            auto p15_end = std::chrono::high_resolution_clock::now();
            double p15_ms =
                std::chrono::duration<double, std::milli>(p15_end - p15_start)
                    .count();
            size_t total_basis = 0;
            for (int d : symmetrized_block_ham_sizes)
                total_basis += static_cast<size_t>(d);
            std::cout << "\n=== Matrix-Free Sector Generation Complete (lazy) ==="
                      << std::endl;
            std::cout << "Total sectors: " << num_sectors << std::endl;
            std::cout << "Total symmetrized basis: " << total_basis << std::endl;
            std::cout << "Pass 1 time: " << std::fixed << std::setprecision(1)
                      << pass1_ms << " ms" << std::endl;
            std::cout << "Pass 1.5 time: " << std::fixed << std::setprecision(1)
                      << p15_ms << " ms" << std::endl;
            std::cout << "Orbit CSR + reverse index are built one sector at a "
                         "time (LRU-1) at solve/bind time." << std::endl;
            return;
        }

        // =====================================================================
        // EAGER MODE (small systems): original behavior.
        // PASS 2 (parallelizable): Compute orbit data per sector
        // =====================================================================
        std::cout << "Pass 2: Computing orbit data for " << num_sectors
                  << " sectors..." << std::endl;
        
        auto pass2_start = std::chrono::high_resolution_clock::now();
        size_t total_orbit_elements = 0;
        
        for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
            auto& sector = sectors_[sector_idx];
            
            struct OrbitResult {
                uint64_t orbit_rep;
                std::vector<uint64_t> orbit_elements;
                std::vector<Complex> orbit_coefficients;
                double norm;
            };
            
            std::vector<OrbitResult> valid_orbits(num_orbits);
            std::vector<bool> orbit_valid(num_orbits, false);
            
            #pragma omp parallel for schedule(dynamic, 64)
            for (size_t oi = 0; oi < num_orbits; ++oi) {
                uint64_t rep = unique_orbit_reps_[oi];
                std::vector<uint64_t> elements;
                std::vector<Complex> coefficients;
                double norm_sq = 0.0;
                
                computeOrbitData(rep, sector.phase_factors,
                                 elements, coefficients, norm_sq);
                
                if (norm_sq > 1e-10) {
                    valid_orbits[oi].orbit_rep = rep;
                    valid_orbits[oi].orbit_elements = std::move(elements);
                    valid_orbits[oi].orbit_coefficients = std::move(coefficients);
                    valid_orbits[oi].norm = std::sqrt(norm_sq);
                    orbit_valid[oi] = true;
                }
            }
            
            for (size_t oi = 0; oi < num_orbits; ++oi) {
                if (!orbit_valid[oi]) continue;
                
                auto& orb = valid_orbits[oi];
                SymBasisState state(orb.orbit_rep, sector.quantum_numbers, orb.norm);
                state.orbit_elements = std::move(orb.orbit_elements);
                state.orbit_coefficients = std::move(orb.orbit_coefficients);
                // Sort once so all subsequent matvec lookups are O(log |orbit|).
                state.sortOrbit();
                
                size_t basis_idx = sector.basis_states.size();
                for (uint64_t elem : state.orbit_elements) {
                    state_to_sector_basis_[sector_idx][elem] = basis_idx;
                }
                
                total_orbit_elements += state.orbit_elements.size();
                sector.basis_states.push_back(std::move(state));
            }

            // Phase 3a #5: sort the per-sector lookup once now that all
            // (state -> basis_idx) pairs have been appended. All subsequent
            // applySymmetrized() calls expect a finalized index.
            state_to_sector_basis_[sector_idx].finalize();

            symmetrized_block_ham_sizes[sector_idx] = sector.basis_states.size();
            
            if (sector_idx % std::max(size_t(1), num_sectors / 20) == 0 ||
                sector_idx == num_sectors - 1) {
                std::cout << "  Sector " << (sector_idx + 1) << "/" << num_sectors
                          << " -> " << sector.basis_states.size() << " basis states" << std::endl;
            }
        }
        
        auto pass2_end = std::chrono::high_resolution_clock::now();
        double pass2_ms = std::chrono::duration<double, std::milli>(pass2_end - pass2_start).count();
        
        size_t total_basis = 0;
        for (const auto& sector : sectors_) {
            total_basis += sector.basis_states.size();
        }

        // Phase 3a #5: report cumulative lookup-index memory so the savings
        // are visible in the standard build log (compare to the old
        // unordered_map footprint of ~32-40 B/entry vs our 16 B/entry).
        std::size_t lookup_bytes = 0;
        for (const auto& idx : state_to_sector_basis_) {
            lookup_bytes += idx.size_bytes();
        }

        std::cout << "\n=== Matrix-Free Sector Generation Complete ===" << std::endl;
        std::cout << "Total sectors: " << sectors_.size() << std::endl;
        std::cout << "Total symmetrized basis: " << total_basis << std::endl;
        std::cout << "Total orbit elements stored: " << total_orbit_elements << std::endl;
        std::cout << "Pass 1 time: " << std::fixed << std::setprecision(1) << pass1_ms << " ms" << std::endl;
        std::cout << "Pass 2 time: " << std::fixed << std::setprecision(1) << pass2_ms << " ms" << std::endl;
        std::cout << "Lookup index footprint: "
                  << std::fixed << std::setprecision(2)
                  << (lookup_bytes / (1024.0 * 1024.0)) << " MiB ("
                  << (total_orbit_elements > 0
                          ? (double(lookup_bytes) / total_orbit_elements)
                          : 0.0)
                  << " B/entry)" << std::endl;
        std::cout << "Memory saved vs full expansion: " 
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - double(total_orbit_elements) / (total_basis * dim)))
                  << "%" << std::endl;
    }

    // ------------------------------------------------------------------
    // Streaming sector materialization helpers ("stream sym sectors").
    // Twin of the FixedSz block; uses computeOrbitData (full Hilbert).
    // ------------------------------------------------------------------
    void materializeSectorOrbits_(std::size_t k) const {
        auto& sector = sectors_[k];
        if (!sector.basis_states.empty()) return;
        const size_t num_orbits = unique_orbit_reps_.size();
        struct OrbitResult {
            uint64_t orbit_rep;
            std::vector<uint64_t> orbit_elements;
            std::vector<Complex> orbit_coefficients;
            double norm;
        };
        std::vector<OrbitResult> valid_orbits(num_orbits);
        std::vector<char> orbit_valid(num_orbits, 0);
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t oi = 0; oi < num_orbits; ++oi) {
            std::vector<uint64_t> elements;
            std::vector<Complex> coefficients;
            double norm_sq = 0.0;
            computeOrbitData(unique_orbit_reps_[oi], sector.phase_factors,
                             elements, coefficients, norm_sq);
            if (norm_sq > 1e-10) {
                valid_orbits[oi].orbit_rep = unique_orbit_reps_[oi];
                valid_orbits[oi].orbit_elements = std::move(elements);
                valid_orbits[oi].orbit_coefficients = std::move(coefficients);
                valid_orbits[oi].norm = std::sqrt(norm_sq);
                orbit_valid[oi] = 1;
            }
        }
        sector.basis_states.clear();
        if (k < symmetrized_block_ham_sizes.size()) {
            sector.basis_states.reserve(
                static_cast<size_t>(symmetrized_block_ham_sizes[k]));
        }
        for (size_t oi = 0; oi < num_orbits; ++oi) {
            if (!orbit_valid[oi]) continue;
            auto& orb = valid_orbits[oi];
            SymBasisState state(orb.orbit_rep, sector.quantum_numbers, orb.norm);
            state.orbit_elements = std::move(orb.orbit_elements);
            state.orbit_coefficients = std::move(orb.orbit_coefficients);
            state.sortOrbit();
            sector.basis_states.push_back(std::move(state));
        }
    }

    void releaseSectorOrbits_(std::size_t k) const {
        if (k >= sectors_.size()) return;
        std::vector<SymBasisState>().swap(sectors_[k].basis_states);
        if (k < state_to_sector_basis_.size()) {
            state_to_sector_basis_[k] = ed::core::SortedUint64Index{};
        }
        if (k < sector_lookup_dense_.size()) {
            std::vector<std::int32_t>().swap(sector_lookup_dense_[k]);
        }
        if (k < reverse_index_built_.size()) reverse_index_built_[k] = 0;
        dense_lookup_state_.store(0, std::memory_order_release);
    }

    void ensureSectorMaterialized_(std::size_t k) const {
        if (!lazy_sectors_enabled_) return;
        if (materialized_sector_.load(std::memory_order_acquire)
                == static_cast<std::ptrdiff_t>(k)) {
            return;
        }
        std::lock_guard<std::mutex> lock(materialize_mu_);
        if (materialized_sector_.load(std::memory_order_relaxed)
                == static_cast<std::ptrdiff_t>(k)) {
            return;
        }
        const std::ptrdiff_t prev =
            materialized_sector_.load(std::memory_order_relaxed);
        if (prev >= 0) releaseSectorOrbits_(static_cast<std::size_t>(prev));
        materializeSectorOrbits_(k);
        materialized_sector_.store(static_cast<std::ptrdiff_t>(k),
                                   std::memory_order_release);
    }

    void ensureSectorReverseIndex_(std::size_t k) const {
        if (!lazy_sectors_enabled_) return;
        if (k < reverse_index_built_.size() && reverse_index_built_[k]) return;
        std::lock_guard<std::mutex> lock(materialize_mu_);
        if (k < reverse_index_built_.size() && reverse_index_built_[k]) return;
        const auto& sector = sectors_[k];
        auto& idx = state_to_sector_basis_[k];
        idx = ed::core::SortedUint64Index{};
        for (size_t j = 0; j < sector.basis_states.size(); ++j) {
            for (uint64_t elem : sector.basis_states[j].orbit_elements) {
                idx[elem] = j;
            }
        }
        idx.finalize();
        if (k < reverse_index_built_.size()) reverse_index_built_[k] = 1;
    }
    
    /**
     * @brief Matrix-free Hamiltonian application in symmetrized sector
     * 
     * This is the key function that avoids expanding to 2^N dimension.
     * 
     * Algorithm:
     * 1. For each input coefficient c_j with basis state |φ_j⟩
     * 2. For each orbit element |s⟩ in |φ_j⟩ with coefficient α_s
     * 3. Apply each Hamiltonian term to |s⟩ -> |s'⟩ with matrix element h
     * 4. Look up which basis state |φ_k⟩ contains |s'⟩
     * 5. Accumulate: out[k] += c_j * α_s * h * conj(β_{s'}) / (norm_j * norm_k * |G|)
     *    where β_{s'} is the coefficient of |s'⟩ in |φ_k⟩
     * 
     * Memory: O(sector_dim) for output, no 2^N intermediates
     */
    void applySymmetrized(size_t sector_idx, const Complex* in, Complex* out) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        ensureSectorMaterialized_(sector_idx);

        const auto& sector = sectors_[sector_idx];
        const size_t sector_dim = sector.basis_states.size();
        const SectorLookupHandle lookup = makeSectorLookup_(sector_idx);

        // Initialize output
        std::fill(out, out + sector_dim, Complex(0.0, 0.0));

        // Pre-compute normalization factors
        const double group_size = static_cast<double>(getGroupSize());
        const double group_norm = 1.0 / group_size;

        // Wave A3 + A4 (May 2026): parallel matvec over input basis,
        // mirrors the fixed-Sz path's structure. Thread-local scratch
        // (resized in place on growth) replaces the previous per-matvec
        // heap allocation -- significant at small dim where allocator
        // latency dominates.
        #pragma omp parallel if(sector_dim > 100)
        {
            static thread_local std::vector<Complex> local_out;
            local_out.assign(sector_dim, Complex(0.0, 0.0));

            #pragma omp for schedule(dynamic, 4)
            for (size_t j = 0; j < sector_dim; ++j) {
                Complex c_j = in[j];
                if (std::abs(c_j) < 1e-15) continue;

                const auto& state_j = sector.basis_states[j];
                const double norm_j = state_j.norm;

                for (size_t orbit_idx = 0;
                     orbit_idx < state_j.orbit_elements.size(); ++orbit_idx) {
                    uint64_t s = state_j.orbit_elements[orbit_idx];
                    Complex alpha_s = state_j.orbit_coefficients[orbit_idx];
                    if (std::abs(alpha_s) < 1e-15) continue;

                    applyHamiltonianTermsFullSpace(
                        s, c_j * alpha_s / norm_j,
                        sector, lookup, group_norm, local_out);
                }
            }

            #pragma omp critical
            {
                for (size_t k = 0; k < sector_dim; ++k) {
                    out[k] += local_out[k];
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Unified symmetric matvec via ``apply_terms<SymmetryBasisPolicy,
    // Scalar>`` ("Unify all 16 matvec cells" plan, May 2026).
    //
    // The per-term scatter goes through the unified
    // ``ed::matvec::kernel::apply_terms`` kernel that already powers
    // ``Operator::apply`` and ``FixedSzOperator::apply``. The orbit
    // walk + per-emit ``conj(beta) * group_norm / norm_k`` modifier
    // are absorbed via the BasisPolicy ABI extension introduced in
    // Wave 0 (``iter_orbit`` + ``coeff_modifier``).
    //
    // Both methods accept a pre-built ``SectorLookupHandle``. Defined
    // in ``src/symmetry/streaming_symmetry_unified.cpp``.
    // -----------------------------------------------------------------
    void applySymmetrizedUnified(size_t sector_idx,
                                 const Complex* in,
                                 Complex* out) const;
    void applySymmetrizedUnifiedReal(size_t sector_idx,
                                     const double* in,
                                     double* out) const;

private:
    // -----------------------------------------------------------------
    // Wave A2 (May 2026): dense O(1) per-sector reverse lookup table.
    //
    // Each entry is an int32_t (sector-basis index, or -1 for "not in
    // this sector's orbits"). The full Hilbert space is enumerable by
    // ``state``, so the dense table is indexed directly by the
    // computational-basis state value (size 2^n_bits per sector).
    //
    // Gated by ``ED_SYM_DENSE_LOOKUP_BYTES_MAX`` (default 512 MB across
    // all sectors). At N=14 with ~10 sectors this is ~640 KB and the
    // gate is always satisfied; at N=24 with ~24 sectors it would be
    // ~1.5 GB and the gate keeps us on the SortedUint64Index path.
    // -----------------------------------------------------------------
    mutable std::vector<std::vector<std::int32_t>> sector_lookup_dense_;
    mutable std::atomic<int> dense_lookup_state_{0};  // 0=untried, 1=built, 2=skipped
    mutable std::mutex dense_lookup_build_mu_;

    void buildDenseLookupsIfAffordable_() const {
        if (dense_lookup_state_.load(std::memory_order_acquire) != 0) return;
        std::lock_guard<std::mutex> lock(dense_lookup_build_mu_);
        if (dense_lookup_state_.load(std::memory_order_acquire) != 0) return;

        // "stream sym sectors": cross-sector dense lookup is incompatible
        // with lazy single-sector residency. Skip it -- per-sector
        // SortedUint64Index fallback is used instead.
        if (lazy_sectors_enabled_) {
            dense_lookup_state_.store(2, std::memory_order_release);
            return;
        }

        if (sectors_.empty() || state_to_sector_basis_.empty()) {
            dense_lookup_state_.store(2, std::memory_order_release);
            return;
        }

        std::size_t budget_bytes = 512ULL * 1024ULL * 1024ULL;
        if (const char* env = std::getenv("ED_SYM_DENSE_LOOKUP_BYTES_MAX")) {
            try { budget_bytes = std::stoull(env); } catch (...) {}
        }

        const std::uint64_t entries_per_sector = 1ULL << n_bits_;
        const std::size_t bytes_per_sector =
            static_cast<std::size_t>(entries_per_sector) * sizeof(std::int32_t);
        const std::size_t total =
            bytes_per_sector * state_to_sector_basis_.size();
        if (total > budget_bytes) {
            dense_lookup_state_.store(2, std::memory_order_release);
            return;
        }

        sector_lookup_dense_.assign(state_to_sector_basis_.size(),
                                    std::vector<std::int32_t>{});
        for (std::size_t k = 0; k < state_to_sector_basis_.size(); ++k) {
            auto& dense = sector_lookup_dense_[k];
            dense.assign(static_cast<std::size_t>(entries_per_sector), -1);
            const auto& src = state_to_sector_basis_[k];
            const auto& keys = src.keys();
            const auto& vals = src.values();
            for (std::size_t i = 0; i < keys.size(); ++i) {
                dense[static_cast<std::size_t>(keys[i])] =
                    static_cast<std::int32_t>(vals[i]);
            }
        }
        dense_lookup_state_.store(1, std::memory_order_release);
    }

    SectorLookupHandle makeSectorLookup_(std::size_t sector_idx) const {
        // "stream sym sectors": ensure orbit CSR + reverse index resident
        // (no-ops in eager mode).
        ensureSectorMaterialized_(sector_idx);
        ensureSectorReverseIndex_(sector_idx);
        buildDenseLookupsIfAffordable_();
        SectorLookupHandle h;
        h.fallback = &state_to_sector_basis_[sector_idx];
        if (dense_lookup_state_.load(std::memory_order_acquire) == 1
            && sector_idx < sector_lookup_dense_.size()
            && !sector_lookup_dense_[sector_idx].empty()) {
            h.dense = sector_lookup_dense_[sector_idx].data();
            h.lin   = nullptr;  // full-space: state is the index
        }
        return h;
    }

public:
    
    /**
     * @brief Apply Hamiltonian in a specific symmetry sector (vector interface)
     */
    std::vector<Complex> applySymmetrized(size_t sector_idx, 
                                          const std::vector<Complex>& vec) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        ensureSectorMaterialized_(sector_idx);
        
        const size_t sector_dim = sectors_[sector_idx].basis_states.size();
        std::vector<Complex> result(sector_dim);
        
        applySymmetrized(sector_idx, vec.data(), result.data());
        
        return result;
    }

    /// Number of symmetry sectors after generateSymmetrySectorsStreaming().
    [[nodiscard]] std::size_t num_sectors() const noexcept {
        return sectors_.size();
    }

    /**
     * @brief Get sector information
     */
    const SymmetrySector& getSector(size_t sector_idx) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        ensureSectorMaterialized_(sector_idx);
        return sectors_[sector_idx];
    }
    
    size_t getNumSectors() const { return sectors_.size(); }
    
    /**
     * @brief Get dimension of a specific sector
     */
    uint64_t getSectorDimension(size_t sector_idx) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        // "stream sym sectors": dimension known from Pass 1.5 without
        // materializing the orbit CSR.
        if (sector_idx < symmetrized_block_ham_sizes.size()) {
            return static_cast<uint64_t>(
                symmetrized_block_ham_sizes[sector_idx]);
        }
        return sectors_[sector_idx].basis_states.size();
    }

    /**
     * @brief Lookup a computational-basis state's index in the orbit
     *        basis of sector ``sector_idx``.
     *
     * Returns ``ed::core::SortedUint64Index::kNotFound`` (= SIZE_MAX)
     * when ``s`` does not appear in any orbit of that sector.
     *
     * Used by the cross-sector orbit-basis observable
     * (``ed/dssf/cross_sector_orbit_observable.h``) to project a
     * computational-basis target state back into the target sector's
     * orbit basis. Mirrors the private ``state_to_sector_basis_``
     * find call used inside ``applySymmetrized`` but exposed publicly
     * so the cross-sector observable does not need to friend this
     * class.
     */
    std::size_t lookupBasisIndex(std::size_t sector_idx,
                                 std::uint64_t s) const {
        if (sector_idx >= state_to_sector_basis_.size()) {
            return ed::core::SortedUint64Index::kNotFound;
        }
        // "stream sym sectors": ensure orbit CSR + reverse index resident.
        ensureSectorMaterialized_(sector_idx);
        ensureSectorReverseIndex_(sector_idx);
        return state_to_sector_basis_[sector_idx].find(s);
    }

    /**
     * @brief Save sector metadata (lightweight, only orbit representatives)
     */
    void saveSectorMetadata(const std::string& dir) const {
        std::string metadata_dir = dir + "/sym_metadata";
        // P0.12: was safe_system_call("mkdir -p ...").
        std::error_code ec;
        std::filesystem::create_directories(metadata_dir, ec);
        
        // Save sector dimensions
        std::ofstream dim_file(metadata_dir + "/sector_dimensions.txt");
        for (const auto& sector : sectors_) {
            dim_file << sector.basis_states.size() << "\n";
        }
        dim_file.close();
        
        // Save orbit representatives for each sector (binary format for efficiency)
        for (size_t i = 0; i < sectors_.size(); ++i) {
            std::string filename = metadata_dir + "/sector_" + std::to_string(i) + "_orbits.bin";
            std::ofstream file(filename, std::ios::binary);
            
            uint64_t num_states = sectors_[i].basis_states.size();
            file.write(reinterpret_cast<const char*>(&num_states), sizeof(uint64_t));
            
            for (const auto& state : sectors_[i].basis_states) {
                file.write(reinterpret_cast<const char*>(&state.orbit_rep), sizeof(uint64_t));
                file.write(reinterpret_cast<const char*>(&state.norm), sizeof(double));
            }
            
            file.close();
        }
        
        std::cout << "Saved sector metadata to " << metadata_dir << std::endl;
    }
    
    /**
     * @brief Load sector metadata
     */
    void loadSectorMetadata(const std::string& dir) {
        std::string metadata_dir = dir + "/sym_metadata";
        
        // Load sector dimensions
        std::ifstream dim_file(metadata_dir + "/sector_dimensions.txt");
        if (!dim_file.is_open()) {
            throw std::runtime_error("Could not open sector dimensions file");
        }
        
        std::vector<uint64_t> dimensions;
        uint64_t dim;
        while (dim_file >> dim) {
            dimensions.push_back(dim);
        }
        dim_file.close();
        
        // Load orbit representatives for each sector
        sectors_.resize(dimensions.size());
        symmetrized_block_ham_sizes.resize(dimensions.size());
        
        for (size_t i = 0; i < dimensions.size(); ++i) {
            std::string filename = metadata_dir + "/sector_" + std::to_string(i) + "_orbits.bin";
            std::ifstream file(filename, std::ios::binary);
            
            if (!file.is_open()) {
                throw std::runtime_error("Could not open orbit file: " + filename);
            }
            
            uint64_t num_states;
            file.read(reinterpret_cast<char*>(&num_states), sizeof(uint64_t));
            
            sectors_[i].sector_id = i;
            sectors_[i].quantum_numbers = symmetry_info.sectors[i].quantum_numbers;
            sectors_[i].phase_factors = symmetry_info.sectors[i].phase_factors;
            sectors_[i].basis_states.resize(num_states);
            
            for (uint64_t j = 0; j < num_states; ++j) {
                file.read(reinterpret_cast<char*>(&sectors_[i].basis_states[j].orbit_rep), 
                         sizeof(uint64_t));
                file.read(reinterpret_cast<char*>(&sectors_[i].basis_states[j].norm), 
                         sizeof(double));
                sectors_[i].basis_states[j].quantum_numbers = sectors_[i].quantum_numbers;
            }
            
            symmetrized_block_ham_sizes[i] = num_states;
            file.close();
        }
        
        std::cout << "Loaded sector metadata for " << sectors_.size() << " sectors" << std::endl;
    }

    // ===================== HDF5 orbit basis caching (full Hilbert space) =====

    /**
     * @brief Build the cache file path for the full-space (no Sz conservation) case.
     */
    static std::string getOrbitCachePath(const std::string& cache_dir,
                                          uint64_t n_sites) {
        return cache_dir + "/orbit_basis_N" + std::to_string(n_sites)
               + "_fullspace.h5";
    }

    // orbitCacheExists was retired in the minimalist-architecture rev
    // (May 2026): the existence check inside saveOrbitBasisHDF5 /
    // loadOrbitBasisHDF5 (and the HDF5 driver itself) raises if the
    // file is missing, so the standalone probe had no callers.

    /**
     * @brief Save full orbit basis to HDF5 for later reuse (full-space variant).
     *
     * HDF5 layout identical to the fixed-Sz version but without n_up attribute.
     */
    void saveOrbitBasisHDF5(const std::string& cache_dir) const {
        // P0.12: was safe_system_call("mkdir -p ...").
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        std::string filepath = getOrbitCachePath(cache_dir, n_bits_);

        std::cout << "\n=== Saving orbit basis cache (full-space) to "
                  << filepath << " ===" << std::endl;

        try {
            H5::H5File file(filepath, H5F_ACC_TRUNC);
            file.createGroup("/orbit_basis");

            // --- Metadata attributes ---
            {
                H5::DataSpace scalar(H5S_SCALAR);
                auto grp = file.openGroup("/orbit_basis");

                uint64_t ns = n_bits_;
                auto a1 = grp.createAttribute("n_sites",
                    H5::PredType::NATIVE_UINT64, scalar);
                a1.write(H5::PredType::NATIVE_UINT64, &ns);

                uint64_t nsec = sectors_.size();
                auto a2 = grp.createAttribute("num_sectors",
                    H5::PredType::NATIVE_UINT64, scalar);
                a2.write(H5::PredType::NATIVE_UINT64, &nsec);

                uint64_t gsz = symmetry_info.max_clique.size();
                auto a3 = grp.createAttribute("group_size",
                    H5::PredType::NATIVE_UINT64, scalar);
                a3.write(H5::PredType::NATIVE_UINT64, &gsz);
            }

            // --- Per-sector data ---
            for (size_t si = 0; si < sectors_.size(); ++si) {
                // "stream sym sectors": materialize on demand so saving in
                // lazy mode does not serialize empty sectors (LRU-1 keeps
                // peak bounded).
                ensureSectorMaterialized_(si);
                const auto& sector = sectors_[si];
                std::string grp_name = "/orbit_basis/sector_"
                                       + std::to_string(si);
                file.createGroup(grp_name);
                auto grp = file.openGroup(grp_name);

                // Sector attributes
                {
                    H5::DataSpace scalar(H5S_SCALAR);
                    uint64_t sid = sector.sector_id;
                    auto a = grp.createAttribute("sector_id",
                        H5::PredType::NATIVE_UINT64, scalar);
                    a.write(H5::PredType::NATIVE_UINT64, &sid);

                    uint64_t nb = sector.basis_states.size();
                    auto a2 = grp.createAttribute("num_basis",
                        H5::PredType::NATIVE_UINT64, scalar);
                    a2.write(H5::PredType::NATIVE_UINT64, &nb);
                }

                // Quantum numbers
                if (!sector.quantum_numbers.empty()) {
                    hsize_t dims[1] = {sector.quantum_numbers.size()};
                    H5::DataSpace ds(1, dims);
                    auto dset = grp.createDataSet("quantum_numbers",
                        H5::PredType::NATIVE_INT, ds);
                    dset.write(sector.quantum_numbers.data(),
                               H5::PredType::NATIVE_INT);
                }

                // Phase factors (complex → separate real/imag)
                if (!sector.phase_factors.empty()) {
                    std::vector<double> pf_real(sector.phase_factors.size());
                    std::vector<double> pf_imag(sector.phase_factors.size());
                    for (size_t k = 0; k < sector.phase_factors.size(); ++k) {
                        pf_real[k] = sector.phase_factors[k].real();
                        pf_imag[k] = sector.phase_factors[k].imag();
                    }
                    hsize_t dims[1] = {sector.phase_factors.size()};
                    H5::DataSpace ds(1, dims);
                    auto d1 = grp.createDataSet("phase_factors_real",
                        H5::PredType::NATIVE_DOUBLE, ds);
                    d1.write(pf_real.data(), H5::PredType::NATIVE_DOUBLE);
                    auto d2 = grp.createDataSet("phase_factors_imag",
                        H5::PredType::NATIVE_DOUBLE, ds);
                    d2.write(pf_imag.data(), H5::PredType::NATIVE_DOUBLE);
                }

                // --- CSR orbit data ---
                size_t num_basis = sector.basis_states.size();
                std::vector<int64_t> offsets(num_basis + 1);
                std::vector<double> norms(num_basis);
                offsets[0] = 0;
                for (size_t j = 0; j < num_basis; ++j) {
                    offsets[j + 1] = offsets[j]
                        + static_cast<int64_t>(
                              sector.basis_states[j].orbit_elements.size());
                    norms[j] = sector.basis_states[j].norm;
                }
                size_t total_elems = offsets[num_basis];

                std::vector<uint64_t> flat_elements(total_elems);
                std::vector<double>   flat_coeff_real(total_elems);
                std::vector<double>   flat_coeff_imag(total_elems);
                for (size_t j = 0; j < num_basis; ++j) {
                    const auto& bs = sector.basis_states[j];
                    int64_t off = offsets[j];
                    for (size_t e = 0; e < bs.orbit_elements.size(); ++e) {
                        flat_elements[off + e]   = bs.orbit_elements[e];
                        flat_coeff_real[off + e]  = bs.orbit_coefficients[e].real();
                        flat_coeff_imag[off + e]  = bs.orbit_coefficients[e].imag();
                    }
                }

                // Write offsets
                {
                    hsize_t dims[1] = {static_cast<hsize_t>(num_basis + 1)};
                    H5::DataSpace ds(1, dims);
                    auto d = grp.createDataSet("orbit_offsets",
                        H5::PredType::NATIVE_INT64, ds);
                    d.write(offsets.data(), H5::PredType::NATIVE_INT64);
                }
                // Write norms
                {
                    hsize_t dims[1] = {static_cast<hsize_t>(num_basis)};
                    H5::DataSpace ds(1, dims);
                    auto d = grp.createDataSet("orbit_norms",
                        H5::PredType::NATIVE_DOUBLE, ds);
                    d.write(norms.data(), H5::PredType::NATIVE_DOUBLE);
                }
                // Write orbit elements
                if (total_elems > 0) {
                    hsize_t dims[1] = {static_cast<hsize_t>(total_elems)};
                    H5::DataSpace ds(1, dims);
                    auto d = grp.createDataSet("orbit_elements",
                        H5::PredType::NATIVE_UINT64, ds);
                    d.write(flat_elements.data(), H5::PredType::NATIVE_UINT64);
                }
                // Write orbit coefficients
                if (total_elems > 0) {
                    hsize_t dims[1] = {static_cast<hsize_t>(total_elems)};
                    H5::DataSpace ds(1, dims);
                    auto d1 = grp.createDataSet("orbit_coefficients_real",
                        H5::PredType::NATIVE_DOUBLE, ds);
                    d1.write(flat_coeff_real.data(), H5::PredType::NATIVE_DOUBLE);
                    auto d2 = grp.createDataSet("orbit_coefficients_imag",
                        H5::PredType::NATIVE_DOUBLE, ds);
                    d2.write(flat_coeff_imag.data(), H5::PredType::NATIVE_DOUBLE);
                }

                grp.close();
            }

            file.close();

            size_t total_basis = 0, total_orbit = 0;
            for (const auto& s : sectors_) {
                total_basis += s.basis_states.size();
                for (const auto& bs : s.basis_states)
                    total_orbit += bs.orbit_elements.size();
            }
            std::cout << "Cached " << sectors_.size() << " sectors, "
                      << total_basis << " basis states, "
                      << total_orbit << " orbit elements" << std::endl;
            std::cout << "=== Orbit basis cache saved ===" << std::endl;

        } catch (H5::Exception& e) {
            std::cerr << "\nError: Failed to save orbit basis cache: "
                      << e.getCDetailMsg() << std::endl;
            // Non-fatal: sectors are already in memory, diagonalization can proceed
        }
    }

    /**
     * @brief Load orbit basis from HDF5 cache (full-space variant).
     *
     * Restores sectors_, state_to_sector_basis_, and symmetrized_block_ham_sizes.
     * @return true on success, false if cache not found / mismatch.
     */
    bool loadOrbitBasisHDF5(const std::string& cache_dir) {
        std::string filepath = getOrbitCachePath(cache_dir, n_bits_);

        {
            std::ifstream f(filepath);
            if (!f.good()) return false;
        }

        std::cout << "\n=== Loading orbit basis cache (full-space) from "
                  << filepath << " ===" << std::endl;

        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            auto root = file.openGroup("/orbit_basis");

            // Verify metadata
            uint64_t cached_n_sites, cached_num_sectors, cached_group_size;
            root.openAttribute("n_sites").read(
                H5::PredType::NATIVE_UINT64, &cached_n_sites);
            root.openAttribute("num_sectors").read(
                H5::PredType::NATIVE_UINT64, &cached_num_sectors);
            root.openAttribute("group_size").read(
                H5::PredType::NATIVE_UINT64, &cached_group_size);

            if (cached_n_sites != n_bits_) {
                std::cerr << "Cache mismatch: n_sites " << cached_n_sites
                          << " vs " << n_bits_ << std::endl;
                return false;
            }

            // Store group size from cache so that matvec / expansion code
            // can function without loading symmetry_info from disk.
            cached_group_size_ = cached_group_size;
            std::cout << "  group_size (from cache): " << cached_group_size_ << std::endl;

            // Allocate sector storage
            sectors_.resize(cached_num_sectors);
            symmetrized_block_ham_sizes.assign(cached_num_sectors, 0);
            state_to_sector_basis_.resize(cached_num_sectors);

            size_t total_orbit_elements = 0;

            for (size_t si = 0; si < cached_num_sectors; ++si) {
                std::string grp_name = "/orbit_basis/sector_"
                                       + std::to_string(si);
                auto grp = file.openGroup(grp_name);
                auto& sector = sectors_[si];

                // Sector attributes
                uint64_t sid, nb;
                grp.openAttribute("sector_id").read(
                    H5::PredType::NATIVE_UINT64, &sid);
                grp.openAttribute("num_basis").read(
                    H5::PredType::NATIVE_UINT64, &nb);
                sector.sector_id = sid;

                // Quantum numbers
                {
                    auto dset = grp.openDataSet("quantum_numbers");
                    auto space = dset.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    sector.quantum_numbers.resize(dims[0]);
                    dset.read(sector.quantum_numbers.data(),
                              H5::PredType::NATIVE_INT);
                }

                // Phase factors
                {
                    auto d1 = grp.openDataSet("phase_factors_real");
                    auto d2 = grp.openDataSet("phase_factors_imag");
                    auto space = d1.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    std::vector<double> pf_real(dims[0]), pf_imag(dims[0]);
                    d1.read(pf_real.data(), H5::PredType::NATIVE_DOUBLE);
                    d2.read(pf_imag.data(), H5::PredType::NATIVE_DOUBLE);
                    sector.phase_factors.resize(dims[0]);
                    for (size_t k = 0; k < dims[0]; ++k) {
                        sector.phase_factors[k] =
                            Complex(pf_real[k], pf_imag[k]);
                    }
                }

                // CSR orbit data
                std::vector<int64_t> offsets;
                std::vector<double> norms;
                {
                    auto dset = grp.openDataSet("orbit_offsets");
                    auto space = dset.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    offsets.resize(dims[0]);
                    dset.read(offsets.data(), H5::PredType::NATIVE_INT64);
                }
                {
                    auto dset = grp.openDataSet("orbit_norms");
                    auto space = dset.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    norms.resize(dims[0]);
                    dset.read(norms.data(), H5::PredType::NATIVE_DOUBLE);
                }

                size_t total_elems = (offsets.size() > 1)
                                         ? offsets.back() : 0;
                std::vector<uint64_t> flat_elements;
                std::vector<double> flat_coeff_real, flat_coeff_imag;
                if (total_elems > 0) {
                    flat_elements.resize(total_elems);
                    flat_coeff_real.resize(total_elems);
                    flat_coeff_imag.resize(total_elems);
                    grp.openDataSet("orbit_elements")
                        .read(flat_elements.data(),
                              H5::PredType::NATIVE_UINT64);
                    grp.openDataSet("orbit_coefficients_real")
                        .read(flat_coeff_real.data(),
                              H5::PredType::NATIVE_DOUBLE);
                    grp.openDataSet("orbit_coefficients_imag")
                        .read(flat_coeff_imag.data(),
                              H5::PredType::NATIVE_DOUBLE);
                }

                // Reconstruct SymBasisState objects and lookup table
                sector.basis_states.resize(nb);
                for (size_t j = 0; j < nb; ++j) {
                    auto& bs = sector.basis_states[j];
                    bs.quantum_numbers = sector.quantum_numbers;
                    bs.norm = norms[j];
                    int off = offsets[j];
                    int len = offsets[j + 1] - off;
                    bs.orbit_elements.resize(len);
                    bs.orbit_coefficients.resize(len);
                    for (int e = 0; e < len; ++e) {
                        bs.orbit_elements[e]    = flat_elements[off + e];
                        bs.orbit_coefficients[e] =
                            Complex(flat_coeff_real[off + e],
                                    flat_coeff_imag[off + e]);
                    }
                    // Sort once -> O(log |orbit|) lookups in applySymmetrized.
                    // After sorting, [0] is the canonical (smallest) representative.
                    bs.sortOrbit();
                    bs.orbit_rep = bs.orbit_elements.empty()
                                       ? 0 : bs.orbit_elements[0];

                    // Rebuild lookup table
                    for (uint64_t elem : bs.orbit_elements) {
                        state_to_sector_basis_[si][elem] = j;
                    }
                    total_orbit_elements += len;
                }

                // Phase 3a #5: finalize the lookup index for this sector
                // before the next outer iteration moves on.
                state_to_sector_basis_[si].finalize();

                symmetrized_block_ham_sizes[si] = nb;
                grp.close();
            }

            file.close();

            size_t total_basis = 0;
            for (const auto& s : sectors_)
                total_basis += s.basis_states.size();

            std::cout << "Loaded " << sectors_.size() << " sectors, "
                      << total_basis << " basis states, "
                      << total_orbit_elements << " orbit elements"
                      << std::endl;

            // Validate: every non-empty sector must have orbit elements
            for (size_t si = 0; si < sectors_.size(); ++si) {
                const auto& sector = sectors_[si];
                if (!sector.basis_states.empty()) {
                    bool has_empty = false;
                    for (const auto& bs : sector.basis_states) {
                        if (bs.orbit_elements.empty()) { has_empty = true; break; }
                    }
                    if (has_empty) {
                        std::cerr << "Warning: Sector " << si
                                  << " has basis states with empty orbit data "
                                  << "— cache may be corrupted, regenerating"
                                  << std::endl;
                        sectors_.clear();
                        state_to_sector_basis_.clear();
                        symmetrized_block_ham_sizes.clear();
                        return false;
                    }
                }
            }

            std::cout << "=== Orbit basis cache loaded ===" << std::endl;
            return true;

        } catch (H5::Exception& e) {
            std::cerr << "Warning: Failed to load orbit basis cache: "
                      << e.getCDetailMsg() << std::endl;
            return false;
        }
    }

    // ===================== End HDF5 orbit basis caching =====================

    // ===================== Eigenvector expansion ============================

    /**
     * @brief Expand a symmetrized-sector eigenvector to the full 2^N computational basis.
     *
     * Given an eigenvector c = (c_0, c_1, ..., c_{D-1}) in the symmetrized
     * sector basis (D = sector dimension), the full-basis vector is:
     *   |ψ⟩ = Σ_j c_j |φ_j⟩ = Σ_j c_j (1/N_j) Σ_k α_{jk} |s_{jk}⟩
     * where orbit_elements = {s_{jk}}, orbit_coefficients = {α_{jk}}, norm = N_j.
     *
     * @param sector_idx  Index of the symmetry sector
     * @param sym_vec     Eigenvector in the symmetrized sector basis
     * @return Vector of length 2^N in computational basis
     */
    std::vector<Complex> expandToComputationalBasis(
        size_t sector_idx,
        const std::vector<Complex>& sym_vec
    ) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        ensureSectorMaterialized_(sector_idx);
        const auto& sector = sectors_[sector_idx];
        size_t sector_dim = sector.basis_states.size();
        if (sym_vec.size() != sector_dim) {
            throw std::runtime_error(
                "Eigenvector size (" + std::to_string(sym_vec.size())
                + ") != sector dimension (" + std::to_string(sector_dim) + ")");
        }

        uint64_t full_dim = 1ULL << n_bits_;
        std::vector<Complex> full_vec(full_dim, Complex(0.0, 0.0));

        const double group_norm = 1.0 / std::sqrt(
            static_cast<double>(getGroupSize()));

        for (size_t j = 0; j < sector_dim; ++j) {
            if (std::abs(sym_vec[j]) < 1e-15) continue;
            const auto& bs = sector.basis_states[j];
            Complex weight = sym_vec[j] * group_norm / bs.norm;
            for (size_t k = 0; k < bs.orbit_elements.size(); ++k) {
                uint64_t s = bs.orbit_elements[k];
                full_vec[s] += weight * bs.orbit_coefficients[k];
            }
        }
        return full_vec;
    }

    // ===================== End eigenvector expansion =========================
    
private:
    // getOrbitRepresentativeFast / computeOrbitElements /
    // computeSymmetrizedNorm were retired in the minimalist-architecture
    // rev (May 2026): no callers. The streaming-symmetry path uses the
    // standard `getOrbitRepresentative` (cache-free; populated during
    // sector generation) and computes norms inline in
    // `generateSymmetrySectorsStreaming`, so the standalone helpers had
    // become vestigial.

    
    /**
     * @brief Compute orbit elements and coefficients for a basis state in a sector
     *
     * Returns the orbit elements (computational basis states) and their
     * corresponding complex coefficients in the symmetrized state.
     * This is the full-space version (no fixed-Sz restriction).
     *
     * Implementation (May 2026, "Orthogonal symmetry composition"):
     * delegates to the templated
     * ``ed::symmetry::compute_orbit_for_state`` helper, which is the
     * single source of truth for the orbit/character composition
     * shared with the fixed-Sz variant. The output is bit-identical
     * to the legacy inline loop (pinned by
     * ``tests/unit/test_projector_chain.cpp``).
     */
    void computeOrbitData(uint64_t basis,
                          const std::vector<Complex>& phase_factors,
                          std::vector<uint64_t>& orbit_elements,
                          std::vector<Complex>& orbit_coefficients,
                          double& norm_sq) const {
        const ed::symmetry::FullSpaceSubspace full(n_bits_);
        const ed::symmetry::SpatialProjector  spatial(symmetry_info);
        ed::symmetry::compute_orbit_for_state(
            full, spatial, basis, phase_factors,
            orbit_elements, orbit_coefficients, norm_sq);
    }
    
    /**
     * @brief Apply all Hamiltonian terms to a single computational basis state
     * 
     * This is the inner loop of the matrix-free multiplication.
     * Projects results onto the symmetrized basis using the lookup table.
     * Full-space version (no fixed-Sz restriction).
     */
    void applyHamiltonianTermsFullSpace(uint64_t s, Complex weighted_coeff,
                                        const SymmetrySector& sector,
                                        const SectorLookupHandle& lookup,
                                        double group_norm,
                                        std::vector<Complex>& local_out) const {

        // Helper lambda to project result onto sector. Wave A2: lookup
        // is now a SectorLookupHandle; ``.find()`` inlines to either an
        // O(1) dense table read or the SortedUint64Index binary search.
        auto projectResult = [&](uint64_t s_prime, Complex h_element) {
            const std::size_t k = lookup.find(s_prime);
            if (k == ed::core::SortedUint64Index::kNotFound) return;

            const auto& state_k = sector.basis_states[k];
            
            // O(log |orbit|) binary search instead of O(|orbit|) linear scan.
            // For groups of order ~768 (cubic + translation), this is a ~20-50x
            // speedup per call and the function is the SpMV inner loop.
            const Complex beta_s_prime = state_k.findCoeff(s_prime);
            
            // Accumulate: out[k] += weighted_coeff * h * conj(β_{s'}) / norm_k
            local_out[k] += weighted_coeff * h_element * std::conj(beta_s_prime) * group_norm / state_k.norm;
        };
        
        // Apply each one/two-body term from transform_data_
        for (const auto& tdata : transform_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;
            
            if (!tdata.is_two_body) {
                // One-body: S^α_i
                if (tdata.op_type == 2) {
                    // Sz: diagonal
                    double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    // S+ or S-: flip bit
                    uint64_t bit = (s >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
            } else {
                // Two-body: S^α_i S^β_j
                uint64_t bit_i = (s >> tdata.site_index) & 1;
                uint64_t bit_j = (s >> tdata.site_index_2) & 1;
                
                if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                    // Sz_i Sz_j: diagonal
                    double sign_i = bit_i ? -1.0 : 1.0;
                    double sign_j = bit_j ? -1.0 : 1.0;
                    h_element *= spin_l_ * spin_l_ * sign_i * sign_j;
                } else {
                    // Mixed terms
                    if (tdata.op_type != 2) {
                        if (bit_i != tdata.op_type) {
                            s_prime ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    } else {
                        double sign_i = bit_i ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_i;
                    }
                    
                    if (valid && tdata.op_type_2 != 2) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        if (new_bit_j != tdata.op_type_2) {
                            s_prime ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    } else if (valid) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        double sign_j = new_bit_j ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_j;
                    }
                }
            }
            
            if (valid) {
                projectResult(s_prime, h_element);
            }
        }
        
        // Apply three-body terms from three_body_data_
        for (const auto& tdata : three_body_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;
            
            // Apply first operator
            if (tdata.op_type_1 == 2) {
                uint64_t bit = (s_prime >> tdata.site_index_1) & 1;
                double sign = bit ? -1.0 : 1.0;
                h_element *= spin_l_ * sign;
            } else {
                uint64_t bit = (s_prime >> tdata.site_index_1) & 1;
                if (bit != tdata.op_type_1) {
                    s_prime ^= (1ULL << tdata.site_index_1);
                } else {
                    valid = false;
                }
            }
            
            // Apply second operator
            if (valid) {
                if (tdata.op_type_2 == 2) {
                    uint64_t bit = (s_prime >> tdata.site_index_2) & 1;
                    double sign = bit ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> tdata.site_index_2) & 1;
                    if (bit != tdata.op_type_2) {
                        s_prime ^= (1ULL << tdata.site_index_2);
                    } else {
                        valid = false;
                    }
                }
            }
            
            // Apply third operator
            if (valid) {
                if (tdata.op_type_3 == 2) {
                    uint64_t bit = (s_prime >> tdata.site_index_3) & 1;
                    double sign = bit ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> tdata.site_index_3) & 1;
                    if (bit != tdata.op_type_3) {
                        s_prime ^= (1ULL << tdata.site_index_3);
                    } else {
                        valid = false;
                    }
                }
            }
            
            if (valid) {
                projectResult(s_prime, h_element);
            }
        }
    }

    // expandSymmetrizedState / projectOntoSymmetrizedState / applyFullSpace
    // were retired in the minimalist-architecture rev (May 2026): no
    // callers. The streaming-symmetry path projects the Hamiltonian
    // action directly inside `applySymmetrized` (sector_idx) via
    // `applyHamiltonianTermsFullSpace`, so these standalone full-space
    // helpers had become vestigial.

    // -----------------------------------------------------------------
    // Wave A1 (May 2026): real-arithmetic twin of
    // ``applyHamiltonianTermsFullSpace``. Identical bit-flip / Sz / phase
    // logic, but operates on ``double`` arithmetic throughout. Caller
    // (``applySymmetrizedReal``) must have gated on
    // ``isSectorReal(sector_idx) == true``; behaviour is undefined when
    // either the Hamiltonian or the orbit coefficients have a non-zero
    // imaginary part.
    // -----------------------------------------------------------------
    void applyHamiltonianTermsFullSpaceReal(
        uint64_t s, double weighted_coeff,
        const SymmetrySector& sector,
        const SectorLookupHandle& lookup,
        double group_norm,
        std::vector<double>& local_out) const {
        auto projectResult = [&](uint64_t s_prime, double h_element) {
            const std::size_t k = lookup.find(s_prime);
            if (k == ed::core::SortedUint64Index::kNotFound) return;
            const auto& state_k = sector.basis_states[k];
            // beta_{s'} is real under the wave-A1 gating; .real() avoids
            // an extra conj()->real shuttle.
            const double beta_s_prime = state_k.findCoeff(s_prime).real();
            local_out[k] += weighted_coeff * h_element * beta_s_prime
                          * group_norm / state_k.norm;
        };

        for (const auto& tdata : transform_data_) {
            uint64_t s_prime = s;
            double h_element = tdata.coefficient.real();
            bool valid = true;

            if (!tdata.is_two_body) {
                if (tdata.op_type == 2) {
                    double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
            } else {
                uint64_t bit_i = (s >> tdata.site_index) & 1;
                uint64_t bit_j = (s >> tdata.site_index_2) & 1;
                if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                    double sign_i = bit_i ? -1.0 : 1.0;
                    double sign_j = bit_j ? -1.0 : 1.0;
                    h_element *= spin_l_ * spin_l_ * sign_i * sign_j;
                } else {
                    if (tdata.op_type != 2) {
                        if (bit_i != tdata.op_type) {
                            s_prime ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    } else {
                        double sign_i = bit_i ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_i;
                    }
                    if (valid && tdata.op_type_2 != 2) {
                        uint64_t nb = (s_prime >> tdata.site_index_2) & 1;
                        if (nb != tdata.op_type_2) {
                            s_prime ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    } else if (valid) {
                        uint64_t nb = (s_prime >> tdata.site_index_2) & 1;
                        double sign_j = nb ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_j;
                    }
                }
            }
            if (valid) projectResult(s_prime, h_element);
        }

        for (const auto& tdata : three_body_data_) {
            uint64_t s_prime = s;
            double h_element = tdata.coefficient.real();
            bool valid = true;

            auto apply_one = [&](int op_type, int site_idx) {
                if (!valid) return;
                if (op_type == 2) {
                    uint64_t bit = (s_prime >> site_idx) & 1;
                    double sign = bit ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> site_idx) & 1;
                    if (static_cast<int>(bit) != op_type) {
                        s_prime ^= (1ULL << site_idx);
                    } else {
                        valid = false;
                    }
                }
            };
            apply_one(tdata.op_type_1, tdata.site_index_1);
            apply_one(tdata.op_type_2, tdata.site_index_2);
            apply_one(tdata.op_type_3, tdata.site_index_3);
            if (valid) projectResult(s_prime, h_element);
        }
    }
};

// ============================================================================
// Fixed Sz Streaming Symmetry Operator (MATRIX-FREE)
// ============================================================================

/**
 * @brief Truly matrix-free symmetry operator for fixed Sz sector
 * 
 * This implementation is memory-efficient: it doesn't expand to the full
 * fixed-Sz dimension. Instead, it works directly with orbit elements.
 * 
 * Memory complexity: O(sector_dim × average_orbit_size)
 * Time complexity: O(sector_dim × orbit_size × num_hamiltonian_terms)
 * 
 * Key optimizations:
 * 1. Pre-compute orbit elements and coefficients during sector generation
 * 2. Build lookup table: computational_state -> (sector_basis_index)
 * 3. Apply H term-by-term on orbit elements only
 * 4. Project results using orbit lookup (no full-dim vectors)
 */
class FixedSzStreamingSymmetryOperator : public FixedSzOperator {
private:
    // Mutable so that the lazy per-sector orbit materialization below can
    // (re)build sectors_[k].basis_states from inside const matvec / bind
    // entry points. See the "Streaming sector materialization" block.
    mutable std::vector<SymmetrySector> sectors_;
    
    // Lookup table: computational_state -> basis_idx_in_sector (one per sector).
    // Phase 3a #5: same SortedUint64Index swap as in StreamingSymmetryOperator
    // (see comment there). 16 B/entry vs ~32-40 B/entry for unordered_map.
    mutable std::vector<ed::core::SortedUint64Index> state_to_sector_basis_;

    // ------------------------------------------------------------------
    // Streaming sector materialization ("stream sym sectors" plan,
    // Jun 2026). At N=32, n_up=16, |G|=8 the eager Pass 2 used to build
    // every sector's orbit CSR (~115 GB) AND the CPU-only reverse index
    // (~77 GB) simultaneously -> host OOM. We now keep only the cheap
    // unique-orbit-rep list resident and build the orbit CSR for one
    // sector on demand (LRU-1), freeing the previous one, and we build
    // the reverse index lazily only when a CPU matvec actually needs it
    // (the GPU mirror never reads it).
    //
    // ``lazy_sectors_enabled_`` is decided at generation time from the
    // estimated all-sector footprint; small systems stay fully eager so
    // the dense-lookup fast path and existing test behavior are
    // unchanged.
    // ------------------------------------------------------------------
    std::vector<uint64_t>        unique_orbit_reps_;       // Pass 1 result (kept resident)
    bool                         lazy_sectors_enabled_ = false;
    mutable std::atomic<std::ptrdiff_t> materialized_sector_{-1};
    mutable std::vector<char>    reverse_index_built_;     // per-sector flag
    mutable std::mutex           materialize_mu_;
    
    mutable ShardedOrbitCache state_to_orbit_cache_;  // Striped-lock cache (16 shards) -- lower contention than single-mutex map
    
    // Cached group size from HDF5 load (allows skipping symmetry_info loading)
    uint64_t cached_group_size_ = 0;

public:
    FixedSzStreamingSymmetryOperator(uint64_t n_bits, float spin_l, int64_t n_up)
        : FixedSzOperator(n_bits, spin_l, n_up) {
        if (n_bits >= 64) {
            throw std::runtime_error("FixedSzStreamingSymmetryOperator: n_bits = " + std::to_string(n_bits)
                + " >= 64 is not supported (would cause undefined behavior in 1ULL << n_bits)");
        }
    }

    /**
     * @brief Get the symmetry group size.
     * Returns symmetry_info.max_clique.size() if loaded, else the value
     * cached from an HDF5 basis load.
     */
    uint64_t getGroupSize() const {
        if (!symmetry_info.max_clique.empty())
            return symmetry_info.max_clique.size();
        if (cached_group_size_ > 0)
            return cached_group_size_;
        throw std::runtime_error("getGroupSize: neither symmetry_info nor "
                                 "cached group size available");
    }
    
    /**
     * @brief Generate symmetry sectors with pre-computed orbit data (optimized)
     * 
     * This version pre-computes orbit elements and coefficients, and builds
     * lookup tables for efficient matrix-free H*v computation.
     */
    void generateSymmetrySectorsStreamingFixedSz(const std::string& dir) {
        std::cout << "\n=== Generating Symmetry Sectors (Matrix-Free, Fixed Sz) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        const size_t num_sectors = symmetry_info.sectors.size();
        sectors_.resize(num_sectors);
        symmetrized_block_ham_sizes.assign(num_sectors, 0);
        state_to_sector_basis_.resize(num_sectors);
        
        // =====================================================================
        // PASS 1 (parallelizable): Identify unique orbit representatives
        // This avoids re-scanning all basis states for every sector.
        // =====================================================================
        std::cout << "Pass 1: Identifying unique orbits (" << basis_states_.size()
                  << " basis states, group size " << symmetry_info.max_clique.size()
                  << ")..." << std::flush;
        
        auto pass1_start = std::chrono::high_resolution_clock::now();
        
        // Single streaming pass: compute rep(basis) and dedupe without storing
        // orbit_reps[num_basis] (~8 bytes per Sz-sector basis element). The
        // deduped rep list is kept resident in ``unique_orbit_reps_`` so the
        // lazy per-sector materialization can rebuild orbit CSR on demand.
        const size_t num_basis = basis_states_.size();
        std::unordered_set<uint64_t> seen_reps;
        unique_orbit_reps_.clear();
        unique_orbit_reps_.reserve(num_basis / symmetry_info.max_clique.size() + 1);
        
        for (size_t i = 0; i < num_basis; ++i) {
            uint64_t basis = basis_states_[i];
            uint64_t rep = basis;
            for (const auto& perm : symmetry_info.max_clique) {
                uint64_t permuted = applyPermutation(basis, perm);
                // Site permutations preserve popcount, so lookupState always
                // returns >= 0 for permuted images of basis states. We keep
                // the check defensively in case applyPermutation evolves.
                if (lookupState(permuted) >= 0 && permuted < rep) {
                    rep = permuted;
                }
            }
            if (seen_reps.insert(rep).second) {
                unique_orbit_reps_.push_back(rep);
            }
        }
        std::unordered_set<uint64_t>().swap(seen_reps);  // free the dedupe set
        
        auto pass1_end = std::chrono::high_resolution_clock::now();
        double pass1_ms = std::chrono::duration<double, std::milli>(pass1_end - pass1_start).count();
        const size_t num_orbits = unique_orbit_reps_.size();
        std::cout << " found " << num_orbits << " unique orbits"
                  << " (" << std::fixed << std::setprecision(1) << pass1_ms << " ms)" << std::endl;

        // Copy per-sector metadata (sector_id / quantum numbers / phase
        // factors) up front -- cheap and needed by both eager and lazy modes.
        for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
            const auto& sector_meta = symmetry_info.sectors[sector_idx];
            auto& sector = sectors_[sector_idx];
            sector.sector_id       = sector_meta.sector_id;
            sector.quantum_numbers = sector_meta.quantum_numbers;
            sector.phase_factors   = sector_meta.phase_factors;
        }

        // =====================================================================
        // Decide eager vs lazy ("stream sym sectors" plan, Jun 2026). Estimate
        // the all-sector resident footprint (orbit CSR ~24 B/elem + reverse
        // index ~16 B/elem) using the upper bound
        //   total_orbit_elements ~ num_orbits * |G| * num_sectors.
        // Small systems stay fully eager (build everything now) so the
        // dense-lookup fast path and existing test behavior are unchanged.
        // Large systems (e.g. N=32, n_up=16, |G|=8 -> ~192 GB) go lazy.
        // =====================================================================
        std::size_t lazy_budget_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4 GiB
        if (const char* env = std::getenv("ED_SYM_LAZY_SECTORS_BYTES_MAX")) {
            try { lazy_budget_bytes = std::stoull(env); } catch (...) {}
        }
        const std::size_t group_sz =
            std::max<std::size_t>(1, symmetry_info.max_clique.size());
        const long double est_elems =
            static_cast<long double>(num_orbits) * group_sz * num_sectors;
        const long double est_bytes = est_elems * (24.0L + 16.0L);
        bool force_lazy = false, force_eager = false;
        if (const char* env = std::getenv("ED_SYM_LAZY_SECTORS")) {
            if (env[0] == '1') force_lazy = true;
            else if (env[0] == '0') force_eager = true;
        }
        lazy_sectors_enabled_ = !force_eager
            && (force_lazy
                || est_bytes > static_cast<long double>(lazy_budget_bytes));
        reverse_index_built_.assign(num_sectors,
                                    lazy_sectors_enabled_ ? char(0) : char(1));

        if (lazy_sectors_enabled_) {
            // -----------------------------------------------------------
            // LAZY MODE: Pass 1.5 -- norm-only dimension scan. Walk each
            // orbit per sector to count valid (norm_sq > tol) orbits to
            // get the per-sector dimension WITHOUT storing the orbit CSR
            // or the reverse index. O(1) extra memory. The orbit CSR is
            // then built on demand (LRU-1) in materializeSectorOrbits_,
            // and the CPU reverse index lazily in makeSectorLookup_.
            // -----------------------------------------------------------
            std::cout << "Pass 1.5 (lazy streaming; est. eager footprint "
                      << std::fixed << std::setprecision(1)
                      << static_cast<double>(est_bytes
                             / (1024.0L * 1024.0L * 1024.0L))
                      << " GiB > "
                      << (lazy_budget_bytes / (1024.0 * 1024.0 * 1024.0))
                      << " GiB budget): scanning " << num_sectors
                      << " sector dimensions..." << std::endl;
            auto p15_start = std::chrono::high_resolution_clock::now();
            for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
                const auto& sector = sectors_[sector_idx];
                std::atomic<size_t> valid_count{0};
                #pragma omp parallel for schedule(dynamic, 64)
                for (size_t oi = 0; oi < num_orbits; ++oi) {
                    std::vector<uint64_t> elements;
                    std::vector<Complex> coefficients;
                    double norm_sq = 0.0;
                    computeOrbitDataFixedSz(unique_orbit_reps_[oi],
                                            sector.phase_factors,
                                            elements, coefficients, norm_sq);
                    if (norm_sq > 1e-10) {
                        valid_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                symmetrized_block_ham_sizes[sector_idx] =
                    static_cast<int>(valid_count.load());
            }
            auto p15_end = std::chrono::high_resolution_clock::now();
            double p15_ms =
                std::chrono::duration<double, std::milli>(p15_end - p15_start)
                    .count();

            size_t total_basis = 0;
            for (int d : symmetrized_block_ham_sizes)
                total_basis += static_cast<size_t>(d);
            std::cout << "\n=== Matrix-Free Sector Generation Complete (lazy) ==="
                      << std::endl;
            std::cout << "Total sectors: " << num_sectors << std::endl;
            std::cout << "Total symmetrized basis: " << total_basis << std::endl;
            std::cout << "Pass 1 time: " << std::fixed << std::setprecision(1)
                      << pass1_ms << " ms" << std::endl;
            std::cout << "Pass 1.5 time: " << std::fixed << std::setprecision(1)
                      << p15_ms << " ms" << std::endl;
            std::cout << "Orbit CSR + reverse index are built one sector at a "
                         "time (LRU-1) at solve/bind time." << std::endl;
            return;
        }

        // =====================================================================
        // EAGER MODE (small systems): original behavior -- build every
        // sector's orbit CSR + reverse index now.
        //
        // PASS 2 (parallelizable): For each sector, compute orbit data for
        // each unique orbit rep.  The orbit data (elements + coefficients) is
        // phase-factor-dependent, so we must do this per sector.  However the
        // inner loop over orbit reps is embarrassingly parallel.
        // =====================================================================
        std::cout << "Pass 2: Computing orbit data for " << num_sectors
                  << " sectors..." << std::endl;
        
        auto pass2_start = std::chrono::high_resolution_clock::now();
        size_t total_orbit_elements = 0;
        
        for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
            auto& sector = sectors_[sector_idx];
            
            // Parallel orbit data computation for this sector
            // Each thread computes orbit data for a subset of orbit reps
            struct OrbitResult {
                uint64_t orbit_rep;
                std::vector<uint64_t> orbit_elements;
                std::vector<Complex> orbit_coefficients;
                double norm;
            };
            
            std::vector<OrbitResult> valid_orbits(num_orbits);
            std::vector<bool> orbit_valid(num_orbits, false);
            
            #pragma omp parallel for schedule(dynamic, 64)
            for (size_t oi = 0; oi < num_orbits; ++oi) {
                uint64_t rep = unique_orbit_reps_[oi];
                std::vector<uint64_t> elements;
                std::vector<Complex> coefficients;
                double norm_sq = 0.0;
                
                computeOrbitDataFixedSz(rep, sector.phase_factors,
                                        elements, coefficients, norm_sq);
                
                if (norm_sq > 1e-10) {
                    valid_orbits[oi].orbit_rep = rep;
                    valid_orbits[oi].orbit_elements = std::move(elements);
                    valid_orbits[oi].orbit_coefficients = std::move(coefficients);
                    valid_orbits[oi].norm = std::sqrt(norm_sq);
                    orbit_valid[oi] = true;
                }
            }
            
            // Sequential gathering of valid orbits into sector (maintains deterministic order)
            for (size_t oi = 0; oi < num_orbits; ++oi) {
                if (!orbit_valid[oi]) continue;
                
                auto& orb = valid_orbits[oi];
                SymBasisState state(orb.orbit_rep, sector.quantum_numbers, orb.norm);
                state.orbit_elements = std::move(orb.orbit_elements);
                state.orbit_coefficients = std::move(orb.orbit_coefficients);
                // Sort once so all subsequent matvec lookups are O(log |orbit|).
                state.sortOrbit();
                
                size_t basis_idx = sector.basis_states.size();
                for (uint64_t elem : state.orbit_elements) {
                    state_to_sector_basis_[sector_idx][elem] = basis_idx;
                }
                
                total_orbit_elements += state.orbit_elements.size();
                sector.basis_states.push_back(std::move(state));
            }

            // Phase 3a #5: sort the per-sector lookup once now that all
            // (state -> basis_idx) pairs have been appended.
            state_to_sector_basis_[sector_idx].finalize();

            symmetrized_block_ham_sizes[sector_idx] = sector.basis_states.size();
            
            if (sector_idx % std::max(size_t(1), num_sectors / 20) == 0 ||
                sector_idx == num_sectors - 1) {
                std::cout << "  Sector " << (sector_idx + 1) << "/" << num_sectors
                          << " -> " << sector.basis_states.size() << " basis states" << std::endl;
            }
        }
        
        auto pass2_end = std::chrono::high_resolution_clock::now();
        double pass2_ms = std::chrono::duration<double, std::milli>(pass2_end - pass2_start).count();
        
        size_t total_basis = 0;
        for (const auto& sector : sectors_) {
            total_basis += sector.basis_states.size();
        }

        // Phase 3a #5: report cumulative lookup-index memory.
        std::size_t lookup_bytes = 0;
        for (const auto& idx : state_to_sector_basis_) {
            lookup_bytes += idx.size_bytes();
        }

        std::cout << "\n=== Matrix-Free Sector Generation Complete ===" << std::endl;
        std::cout << "Total sectors: " << sectors_.size() << std::endl;
        std::cout << "Total symmetrized basis: " << total_basis << std::endl;
        std::cout << "Total orbit elements stored: " << total_orbit_elements << std::endl;
        std::cout << "Pass 1 time: " << std::fixed << std::setprecision(1) << pass1_ms << " ms" << std::endl;
        std::cout << "Pass 2 time: " << std::fixed << std::setprecision(1) << pass2_ms << " ms" << std::endl;
        std::cout << "Lookup index footprint: "
                  << std::fixed << std::setprecision(2)
                  << (lookup_bytes / (1024.0 * 1024.0)) << " MiB ("
                  << (total_orbit_elements > 0
                          ? (double(lookup_bytes) / total_orbit_elements)
                          : 0.0)
                  << " B/entry)" << std::endl;
        std::cout << "Memory saved vs full expansion: " 
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - double(total_orbit_elements) / (total_basis * fixed_sz_dim_)))
                  << "%" << std::endl;
    }

    // ------------------------------------------------------------------
    // Streaming sector materialization helpers ("stream sym sectors").
    // Active only when ``lazy_sectors_enabled_`` is true. In eager mode
    // these are cheap no-ops (sectors already materialized).
    // ------------------------------------------------------------------

    /// Build sectors_[k].basis_states orbit CSR from the resident
    /// ``unique_orbit_reps_`` for a single sector. Does NOT build the CPU
    /// reverse index (that is lazy in makeSectorLookup_).
    void materializeSectorOrbits_(std::size_t k) const {
        auto& sector = sectors_[k];
        if (!sector.basis_states.empty()) return;  // already resident
        const size_t num_orbits = unique_orbit_reps_.size();

        struct OrbitResult {
            uint64_t orbit_rep;
            std::vector<uint64_t> orbit_elements;
            std::vector<Complex> orbit_coefficients;
            double norm;
        };
        std::vector<OrbitResult> valid_orbits(num_orbits);
        std::vector<char> orbit_valid(num_orbits, 0);

        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t oi = 0; oi < num_orbits; ++oi) {
            std::vector<uint64_t> elements;
            std::vector<Complex> coefficients;
            double norm_sq = 0.0;
            computeOrbitDataFixedSz(unique_orbit_reps_[oi],
                                    sector.phase_factors,
                                    elements, coefficients, norm_sq);
            if (norm_sq > 1e-10) {
                valid_orbits[oi].orbit_rep = unique_orbit_reps_[oi];
                valid_orbits[oi].orbit_elements = std::move(elements);
                valid_orbits[oi].orbit_coefficients = std::move(coefficients);
                valid_orbits[oi].norm = std::sqrt(norm_sq);
                orbit_valid[oi] = 1;
            }
        }

        sector.basis_states.clear();
        if (k < symmetrized_block_ham_sizes.size()) {
            sector.basis_states.reserve(
                static_cast<size_t>(symmetrized_block_ham_sizes[k]));
        }
        for (size_t oi = 0; oi < num_orbits; ++oi) {
            if (!orbit_valid[oi]) continue;
            auto& orb = valid_orbits[oi];
            SymBasisState state(orb.orbit_rep, sector.quantum_numbers, orb.norm);
            state.orbit_elements = std::move(orb.orbit_elements);
            state.orbit_coefficients = std::move(orb.orbit_coefficients);
            state.sortOrbit();
            sector.basis_states.push_back(std::move(state));
        }
    }

    /// Free a sector's orbit CSR + reverse index + dense lookup so the
    /// resident footprint stays bounded to a single sector (LRU-1).
    void releaseSectorOrbits_(std::size_t k) const {
        if (k >= sectors_.size()) return;
        std::vector<SymBasisState>().swap(sectors_[k].basis_states);
        if (k < state_to_sector_basis_.size()) {
            state_to_sector_basis_[k] = ed::core::SortedUint64Index{};
        }
        if (k < sector_lookup_dense_.size()) {
            std::vector<std::int32_t>().swap(sector_lookup_dense_[k]);
        }
        if (k < reverse_index_built_.size()) reverse_index_built_[k] = 0;
        // Dense lookup (if it was built) is keyed across all sectors; force
        // a rebuild decision next time.
        dense_lookup_state_.store(0, std::memory_order_release);
    }

    /// LRU-1 chokepoint: ensure sector ``k`` orbit CSR is resident,
    /// releasing the previously materialized sector. Cheap fast path when
    /// ``k`` is already the resident sector (or in eager mode).
    void ensureSectorMaterialized_(std::size_t k) const {
        if (!lazy_sectors_enabled_) return;
        if (materialized_sector_.load(std::memory_order_acquire)
                == static_cast<std::ptrdiff_t>(k)) {
            return;
        }
        std::lock_guard<std::mutex> lock(materialize_mu_);
        if (materialized_sector_.load(std::memory_order_relaxed)
                == static_cast<std::ptrdiff_t>(k)) {
            return;
        }
        const std::ptrdiff_t prev =
            materialized_sector_.load(std::memory_order_relaxed);
        if (prev >= 0) releaseSectorOrbits_(static_cast<std::size_t>(prev));
        materializeSectorOrbits_(k);
        materialized_sector_.store(static_cast<std::ptrdiff_t>(k),
                                   std::memory_order_release);
        // Bounded-peak diagnostic: report the resident sector footprint so a
        // cluster log shows only one sector's orbit CSR alive at a time.
        std::size_t elems = 0;
        for (const auto& bs : sectors_[k].basis_states)
            elems += bs.orbit_elements.size();
        const double gib =
            static_cast<double>(elems) * (sizeof(uint64_t) + sizeof(Complex))
            / (1024.0 * 1024.0 * 1024.0);
        std::cout << "[lazy-sym] materialized sector " << k << " (dim="
                  << sectors_[k].basis_states.size() << ", " << elems
                  << " orbit elems, ~" << std::fixed << std::setprecision(2)
                  << gib << " GiB orbit CSR resident; previous sector "
                  << (prev >= 0 ? std::to_string(prev) : std::string("none"))
                  << " freed)" << std::endl;
    }

    /// Build the CPU reverse index (state -> sector-basis index) for a
    /// single sector on first use. No-op in eager mode (already built at
    /// generation) and on repeat calls for the same sector.
    void ensureSectorReverseIndex_(std::size_t k) const {
        if (!lazy_sectors_enabled_) return;
        if (k < reverse_index_built_.size() && reverse_index_built_[k]) return;
        std::lock_guard<std::mutex> lock(materialize_mu_);
        if (k < reverse_index_built_.size() && reverse_index_built_[k]) return;
        const auto& sector = sectors_[k];
        auto& idx = state_to_sector_basis_[k];
        idx = ed::core::SortedUint64Index{};
        for (size_t j = 0; j < sector.basis_states.size(); ++j) {
            for (uint64_t elem : sector.basis_states[j].orbit_elements) {
                idx[elem] = j;
            }
        }
        idx.finalize();
        if (k < reverse_index_built_.size()) reverse_index_built_[k] = 1;
    }
    
    /**
     * @brief Matrix-free Hamiltonian application in symmetrized sector
     * 
     * This is the key function that avoids expanding to fixed-Sz dimension.
     * 
     * Algorithm:
     * 1. For each input coefficient c_j with basis state |φ_j⟩
     * 2. For each orbit element |s⟩ in |φ_j⟩ with coefficient α_s
     * 3. Apply each Hamiltonian term to |s⟩ -> |s'⟩ with matrix element h
     * 4. Look up which basis state |φ_k⟩ contains |s'⟩
     * 5. Accumulate: out[k] += c_j * α_s * h * conj(β_{s'})
     *    where β_{s'} is the coefficient of |s'⟩ in |φ_k⟩
     * 
     * Memory: O(sector_dim) for output, no full-dim intermediates
     */
    void applySymmetrizedFixedSz(size_t sector_idx, const Complex* in, Complex* out) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        ensureSectorMaterialized_(sector_idx);
        
        const auto& sector = sectors_[sector_idx];
        const size_t sector_dim = sector.basis_states.size();
        const SectorLookupHandle lookup = makeSectorLookup_(sector_idx);
        
        // Initialize output
        std::fill(out, out + sector_dim, Complex(0.0, 0.0));
        
        // Pre-compute normalization factors
        const double group_size = static_cast<double>(getGroupSize());
        const double group_norm = 1.0 / group_size;
        
        // Process each input basis state
        #pragma omp parallel if(sector_dim > 100)
        {
            // Wave A4 (May 2026): thread-local scratch persists across
            // matvec calls -- ``assign`` reuses the existing allocation
            // when sector_dim stays the same (the common case in a
            // Lanczos / KPM / Chebyshev recurrence).
            static thread_local std::vector<Complex> local_out;
            local_out.assign(sector_dim, Complex(0.0, 0.0));

            #pragma omp for schedule(dynamic, 4)
            for (size_t j = 0; j < sector_dim; ++j) {
                Complex c_j = in[j];
                if (std::abs(c_j) < 1e-15) continue;
                
                const auto& state_j = sector.basis_states[j];
                const double norm_j = state_j.norm;
                
                // Iterate over orbit elements of |φ_j⟩
                for (size_t orbit_idx = 0; orbit_idx < state_j.orbit_elements.size(); ++orbit_idx) {
                    uint64_t s = state_j.orbit_elements[orbit_idx];
                    Complex alpha_s = state_j.orbit_coefficients[orbit_idx];
                    
                    // Skip if coefficient is negligible
                    if (std::abs(alpha_s) < 1e-15) continue;
                    
                    // Apply Hamiltonian terms to |s⟩
                    // Use optimized term-by-term application
                    applyHamiltonianTerms(s, c_j * alpha_s / norm_j, 
                                         sector, lookup, group_norm, local_out);
                }
            }
            
            // Merge thread-local results
            #pragma omp critical
            {
                for (size_t k = 0; k < sector_dim; ++k) {
                    out[k] += local_out[k];
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Wave 1 (May 2026, "Unify all 16 matvec cells" plan): unified
    // Sz+symmetry matvec via ``apply_terms<SymmetryBasisPolicy, Scalar>``.
    // See the twin block in ``StreamingSymmetryOperator`` for the
    // contract.
    // -----------------------------------------------------------------
    void applySymmetrizedFixedSzUnified(size_t sector_idx,
                                        const Complex* in,
                                        Complex* out) const;
    void applySymmetrizedFixedSzUnifiedReal(size_t sector_idx,
                                            const double* in,
                                            double* out) const;

private:
    // -----------------------------------------------------------------
    // Wave A2 (May 2026): per-sector dense O(1) reverse lookup table.
    //
    // Indexed by the parent ``FixedSzOperator``'s LinIndexTable
    // position (size = fixed_sz_dim). Each entry holds the sector-basis
    // index (or -1 for "not in this sector"). 4 bytes per fixed-Sz
    // state per sector.
    // -----------------------------------------------------------------
    mutable std::vector<std::vector<std::int32_t>> sector_lookup_dense_;
    mutable std::atomic<int> dense_lookup_state_{0};
    mutable std::mutex dense_lookup_build_mu_;

    void buildDenseLookupsIfAffordable_() const {
        if (dense_lookup_state_.load(std::memory_order_acquire) != 0) return;
        std::lock_guard<std::mutex> lock(dense_lookup_build_mu_);
        if (dense_lookup_state_.load(std::memory_order_acquire) != 0) return;

        // "stream sym sectors": the cross-sector dense lookup assumes every
        // sector's reverse index is resident simultaneously, which is exactly
        // what lazy mode avoids. Disable it -- the per-sector SortedUint64Index
        // fallback (built lazily in makeSectorLookup_) is used instead.
        if (lazy_sectors_enabled_) {
            dense_lookup_state_.store(2, std::memory_order_release);
            return;
        }

        if (sectors_.empty() || state_to_sector_basis_.empty()) {
            dense_lookup_state_.store(2, std::memory_order_release);
            return;
        }

        std::size_t budget_bytes = 512ULL * 1024ULL * 1024ULL;
        if (const char* env = std::getenv("ED_SYM_DENSE_LOOKUP_BYTES_MAX")) {
            try { budget_bytes = std::stoull(env); } catch (...) {}
        }

        const std::size_t fz_dim = static_cast<std::size_t>(fixed_sz_dim_);
        const std::size_t bytes_per_sector = fz_dim * sizeof(std::int32_t);
        const std::size_t total =
            bytes_per_sector * state_to_sector_basis_.size();
        if (total > budget_bytes) {
            dense_lookup_state_.store(2, std::memory_order_release);
            return;
        }

        sector_lookup_dense_.assign(state_to_sector_basis_.size(),
                                    std::vector<std::int32_t>{});
        for (std::size_t k = 0; k < state_to_sector_basis_.size(); ++k) {
            auto& dense = sector_lookup_dense_[k];
            dense.assign(fz_dim, -1);
            const auto& src = state_to_sector_basis_[k];
            const auto& keys = src.keys();
            const auto& vals = src.values();
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const std::int64_t li = this->lookupState(keys[i]);
                if (li < 0 || static_cast<std::size_t>(li) >= fz_dim) {
                    // State outside this operator's fixed-Sz sector --
                    // skip; the SortedUint64Index path retains
                    // correctness for any pathological cases.
                    continue;
                }
                dense[static_cast<std::size_t>(li)] =
                    static_cast<std::int32_t>(vals[i]);
            }
        }
        dense_lookup_state_.store(1, std::memory_order_release);
    }

    SectorLookupHandle makeSectorLookup_(std::size_t sector_idx) const {
        // "stream sym sectors": ensure the sector's orbit CSR is resident
        // (LRU-1) and build the CPU reverse index for it on first use. Both
        // are no-ops in eager mode.
        ensureSectorMaterialized_(sector_idx);
        ensureSectorReverseIndex_(sector_idx);
        buildDenseLookupsIfAffordable_();
        SectorLookupHandle h;
        h.fallback = &state_to_sector_basis_[sector_idx];
        if (dense_lookup_state_.load(std::memory_order_acquire) == 1
            && sector_idx < sector_lookup_dense_.size()
            && !sector_lookup_dense_[sector_idx].empty()) {
            h.dense = sector_lookup_dense_[sector_idx].data();
            h.lin   = &this->lin_index_table();
        }
        return h;
    }

public:
    
    std::vector<Complex> applySymmetrizedFixedSz(size_t sector_idx,
                                                 const std::vector<Complex>& vec) const {
        ensureSectorMaterialized_(sector_idx);
        const size_t sector_dim = sectors_[sector_idx].basis_states.size();
        std::vector<Complex> result(sector_dim);
        applySymmetrizedFixedSz(sector_idx, vec.data(), result.data());
        return result;
    }

    // -----------------------------------------------------------------
    // Wave C1 / C5 deferral notes (close symmetry compute gap, May 2026)
    // -----------------------------------------------------------------
    // **C1** -- ``SymmetryBasisPolicy`` for the unified
    // ``apply_terms<BasisPolicy>`` kernel. The plan envisions a
    // value-type policy whose ``state_of(i)`` walks the orbit and whose
    // ``index_of(s)`` does the LinIndex lookup (already wired in Wave
    // A2). The blocker is character-phase + orbit-norm propagation:
    // the existing ``apply_terms`` interface assumes ``index_of`` is
    // a pure (state -> index) map, so the per-term scatter applies
    // ``coeff * input[i]`` only. The symmetrized matvec additionally
    // multiplies by ``char_factor(orbit_element) *
    // sqrt(N_orbit_new / N_orbit_old)``, which requires extending the
    // BasisPolicy ABI with a ``coeff_modifier(new_state, src_idx)``
    // hook and threading it through the radix-scatter at
    // term_kernels.h:282-462. Mitigation per plan: gate behind
    // ``ED_SYMMETRY_LEGACY_MATVEC`` for one release cycle once the
    // unified path is exercised. The current Wave-A optimisations
    // (real-arithmetic, O(1) LinIndex lookup, OMP, thread_local
    // scratch) already cover most of the per-matvec gap; C1 buys the
    // remaining ~2-3x by sharing the SoA + radix scatter
    // infrastructure with ``Operator`` / ``FixedSzOperator``.
    //
    // **C5** -- assembled CSR cache for small symmetry sectors. The
    // ``CpuMatVecBackend`` already owns CSR caches for the full and
    // fixed-Sz operators (see matvec_backend.h:150-216 + the
    // ``ED_CSR_DIM_MAX`` / ``ED_CSR_FORCE`` env vars); the missing
    // piece is exposing the symmetrized matvec as a Backend the
    // SectorView dispatches through. The pay-off is narrow: at small
    // sector dim (~hundreds) the build cost is < O(dim * matvec_cost)
    // and amortises over a Lanczos sweep (~hundreds of SpMVs). Beyond
    // sector_dim ~ thousands the matrix-free path with Wave-A wins
    // already dominates. C5 is marked optional in the plan; the
    // current sectors at production N=12-14 are above the break-even
    // threshold so we defer until the small-N regime matters.
    // -----------------------------------------------------------------

    /// Number of symmetry sectors after generation.
    [[nodiscard]] std::size_t num_sectors() const noexcept {
        return sectors_.size();
    }

    const SymmetrySector& getSector(size_t sector_idx) const {
        ensureSectorMaterialized_(sector_idx);
        return sectors_[sector_idx];
    }
    
    size_t getNumSectors() const { return sectors_.size(); }
    
    uint64_t getSectorDimension(size_t sector_idx) const {
        // "stream sym sectors": dimension is known from Pass 1.5 without
        // materializing the orbit CSR (eager mode: same value).
        if (sector_idx < symmetrized_block_ham_sizes.size()) {
            return static_cast<uint64_t>(
                symmetrized_block_ham_sizes[sector_idx]);
        }
        return sectors_[sector_idx].basis_states.size();
    }

    /// Public computational-state -> orbit-basis-index lookup for
    /// sector ``sector_idx``. Returns
    /// ``ed::core::SortedUint64Index::kNotFound`` when ``s`` is not
    /// in any orbit of that sector. See the docstring on
    /// ``StreamingSymmetryOperator::lookupBasisIndex`` for the
    /// motivation -- this twin lets ``CrossSectorOrbitObservable``
    /// work on the fixed-Sz streaming-symmetry variant too.
    std::size_t lookupBasisIndex(std::size_t sector_idx,
                                 std::uint64_t s) const {
        if (sector_idx >= state_to_sector_basis_.size()) {
            return ed::core::SortedUint64Index::kNotFound;
        }
        // "stream sym sectors": ensure the orbit CSR + reverse index for
        // this sector are resident (no-ops in eager mode).
        ensureSectorMaterialized_(sector_idx);
        ensureSectorReverseIndex_(sector_idx);
        return state_to_sector_basis_[sector_idx].find(s);
    }

    // ========================================================================
    // Orbit Basis Caching (HDF5)
    // ========================================================================

    /**
     * @brief Get the default cache file path for the current system
     *
     * The cache file encodes n_sites and n_up so it's unique per geometry+Sz.
     */
    static std::string getOrbitCachePath(const std::string& cache_dir,
                                         uint64_t n_sites, int64_t n_up) {
        return cache_dir + "/orbit_basis_N" + std::to_string(n_sites)
               + "_nup" + std::to_string(n_up) + ".h5";
    }

    // orbitCacheExists (fixed-Sz variant) was retired in the
    // minimalist-architecture rev (May 2026): no callers; existence
    // is implicit in saveOrbitBasisHDF5 / loadOrbitBasisHDF5.

    /**
     * @brief Save full orbit basis to HDF5 for later reuse
     *
     * Stores per-sector CSR orbit data (elements, coefficients, offsets, norms)
     * plus sector metadata (quantum numbers, phase factors, sector id).
     * This data depends ONLY on the lattice geometry and Sz sector, NOT on
     * the Hamiltonian couplings, so it can be reused across parameter sweeps.
     *
     * HDF5 layout:
     *   /orbit_basis/
     *     attrs: n_sites, n_up, num_sectors, group_size
     *     /sector_<i>/
     *       attrs: sector_id, num_basis
     *       datasets: quantum_numbers, phase_factors_real, phase_factors_imag,
     *                 orbit_offsets, orbit_norms, orbit_elements,
     *                 orbit_coefficients_real, orbit_coefficients_imag
     */
    void saveOrbitBasisHDF5(const std::string& cache_dir) const {
        // P0.12: was safe_system_call("mkdir -p ...").
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        std::string filepath = getOrbitCachePath(cache_dir, n_bits_, n_up_);

        std::cout << "\n=== Saving orbit basis cache to " << filepath << " ===" << std::endl;

        try {
            H5::H5File file(filepath, H5F_ACC_TRUNC);

            // Create root group
            file.createGroup("/orbit_basis");

            // --- Metadata attributes ---
            {
                H5::DataSpace scalar(H5S_SCALAR);
                auto grp = file.openGroup("/orbit_basis");

                uint64_t ns = n_bits_;
                auto a1 = grp.createAttribute("n_sites", H5::PredType::NATIVE_UINT64, scalar);
                a1.write(H5::PredType::NATIVE_UINT64, &ns);

                int64_t nup = n_up_;
                auto a2 = grp.createAttribute("n_up", H5::PredType::NATIVE_INT64, scalar);
                a2.write(H5::PredType::NATIVE_INT64, &nup);

                uint64_t nsec = sectors_.size();
                auto a3 = grp.createAttribute("num_sectors", H5::PredType::NATIVE_UINT64, scalar);
                a3.write(H5::PredType::NATIVE_UINT64, &nsec);

                uint64_t gsz = symmetry_info.max_clique.size();
                auto a4 = grp.createAttribute("group_size", H5::PredType::NATIVE_UINT64, scalar);
                a4.write(H5::PredType::NATIVE_UINT64, &gsz);
            }

            // --- Per-sector data ---
            for (size_t si = 0; si < sectors_.size(); ++si) {
                // "stream sym sectors": materialize on demand (LRU-1).
                ensureSectorMaterialized_(si);
                const auto& sector = sectors_[si];
                std::string grp_name = "/orbit_basis/sector_" + std::to_string(si);
                file.createGroup(grp_name);
                auto grp = file.openGroup(grp_name);

                // Sector attributes
                {
                    H5::DataSpace scalar(H5S_SCALAR);
                    uint64_t sid = sector.sector_id;
                    auto a = grp.createAttribute("sector_id", H5::PredType::NATIVE_UINT64, scalar);
                    a.write(H5::PredType::NATIVE_UINT64, &sid);

                    uint64_t nb = sector.basis_states.size();
                    auto a2 = grp.createAttribute("num_basis", H5::PredType::NATIVE_UINT64, scalar);
                    a2.write(H5::PredType::NATIVE_UINT64, &nb);
                }

                // Quantum numbers
                if (!sector.quantum_numbers.empty()) {
                    hsize_t dims[1] = {sector.quantum_numbers.size()};
                    H5::DataSpace ds(1, dims);
                    auto dset = grp.createDataSet("quantum_numbers",
                                                   H5::PredType::NATIVE_INT, ds);
                    dset.write(sector.quantum_numbers.data(), H5::PredType::NATIVE_INT);
                }

                // Phase factors (complex → separate real/imag)
                if (!sector.phase_factors.empty()) {
                    std::vector<double> pf_real(sector.phase_factors.size());
                    std::vector<double> pf_imag(sector.phase_factors.size());
                    for (size_t k = 0; k < sector.phase_factors.size(); ++k) {
                        pf_real[k] = sector.phase_factors[k].real();
                        pf_imag[k] = sector.phase_factors[k].imag();
                    }
                    hsize_t dims[1] = {sector.phase_factors.size()};
                    H5::DataSpace ds(1, dims);
                    auto d1 = grp.createDataSet("phase_factors_real",
                                                 H5::PredType::NATIVE_DOUBLE, ds);
                    d1.write(pf_real.data(), H5::PredType::NATIVE_DOUBLE);
                    auto d2 = grp.createDataSet("phase_factors_imag",
                                                 H5::PredType::NATIVE_DOUBLE, ds);
                    d2.write(pf_imag.data(), H5::PredType::NATIVE_DOUBLE);
                }

                // --- CSR orbit data (flattened exactly as extractOrbitData does) ---
                size_t num_basis = sector.basis_states.size();
                std::vector<int64_t> offsets(num_basis + 1);
                std::vector<double> norms(num_basis);
                offsets[0] = 0;
                for (size_t j = 0; j < num_basis; ++j) {
                    offsets[j + 1] = offsets[j] +
                        static_cast<int64_t>(sector.basis_states[j].orbit_elements.size());
                    norms[j] = sector.basis_states[j].norm;
                }
                size_t total_elems = offsets[num_basis];

                std::vector<uint64_t> flat_elements(total_elems);
                std::vector<double> flat_coeff_real(total_elems);
                std::vector<double> flat_coeff_imag(total_elems);
                for (size_t j = 0; j < num_basis; ++j) {
                    const auto& bs = sector.basis_states[j];
                    int64_t off = offsets[j];
                    for (size_t e = 0; e < bs.orbit_elements.size(); ++e) {
                        flat_elements[off + e] = bs.orbit_elements[e];
                        flat_coeff_real[off + e] = bs.orbit_coefficients[e].real();
                        flat_coeff_imag[off + e] = bs.orbit_coefficients[e].imag();
                    }
                }

                // Write offsets
                {
                    hsize_t dims[1] = {static_cast<hsize_t>(num_basis + 1)};
                    H5::DataSpace ds(1, dims);
                    auto d = grp.createDataSet("orbit_offsets",
                                                H5::PredType::NATIVE_INT64, ds);
                    d.write(offsets.data(), H5::PredType::NATIVE_INT64);
                }
                // Write norms
                {
                    hsize_t dims[1] = {static_cast<hsize_t>(num_basis)};
                    H5::DataSpace ds(1, dims);
                    auto d = grp.createDataSet("orbit_norms",
                                                H5::PredType::NATIVE_DOUBLE, ds);
                    d.write(norms.data(), H5::PredType::NATIVE_DOUBLE);
                }
                // Write orbit elements
                if (total_elems > 0) {
                    hsize_t dims[1] = {static_cast<hsize_t>(total_elems)};
                    H5::DataSpace ds(1, dims);
                    auto d = grp.createDataSet("orbit_elements",
                                                H5::PredType::NATIVE_UINT64, ds);
                    d.write(flat_elements.data(), H5::PredType::NATIVE_UINT64);
                }
                // Write orbit coefficients (real + imag)
                if (total_elems > 0) {
                    hsize_t dims[1] = {static_cast<hsize_t>(total_elems)};
                    H5::DataSpace ds(1, dims);
                    auto d1 = grp.createDataSet("orbit_coefficients_real",
                                                 H5::PredType::NATIVE_DOUBLE, ds);
                    d1.write(flat_coeff_real.data(), H5::PredType::NATIVE_DOUBLE);
                    auto d2 = grp.createDataSet("orbit_coefficients_imag",
                                                 H5::PredType::NATIVE_DOUBLE, ds);
                    d2.write(flat_coeff_imag.data(), H5::PredType::NATIVE_DOUBLE);
                }

                grp.close();
            }

            file.close();

            // Print summary
            size_t total_basis = 0, total_orbit = 0;
            for (const auto& s : sectors_) {
                total_basis += s.basis_states.size();
                for (const auto& bs : s.basis_states)
                    total_orbit += bs.orbit_elements.size();
            }
            std::cout << "Cached " << sectors_.size() << " sectors, "
                      << total_basis << " basis states, "
                      << total_orbit << " orbit elements" << std::endl;
            std::cout << "=== Orbit basis cache saved ===" << std::endl;

        } catch (H5::Exception& e) {
            std::cerr << "\nError: Failed to save orbit basis cache: "
                      << e.getCDetailMsg() << std::endl;
            // Non-fatal: sectors are already in memory, diagonalization can proceed
        }
    }

    /**
     * @brief Load orbit basis from HDF5 cache
     *
     * Restores sectors_, state_to_sector_basis_, and symmetrized_block_ham_sizes
     * from a previously saved cache file.  The symmetry_info must already be
     * loaded (via loadFromDirectory) so that phase_factors/quantum_numbers
     * can be cross-checked, but the expensive orbit enumeration is skipped.
     *
     * @return true if loaded successfully, false if cache not found / mismatch
     */
    bool loadOrbitBasisHDF5(const std::string& cache_dir) {
        std::string filepath = getOrbitCachePath(cache_dir, n_bits_, n_up_);

        {
            std::ifstream f(filepath);
            if (!f.good()) return false;
        }

        std::cout << "\n=== Loading orbit basis cache from " << filepath << " ===" << std::endl;

        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            auto root = file.openGroup("/orbit_basis");

            // Verify metadata
            uint64_t cached_n_sites, cached_num_sectors, cached_group_size;
            int64_t cached_n_up;
            root.openAttribute("n_sites").read(H5::PredType::NATIVE_UINT64, &cached_n_sites);
            root.openAttribute("n_up").read(H5::PredType::NATIVE_INT64, &cached_n_up);
            root.openAttribute("num_sectors").read(H5::PredType::NATIVE_UINT64, &cached_num_sectors);
            root.openAttribute("group_size").read(H5::PredType::NATIVE_UINT64, &cached_group_size);

            if (cached_n_sites != n_bits_) {
                std::cerr << "Cache mismatch: n_sites " << cached_n_sites
                          << " vs " << n_bits_ << std::endl;
                return false;
            }
            if (cached_n_up != n_up_) {
                std::cerr << "Cache mismatch: n_up " << cached_n_up
                          << " vs " << n_up_ << std::endl;
                return false;
            }

            // Store group size from cache so that matvec / expansion code
            // can function without loading symmetry_info from disk.
            cached_group_size_ = cached_group_size;
            std::cout << "  group_size (from cache): " << cached_group_size_ << std::endl;

            // Allocate sector storage
            sectors_.resize(cached_num_sectors);
            symmetrized_block_ham_sizes.assign(cached_num_sectors, 0);
            state_to_sector_basis_.resize(cached_num_sectors);

            size_t total_orbit_elements = 0;

            for (size_t si = 0; si < cached_num_sectors; ++si) {
                std::string grp_name = "/orbit_basis/sector_" + std::to_string(si);
                auto grp = file.openGroup(grp_name);
                auto& sector = sectors_[si];

                // Sector attributes
                uint64_t sid, nb;
                grp.openAttribute("sector_id").read(H5::PredType::NATIVE_UINT64, &sid);
                grp.openAttribute("num_basis").read(H5::PredType::NATIVE_UINT64, &nb);
                sector.sector_id = sid;

                // Quantum numbers
                {
                    auto dset = grp.openDataSet("quantum_numbers");
                    auto space = dset.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    sector.quantum_numbers.resize(dims[0]);
                    dset.read(sector.quantum_numbers.data(), H5::PredType::NATIVE_INT);
                }

                // Phase factors
                {
                    auto d1 = grp.openDataSet("phase_factors_real");
                    auto d2 = grp.openDataSet("phase_factors_imag");
                    auto space = d1.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    std::vector<double> pf_real(dims[0]), pf_imag(dims[0]);
                    d1.read(pf_real.data(), H5::PredType::NATIVE_DOUBLE);
                    d2.read(pf_imag.data(), H5::PredType::NATIVE_DOUBLE);
                    sector.phase_factors.resize(dims[0]);
                    for (size_t k = 0; k < dims[0]; ++k) {
                        sector.phase_factors[k] = Complex(pf_real[k], pf_imag[k]);
                    }
                }

                // CSR orbit data
                std::vector<int64_t> offsets;
                std::vector<double> norms;
                {
                    auto dset = grp.openDataSet("orbit_offsets");
                    auto space = dset.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    offsets.resize(dims[0]);
                    dset.read(offsets.data(), H5::PredType::NATIVE_INT64);
                }
                {
                    auto dset = grp.openDataSet("orbit_norms");
                    auto space = dset.getSpace();
                    hsize_t dims[1];
                    space.getSimpleExtentDims(dims);
                    norms.resize(dims[0]);
                    dset.read(norms.data(), H5::PredType::NATIVE_DOUBLE);
                }

                size_t total_elems = (offsets.size() > 1) ? offsets.back() : 0;
                std::vector<uint64_t> flat_elements;
                std::vector<double> flat_coeff_real, flat_coeff_imag;
                if (total_elems > 0) {
                    flat_elements.resize(total_elems);
                    flat_coeff_real.resize(total_elems);
                    flat_coeff_imag.resize(total_elems);
                    grp.openDataSet("orbit_elements")
                        .read(flat_elements.data(), H5::PredType::NATIVE_UINT64);
                    grp.openDataSet("orbit_coefficients_real")
                        .read(flat_coeff_real.data(), H5::PredType::NATIVE_DOUBLE);
                    grp.openDataSet("orbit_coefficients_imag")
                        .read(flat_coeff_imag.data(), H5::PredType::NATIVE_DOUBLE);
                }

                // Reconstruct SymBasisState objects and lookup table
                sector.basis_states.resize(nb);
                for (size_t j = 0; j < nb; ++j) {
                    auto& bs = sector.basis_states[j];
                    bs.quantum_numbers = sector.quantum_numbers;
                    bs.norm = norms[j];
                    int off = offsets[j];
                    int len = offsets[j + 1] - off;
                    bs.orbit_elements.resize(len);
                    bs.orbit_coefficients.resize(len);
                    for (int e = 0; e < len; ++e) {
                        bs.orbit_elements[e] = flat_elements[off + e];
                        bs.orbit_coefficients[e] = Complex(flat_coeff_real[off + e],
                                                            flat_coeff_imag[off + e]);
                    }
                    // Sort once -> O(log |orbit|) lookups in applySymmetrized.
                    // After sorting, [0] is the canonical (smallest) representative.
                    bs.sortOrbit();
                    bs.orbit_rep = bs.orbit_elements.empty() ? 0 : bs.orbit_elements[0];

                    // Rebuild lookup table
                    for (uint64_t elem : bs.orbit_elements) {
                        state_to_sector_basis_[si][elem] = j;
                    }
                    total_orbit_elements += len;
                }

                // Phase 3a #5: finalize the lookup index for this sector
                // before the next outer iteration moves on.
                state_to_sector_basis_[si].finalize();

                symmetrized_block_ham_sizes[si] = nb;
                grp.close();
            }

            file.close();

            size_t total_basis = 0;
            for (const auto& s : sectors_)
                total_basis += s.basis_states.size();

            std::cout << "Loaded " << sectors_.size() << " sectors, "
                      << total_basis << " basis states, "
                      << total_orbit_elements << " orbit elements"
                      << std::endl;

            // Validate: every non-empty sector must have orbit elements
            for (size_t si = 0; si < sectors_.size(); ++si) {
                const auto& sector = sectors_[si];
                if (!sector.basis_states.empty()) {
                    bool has_empty = false;
                    for (const auto& bs : sector.basis_states) {
                        if (bs.orbit_elements.empty()) { has_empty = true; break; }
                    }
                    if (has_empty) {
                        std::cerr << "Warning: Sector " << si
                                  << " has basis states with empty orbit data "
                                  << "— cache may be corrupted, regenerating"
                                  << std::endl;
                        sectors_.clear();
                        state_to_sector_basis_.clear();
                        symmetrized_block_ham_sizes.clear();
                        return false;
                    }
                }
            }

            std::cout << "=== Orbit basis cache loaded ===" << std::endl;
            return true;

        } catch (H5::Exception& e) {
            std::cerr << "Warning: Failed to load orbit basis cache: "
                      << e.getCDetailMsg() << std::endl;
            return false;
        }
    }

    // ===================== Eigenvector expansion ============================

    /**
     * @brief Expand a symmetrized-sector eigenvector to the fixed-Sz computational basis.
     *
     * Returns a vector of length C(N, n_up) indexed by position in basis_states_.
     * For embedding into the full 2^N Hilbert space, call embedToFull() afterwards.
     *
     * @param sector_idx  Index of the symmetry sector
     * @param sym_vec     Eigenvector in the symmetrized sector basis
     * @return Vector of length C(N, n_up) in fixed-Sz basis
     */
    std::vector<Complex> expandToFixedSzBasis(
        size_t sector_idx,
        const std::vector<Complex>& sym_vec
    ) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        ensureSectorMaterialized_(sector_idx);
        const auto& sector = sectors_[sector_idx];
        size_t sector_dim = sector.basis_states.size();
        if (sym_vec.size() != sector_dim) {
            throw std::runtime_error(
                "Eigenvector size (" + std::to_string(sym_vec.size())
                + ") != sector dimension (" + std::to_string(sector_dim) + ")");
        }

        // Build reverse lookup: computational state → index in basis_states_
        std::unordered_map<uint64_t, size_t> state_to_idx;
        state_to_idx.reserve(basis_states_.size());
        for (size_t i = 0; i < basis_states_.size(); ++i) {
            state_to_idx[basis_states_[i]] = i;
        }

        const double group_norm = 1.0 / std::sqrt(
            static_cast<double>(getGroupSize()));

        std::vector<Complex> fixed_sz_vec(basis_states_.size(), Complex(0.0, 0.0));

        for (size_t j = 0; j < sector_dim; ++j) {
            if (std::abs(sym_vec[j]) < 1e-15) continue;
            const auto& bs = sector.basis_states[j];
            Complex weight = sym_vec[j] * group_norm / bs.norm;
            for (size_t k = 0; k < bs.orbit_elements.size(); ++k) {
                uint64_t s = bs.orbit_elements[k];
                auto it = state_to_idx.find(s);
                if (it != state_to_idx.end()) {
                    fixed_sz_vec[it->second] += weight * bs.orbit_coefficients[k];
                }
            }
        }
        return fixed_sz_vec;
    }

    /**
     * @brief Expand a symmetrized-sector eigenvector to the full 2^N computational basis.
     *
     * Combines expandToFixedSzBasis() + embedToFull() in one call.
     *
     * @param sector_idx  Index of the symmetry sector
     * @param sym_vec     Eigenvector in the symmetrized sector basis
     * @return Vector of length 2^N in computational basis
     */
    std::vector<Complex> expandToComputationalBasis(
        size_t sector_idx,
        const std::vector<Complex>& sym_vec
    ) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        ensureSectorMaterialized_(sector_idx);
        const auto& sector = sectors_[sector_idx];
        size_t sector_dim = sector.basis_states.size();
        if (sym_vec.size() != sector_dim) {
            throw std::runtime_error(
                "Eigenvector size (" + std::to_string(sym_vec.size())
                + ") != sector dimension (" + std::to_string(sector_dim) + ")");
        }

        uint64_t full_dim = 1ULL << n_bits_;
        std::vector<Complex> full_vec(full_dim, Complex(0.0, 0.0));

        const double group_norm = 1.0 / std::sqrt(
            static_cast<double>(getGroupSize()));

        for (size_t j = 0; j < sector_dim; ++j) {
            if (std::abs(sym_vec[j]) < 1e-15) continue;
            const auto& bs = sector.basis_states[j];
            Complex weight = sym_vec[j] * group_norm / bs.norm;
            for (size_t k = 0; k < bs.orbit_elements.size(); ++k) {
                uint64_t s = bs.orbit_elements[k];
                full_vec[s] += weight * bs.orbit_coefficients[k];
            }
        }
        return full_vec;
    }

    // ===================== End eigenvector expansion =========================
    
private:
    /**
     * @brief Apply all Hamiltonian terms to a single computational basis state
     * 
     * This is the inner loop of the matrix-free multiplication.
     * Projects results onto the symmetrized basis using the lookup table.
     */
    void applyHamiltonianTerms(uint64_t s, Complex weighted_coeff,
                               const SymmetrySector& sector,
                               const SectorLookupHandle& lookup,
                               double group_norm,
                               std::vector<Complex>& local_out) const {
        
        // Apply each one/two-body term from transform_data_
        for (const auto& tdata : transform_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;
            
            if (!tdata.is_two_body) {
                // One-body: S^α_i
                if (tdata.op_type == 2) {
                    // Sz: diagonal
                    double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    // S+ or S-: flip bit
                    uint64_t bit = (s >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
            } else {
                // Two-body: S^α_i S^β_j
                uint64_t bit_i = (s >> tdata.site_index) & 1;
                uint64_t bit_j = (s >> tdata.site_index_2) & 1;
                
                if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                    // Sz_i Sz_j: diagonal
                    double sign_i = bit_i ? -1.0 : 1.0;
                    double sign_j = bit_j ? -1.0 : 1.0;
                    h_element *= spin_l_ * spin_l_ * sign_i * sign_j;
                } else {
                    // Mixed terms
                    if (tdata.op_type != 2) {
                        if (bit_i != tdata.op_type) {
                            s_prime ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    } else {
                        double sign_i = bit_i ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_i;
                    }
                    
                    if (valid && tdata.op_type_2 != 2) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        if (new_bit_j != tdata.op_type_2) {
                            s_prime ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    } else if (valid) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        double sign_j = new_bit_j ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_j;
                    }
                }
            }
            
            if (!valid) continue;

            // Phase 3a #5: SortedUint64Index lookup (returns kNotFound on miss).
            const std::size_t k = lookup.find(s_prime);
            if (k == ed::core::SortedUint64Index::kNotFound) continue;

            const auto& state_k = sector.basis_states[k];
            
            // O(log |orbit|) binary search instead of linear scan.
            const Complex beta_s_prime = state_k.findCoeff(s_prime);
            
            // Accumulate: out[k] += weighted_coeff * h * conj(β_{s'}) / norm_k
            local_out[k] += weighted_coeff * h_element * std::conj(beta_s_prime) * group_norm / state_k.norm;
        }

        // Apply three-body terms from three_body_data_
        // (Mirrors the full-space applyHamiltonianTermsFullSpace logic)
        for (const auto& tdata : three_body_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;

            // Apply first operator
            if (tdata.op_type_1 == 2) {
                uint64_t bit = (s_prime >> tdata.site_index_1) & 1;
                double sign = bit ? -1.0 : 1.0;
                h_element *= spin_l_ * sign;
            } else {
                uint64_t bit = (s_prime >> tdata.site_index_1) & 1;
                if (bit != tdata.op_type_1) {
                    s_prime ^= (1ULL << tdata.site_index_1);
                } else {
                    valid = false;
                }
            }

            // Apply second operator
            if (valid) {
                if (tdata.op_type_2 == 2) {
                    uint64_t bit = (s_prime >> tdata.site_index_2) & 1;
                    double sign = bit ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> tdata.site_index_2) & 1;
                    if (bit != tdata.op_type_2) {
                        s_prime ^= (1ULL << tdata.site_index_2);
                    } else {
                        valid = false;
                    }
                }
            }

            // Apply third operator
            if (valid) {
                if (tdata.op_type_3 == 2) {
                    uint64_t bit = (s_prime >> tdata.site_index_3) & 1;
                    double sign = bit ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> tdata.site_index_3) & 1;
                    if (bit != tdata.op_type_3) {
                        s_prime ^= (1ULL << tdata.site_index_3);
                    } else {
                        valid = false;
                    }
                }
            }

            if (!valid) continue;

            // Phase 3a #5: SortedUint64Index lookup (returns kNotFound on miss).
            const std::size_t k = lookup.find(s_prime);
            if (k == ed::core::SortedUint64Index::kNotFound) continue;

            const auto& state_k = sector.basis_states[k];

            // O(log |orbit|) binary search instead of linear scan.
            const Complex beta_s_prime = state_k.findCoeff(s_prime);

            local_out[k] += weighted_coeff * h_element * std::conj(beta_s_prime) * group_norm / state_k.norm;
        }
    }
    
    // getOrbitRepresentativeFixedSzFast was retired in the
    // minimalist-architecture rev (May 2026): no callers; sector
    // generation populates state_to_orbit_cache_ directly via the
    // standard FixedSz orbit-rep path.

    // -----------------------------------------------------------------
    // Wave A1 (May 2026): real-arithmetic twin of
    // ``applyHamiltonianTerms``. Same bit-flip / Sz / phase logic in
    // double precision. Caller (``applySymmetrizedFixedSzReal``) must
    // gate on ``isSectorReal(sector_idx) == true``.
    // -----------------------------------------------------------------
    void applyHamiltonianTermsReal(
        uint64_t s, double weighted_coeff,
        const SymmetrySector& sector,
        const SectorLookupHandle& lookup,
        double group_norm,
        std::vector<double>& local_out) const {
        for (const auto& tdata : transform_data_) {
            uint64_t s_prime = s;
            double h_element = tdata.coefficient.real();
            bool valid = true;

            if (!tdata.is_two_body) {
                if (tdata.op_type == 2) {
                    double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
            } else {
                uint64_t bit_i = (s >> tdata.site_index) & 1;
                uint64_t bit_j = (s >> tdata.site_index_2) & 1;
                if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                    double sign_i = bit_i ? -1.0 : 1.0;
                    double sign_j = bit_j ? -1.0 : 1.0;
                    h_element *= spin_l_ * spin_l_ * sign_i * sign_j;
                } else {
                    if (tdata.op_type != 2) {
                        if (bit_i != tdata.op_type) {
                            s_prime ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    } else {
                        double sign_i = bit_i ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_i;
                    }
                    if (valid && tdata.op_type_2 != 2) {
                        uint64_t nb = (s_prime >> tdata.site_index_2) & 1;
                        if (nb != tdata.op_type_2) {
                            s_prime ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    } else if (valid) {
                        uint64_t nb = (s_prime >> tdata.site_index_2) & 1;
                        double sign_j = nb ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_j;
                    }
                }
            }
            if (!valid) continue;
            const std::size_t k = lookup.find(s_prime);
            if (k == ed::core::SortedUint64Index::kNotFound) continue;
            const auto& state_k = sector.basis_states[k];
            const double beta_s_prime = state_k.findCoeff(s_prime).real();
            local_out[k] += weighted_coeff * h_element * beta_s_prime
                          * group_norm / state_k.norm;
        }

        for (const auto& tdata : three_body_data_) {
            uint64_t s_prime = s;
            double h_element = tdata.coefficient.real();
            bool valid = true;

            auto apply_one = [&](int op_type, int site_idx) {
                if (!valid) return;
                if (op_type == 2) {
                    uint64_t bit = (s_prime >> site_idx) & 1;
                    double sign = bit ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> site_idx) & 1;
                    if (static_cast<int>(bit) != op_type) {
                        s_prime ^= (1ULL << site_idx);
                    } else {
                        valid = false;
                    }
                }
            };
            apply_one(tdata.op_type_1, tdata.site_index_1);
            apply_one(tdata.op_type_2, tdata.site_index_2);
            apply_one(tdata.op_type_3, tdata.site_index_3);
            if (!valid) continue;
            const std::size_t k = lookup.find(s_prime);
            if (k == ed::core::SortedUint64Index::kNotFound) continue;
            const auto& state_k = sector.basis_states[k];
            const double beta_s_prime = state_k.findCoeff(s_prime).real();
            local_out[k] += weighted_coeff * h_element * beta_s_prime
                          * group_norm / state_k.norm;
        }
    }

    /**
     * @brief Compute orbit elements and coefficients for a basis state in a sector
     *
     * Returns the orbit elements (computational basis states) and their
     * corresponding complex coefficients in the symmetrized state.
     *
     * Implementation (May 2026, "Orthogonal symmetry composition"):
     * delegates to the templated
     * ``ed::symmetry::compute_orbit_for_state`` helper with this
     * operator's ``subspace()`` view, which carries the fixed-Sz
     * ``index_of`` (returns -1 for off-sector states, so the helper's
     * generic ``subspace.index_of(permuted) < 0`` branch reproduces
     * the legacy ``lookupState(permuted) >= 0`` filter exactly).
     */
    void computeOrbitDataFixedSz(uint64_t basis,
                                 const std::vector<Complex>& phase_factors,
                                 std::vector<uint64_t>& orbit_elements,
                                 std::vector<Complex>& orbit_coefficients,
                                 double& norm_sq) const {
        const auto sub = this->subspace();
        const ed::symmetry::SpatialProjector spatial(symmetry_info);
        ed::symmetry::compute_orbit_for_state(
            sub, spatial, basis, phase_factors,
            orbit_elements, orbit_coefficients, norm_sq);
    }
    
    // computeSymmetrizedNormFixedSz was retired in the
    // minimalist-architecture rev (May 2026): no callers. Norms come
    // directly from computeOrbitDataFixedSz when sectors are generated.
};
