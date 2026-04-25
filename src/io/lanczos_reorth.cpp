// =============================================================================
// src/io/lanczos_reorth.cpp
//
// Out-of-core blocked-tile reorthogonalization helpers (Phase 3a #2).
// See include/ed/io/lanczos_reorth.h for the design rationale.
// =============================================================================

#include "ed/io/lanczos_reorth.h"

#include "ed/io/lanczos_basis_buffer.h"
#include "ed/core/blas_lapack_wrapper.h"
#include "ed/parallel/numa.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace lanczos_io {

namespace {

uint64_t parse_tile_size_env() {
    const char* v = std::getenv("ED_LANCZOS_REORTH_TILE");
    if (!v || !*v) return 16;
    try {
        long long parsed = std::stoll(v);
        if (parsed < 1)   return 1;
        if (parsed > 256) return 256;
        return static_cast<uint64_t>(parsed);
    } catch (...) {
        return 16;
    }
}

}  // anonymous namespace

uint64_t reorth_tile_size() {
    // Re-read each call so test code can flip the knob between solver
    // invocations. Cost: a few hundred ns per Lanczos iteration.
    return parse_tile_size_env();
}

uint64_t load_basis_tile(const std::string& temp_dir,
                         uint64_t k_start,
                         uint64_t count,
                         uint64_t N,
                         std::vector<Complex>& tile) {
    if (count == 0 || N == 0) {
        tile.clear();
        return 0;
    }
    tile.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(count),
                Complex(0.0, 0.0));
    // NUMA first-touch (Phase 3a #4): if ED_NUMA_FIRST_TOUCH=1 and the
    // tile is large enough to matter, parallel-zero it again so each
    // OpenMP thread owns the chunk of pages it will later read in
    // blocked_reorth's zgemv. `assign` above served-touched the buffer
    // on the calling thread; we need the parallel touch to spread it
    // across NUMA nodes.
    ed::parallel::first_touch_complex(
        tile.data(),
        static_cast<std::size_t>(N) * static_cast<std::size_t>(count));

    for (uint64_t k = 0; k < count; ++k) {
        Complex* col = tile.data() + static_cast<std::size_t>(k) * N;
        const uint64_t global_idx = k_start + k;

        // Fast path: in-memory buffer (zero-copy pointer + one memcpy).
        if (const Complex* p = get_basis_vector_ptr(temp_dir, global_idx)) {
            std::memcpy(col, p, static_cast<std::size_t>(N) * sizeof(Complex));
            continue;
        }

        // Fallback: legacy on-disk store. One ifstream per column, but
        // accessing consecutive indices keeps the OS page cache hot.
        const std::string filename =
            temp_dir + "/basis_" + std::to_string(global_idx) + ".dat";
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            // Partial load: report the prefix that succeeded.
            return k;
        }
        infile.read(reinterpret_cast<char*>(col),
                    static_cast<std::streamsize>(N) * sizeof(Complex));
        if (!infile) {
            return k;
        }
    }
    return count;
}

void blocked_reorth(uint64_t N,
                    uint64_t count,
                    const Complex* V,
                    Complex* w,
                    double threshold,
                    const uint64_t* global_indices,
                    const std::function<bool(uint64_t)>& skip_predicate) {
    if (count == 0 || N == 0 || V == nullptr || w == nullptr) return;

    std::vector<Complex> overlaps(static_cast<std::size_t>(count),
                                  Complex(0.0, 0.0));
    const Complex one(1.0, 0.0);
    const Complex zero(0.0, 0.0);
    const Complex neg_one(-1.0, 0.0);

    // overlaps = V^H * w   (V is N x count column-major; lda = N)
    cblas_zgemv(CblasColMajor, CblasConjTrans,
                static_cast<int>(N), static_cast<int>(count),
                &one, V, static_cast<int>(N),
                w, 1,
                &zero, overlaps.data(), 1);

    // Threshold + skip filter. We keep both gates because the original
    // per-vector code did the same: skipping below-threshold projections
    // is what makes "selective" re-orth selective in the first place.
    const bool use_threshold = (threshold > 0.0);
    const bool use_skip = static_cast<bool>(skip_predicate);
    for (uint64_t k = 0; k < count; ++k) {
        if (use_skip && skip_predicate(global_indices[k])) {
            overlaps[k] = Complex(0.0, 0.0);
            continue;
        }
        if (use_threshold && std::abs(overlaps[k]) <= threshold) {
            overlaps[k] = Complex(0.0, 0.0);
        }
    }

    // w := w - V * overlaps
    cblas_zgemv(CblasColMajor, CblasNoTrans,
                static_cast<int>(N), static_cast<int>(count),
                &neg_one, V, static_cast<int>(N),
                overlaps.data(), 1,
                &one, w, 1);
}

}  // namespace lanczos_io
