#pragma once

// =============================================================================
// Out-of-core blocked-tile reorthogonalization for Lanczos       (Phase 3a #2)
// =============================================================================
//
// At N=36 with full point-group + Sz symmetry the Krylov basis spills to disk
// (ED_LANCZOS_DISK=1). Today the periodic-full and selective re-orth passes in
// `lanczos_selective_reorth` random-access that on-disk basis ONE vector at a
// time: at iteration j ≈ 200 they trigger ~200 file open + 16N-byte read +
// BLAS-1 zdotc + BLAS-1 zaxpy operations PER iteration. The per-call file
// open dominates, then the BLAS-1 cost is itself bandwidth-bound on a
// 16N-byte vector with no cache reuse.
//
// This header provides two small primitives that let the re-orth loop work
// in tiles of B basis vectors:
//
//   * load_basis_tile(temp_dir, k_start, count, N, tile)
//         Fills a column-major N × count tile from either the in-memory
//         buffer (zero-copy fast path through `lanczos_io::get_basis_vector_ptr`,
//         followed by one memcpy per column) or the legacy on-disk store
//         (one ifstream read per column, but consecutive index access
//         keeps the reads sequential and the OS page cache warm).
//
//   * blocked_reorth(N, count, V, w, threshold, indices, skip_predicate)
//         Performs ONE classical-Gram-Schmidt pass against the tile via
//         two BLAS-2 zgemv calls:
//
//             overlaps = V^H w           (N × count, ConjTrans)
//             threshold + skip filter
//             w := w - V * overlaps      (N × count, NoTrans)
//
//         Replacing 2*count BLAS-1 calls with two BLAS-2 calls cuts the
//         per-iteration call overhead by ~B× and lets the matrix engine
//         exploit register blocking + L2 reuse on the tile.
//
// Stability: a single CGS pass is well-known to be slightly less stable
// than modified Gram-Schmidt (Björck 1994). The default Lanczos pattern
// is therefore "two CGS passes per tile" (CGS2), which Giraud, Langou &
// Rozložník (2005) prove is backward-stable to the same bound as MGS.
// The caller is responsible for invoking blocked_reorth twice when CGS2
// is desired; this header does not impose a policy.
//
// Tile size knob:
//   ED_LANCZOS_REORTH_TILE   default 16, clamped to [1, 256].
// Larger tiles amortize call overhead better but inflate the working
// set; 16 × N complex doubles at N=10⁸ is 25 GB which exceeds typical L3
// but fits in DRAM.
// =============================================================================

#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lanczos_io {

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Reads the configured tile size from ED_LANCZOS_REORTH_TILE (default 16,
// clamped to [1, 256]). Cheap: re-reads each call.
uint64_t reorth_tile_size();

// Load up to `count` consecutive basis vectors starting at `k_start` into a
// column-major N × count tile. Returns the number of columns actually
// loaded (less than `count` if a vector is missing, e.g. a corrupt run).
//
// `tile` is resized to `N * count` complex entries. Layout: column k holds
// basis vector k_start + k at offsets [k*N, (k+1)*N).
//
// In-memory buffers (registered via `register_basis_buffer`) take a
// memcpy fast path; otherwise the legacy on-disk format
// "<temp_dir>/basis_<index>.dat" is read via ifstream.
uint64_t load_basis_tile(const std::string& temp_dir,
                         uint64_t k_start,
                         uint64_t count,
                         uint64_t N,
                         std::vector<Complex>& tile);

// One pass of blocked classical Gram-Schmidt against the tile V (N × count,
// column-major):
//
//     overlaps = V^H * w
//     for k in 0..count:
//         if skip_predicate(global_indices[k]) or |overlaps[k]| <= threshold:
//             overlaps[k] = 0
//     w := w - V * overlaps
//
// Two BLAS-2 zgemv calls; the inner threshold + skip pass is O(count). The
// caller passes `global_indices[k]` = the basis index that column k
// corresponds to so `skip_predicate` can mask out ring-buffer overlap with
// recent vectors that have already been handled separately.
//
// Use threshold <= 0 to disable filtering (every overlap is applied; no
// short-circuit). Set skip_predicate to nullptr or a constant-false lambda
// to disable index skipping.
void blocked_reorth(uint64_t N,
                    uint64_t count,
                    const Complex* V,
                    Complex* w,
                    double threshold,
                    const uint64_t* global_indices,
                    const std::function<bool(uint64_t)>& skip_predicate);

}  // namespace lanczos_io
