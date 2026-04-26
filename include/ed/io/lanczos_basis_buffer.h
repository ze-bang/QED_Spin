#pragma once

// =============================================================================
// In-memory Lanczos basis-vector buffer
// =============================================================================
//
// Historically the Lanczos family of solvers in this package stored each basis
// vector on disk (one binary file per vector) and re-read them during
// reorthogonalization and eigenvector reconstruction. For anything but a tiny
// problem the per-vector file I/O completely dominates the wall-clock cost —
// we pay ~O(m * N) of double-complex traffic through the filesystem layer for
// every Lanczos run.
//
// This header provides a drop-in buffer that the existing read/write helpers
// in lanczos.cpp route through, keyed by the same `temp_dir` strings the
// solvers already use. When a buffer is registered, reads and writes are
// served from RAM; when no buffer is registered (or when the user forces disk
// mode via the environment variable ED_LANCZOS_DISK=1) the solver falls back
// to the legacy on-disk path untouched.
//
// The abstraction is intentionally keyed by string so the minimal patch to
// lanczos.cpp is just:
//     register_basis_buffer(temp_dir, N, max_iter);  // at solver entry
//     ... existing read_basis_vector / write_basis_vector calls ...
//     release_basis_buffer(temp_dir);                // at solver exit
//
// Thread-safety: the registry itself is protected by an internal mutex, and
// a given buffer's vectors are append-only + random-read. Concurrent reads
// from multiple threads are safe once all writes for the relevant indices
// have completed (this mirrors the existing file-based usage pattern).
// =============================================================================

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace lanczos_io {

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Returns true iff the current process has ED_LANCZOS_DISK=1 in its
// environment, meaning the caller should *not* register in-memory buffers and
// should instead use the legacy on-disk path. Evaluated once, cached.
bool force_disk_storage();

// Register an in-memory basis buffer under `key`. `N` is the vector dimension
// and `reserve_vectors` is an upper bound on the number of vectors that will
// be appended (used only for capacity reservation). Subsequent calls with the
// same key replace any previous registration (and drop its storage).
//
// If force_disk_storage() is true this is a no-op.
void register_basis_buffer(const std::string& key,
                           uint64_t N,
                           uint64_t reserve_vectors);

// Release the buffer for `key` and free its memory. No-op if not registered.
void release_basis_buffer(const std::string& key);

// Does an in-memory buffer exist for `key`?
bool has_basis_buffer(const std::string& key);

// Append one vector to the buffer for `key`. Returns false if no buffer is
// registered for this key (caller should then fall back to on-disk storage)
// or if the dimension does not match.
bool append_basis_vector(const std::string& key, const ComplexVector& vec);

// Move-append: avoids a copy of ``vec`` when the caller no longer needs it
// (saves one dim-N memcpy on every Lanczos iteration for eigenvector runs).
bool append_basis_vector(const std::string& key, ComplexVector&& vec);

// Overwrite the vector at `index` in place. Returns false if no buffer is
// registered, the index is out of range, or the dimension does not match.
// Used by restart algorithms (Krylov-Schur, IRL, thick-restart).
bool set_basis_vector(const std::string& key,
                      uint64_t index,
                      const ComplexVector& vec);

// Drop all vectors with index >= new_size. No-op if not registered or if
// new_size >= current size.
void truncate_basis_buffer(const std::string& key, uint64_t new_size);

// Read the i-th stored vector into `out`. Returns false if the buffer is not
// registered, the index is out of range, or a dimension mismatch is detected.
bool get_basis_vector(const std::string& key,
                      uint64_t index,
                      ComplexVector& out);

// Zero-copy accessor. Returns nullptr if the buffer is not registered or the
// index is out of range. The returned pointer is valid as long as the buffer
// exists and no new vectors are appended (appending may invalidate older
// pointers if storage has to grow).
const Complex* get_basis_vector_ptr(const std::string& key, uint64_t index);

// Number of vectors currently stored for `key`; 0 if not registered.
uint64_t basis_buffer_size(const std::string& key);

// Return an estimate (in bytes) of memory currently held by all registered
// buffers. Useful for diagnostic logging.
uint64_t total_basis_buffer_bytes();

} // namespace lanczos_io
