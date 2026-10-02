#pragma once
// =============================================================================
// include/ed/core/footprint.h
//
// The working set of each solver path, in bytes, on the host and on the device: the one
// estimate every memory guard and planner reads (thermal's guard, the GPU sample batch width,
// place()'s device fit, the Krylov-Schur subspace cap, the dense crossover, the dense batches,
// multiplet). A Krylov path counts the length-D vectors it holds at its peak (V = 16 D bytes,
// 8 D on a real lane), read from the kernels; a dense path counts its matrices. Operators (CSR, device mirror)
// are budgeted by their own knobs and are not included.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ed::core {

struct Footprint {
    std::uint64_t host   = 0;
    std::uint64_t device = 0;
};

enum class Path {
    /// `width` FTLM samples, LocalDGKS3 without a kept basis: per sample the seed on the host and
    /// on the lane, and three working vectors; on the device one host vector per sample.
    FtlmSample,
    /// `width` FTLM samples keeping their `krylov`-vector basis (observables): per sample the
    /// basis, the seed, three working vectors and w; on the device also the backend's staging
    /// copy of the basis.
    FtlmSampleKept,
    /// `width` mTPQ samples: the spectral-bounds Lanczos (5 vectors), then psi and H psi per
    /// sample plus its host seed.
    Mtpq,
    /// Thick-restart Krylov-Schur for `k` levels with a `krylov`-vector cycle: the m + 1 basis
    /// columns, the p = k + max(k/2, 8) kept Ritz vectors of a restart (the GEMM's target), the k
    /// pairs found and a working vector: m + p + k + 3 on the host; on the device the staging copy
    /// of the found and basis columns as well.
    KrylovSchur,
    /// The certified ground-state vector keeping `krylov` Krylov vectors: those, the seed, three
    /// recurrence vectors and the Ritz vector; on the device also its host copy.
    GsKeptBasis,
    /// The certified ground-state vector replaying its recurrence (no basis): GsKeptBasis at
    /// krylov = 0.
    GsTwoPass,
    /// A dense block's eigenvalues: the complex matrix and a real copy of it.
    DenseValues,
    /// A dense block's eigenpairs: the matrix and its eigenvectors.
    DenseVectors,
    /// `k` vectors of a multiplet in an Sz sector of dimension `dim`: the vectors, the expanded
    /// state and a seed, and the 8-byte state table.
    Multiplet,
    /// `width` samples of finite-temperature dynamics from a source sector of `dim` states to a
    /// target of `dim_target`: per sample both `krylov`-vector bases with their seeds and working
    /// vectors, and one target scratch vector; on the device also the backend's staging buffer
    /// (one, sized for the larger basis) and the seed on the host.
    DynamicsFtlm,
};

struct Shape {
    std::uint64_t dim    = 0;      ///< D
    std::uint64_t dim_target = 0;  ///< DynamicsFtlm: the target sector's dimension
    std::size_t   krylov = 0;      ///< Lanczos depth, or the Krylov-Schur cycle length
    std::size_t   k      = 0;      ///< levels, or multiplet members
    std::size_t   width  = 1;      ///< samples in lockstep (device)
    std::size_t   scalar_bytes = 16;   ///< bytes of one vector entry: 16 complex, 8 on a real lane
    bool          device = false;  ///< the Krylov vectors live on the device
    bool          tower  = false;  ///< seeds projected onto a spin tower (one more host vector)
};

/// Peak bytes of `p` at shape `s` (saturating; a dimension this large never fits anyway).
[[nodiscard]] inline Footprint footprint(Path p, const Shape& s) {
    const double V = static_cast<double>(s.scalar_bytes) * static_cast<double>(s.dim);
    const double D2 = static_cast<double>(s.dim) * static_cast<double>(s.dim);
    const double M = static_cast<double>(s.krylov), k = static_cast<double>(s.k);
    const double W = static_cast<double>(std::max<std::size_t>(s.width, 1));
    const double t = s.tower ? 1.0 : 0.0;
    double host = 0.0, dev = 0.0;
    switch (p) {
        case Path::FtlmSample:
            if (s.device) { dev = W * 4.0 * V; host = W * (1.0 + t) * V; }
            else          { host = W * (5.0 + t) * V; }
            break;
        case Path::FtlmSampleKept:
            if (s.device) { dev = W * (2.0 * M + 5.0) * V; host = W * (1.0 + t) * V; }
            else          { host = W * (M + 6.0 + t) * V; }
            break;
        case Path::Mtpq:
            if (s.device) { dev = std::max(5.0, 2.0 * W) * V; host = W * (1.0 + t) * V; }
            else          { host = (5.0 + t) * V; }
            break;
        case Path::KrylovSchur: {
            const double p = k + std::max(std::floor(k / 2.0), 8.0);   // the vectors a restart keeps
            if (s.device) { dev = (2.0 * M + p + 2.0 * k + 8.0) * V; host = 3.0 * V; }
            else          { host = (M + p + k + 3.0) * V; }
            break;
        }
        case Path::GsKeptBasis:
        case Path::GsTwoPass: {
            const double kept = p == Path::GsKeptBasis ? std::min(M, static_cast<double>(s.dim)) : 0.0;
            if (s.device) { dev = (kept + 5.0) * V; host = V; }
            else          { host = (kept + 5.0) * V; }
            break;
        }
        case Path::DenseValues:
            host = 24.0 * D2;
            break;
        case Path::DenseVectors:
            host = 32.0 * D2;
            break;
        case Path::Multiplet:
            host = (k + 2.0) * V + 8.0 * static_cast<double>(s.dim);
            break;
        case Path::DynamicsFtlm: {
            const double Vt = 16.0 * static_cast<double>(s.dim_target);
            if (s.device) { dev = W * ((M + 4.0) * V + (M + 5.0) * Vt + M * std::max(V, Vt)); host = W * V; }
            else          { host = W * ((M + 5.0) * V + (M + 5.0) * Vt); }
            break;
        }
    }
    constexpr double cap = 1.8e19;   // below 2^64
    return {static_cast<std::uint64_t>(std::min(host, cap)), static_cast<std::uint64_t>(std::min(dev, cap))};
}

}  // namespace ed::core
