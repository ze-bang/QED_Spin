// =============================================================================
// src/observables/ftlm_cross_irrep_kernel.cpp
//
// Host entry point of the cross-sector FTLM dynamical kernel (the estimator itself is
// ftlm_dynamics_kernel.h, shared with the GPU path).
// =============================================================================

#include <ed/observables/ftlm_cross_irrep_kernel.h>
#include <ed/observables/ftlm_dynamics_kernel.h>

#include <ed/matvec/backends/cpu_backend.h>

#include <vector>

namespace ed::observables {

FtlmCrossIrrepSectorResult ftlm_cross_irrep_kernel_one_sector(
    const std::function<void(const Complex*, Complex*, int)>& H_src,
    const std::function<void(const Complex*, Complex*, int)>& H_dst,
    const std::function<void(const Complex*, Complex*, int)>& O_apply,
    std::size_t                       dim_src,
    std::size_t                       dim_dst,
    const std::vector<double>&        temperatures,
    const std::vector<double>&        omega_grid,
    const FtlmCrossIrrepOptions&      opts)
{
    auto wrap = [](const std::function<void(const Complex*, Complex*, int)>& f) {
        return [&f](const Complex* in, Complex* out, std::size_t n) { f(in, out, static_cast<int>(n)); };
    };
    return ftlm_dynamics_kernel(ed::matvec::default_cpu_backend(), wrap(H_src), wrap(H_dst), wrap(O_apply),
                                dim_src, dim_dst, temperatures, omega_grid, opts);
}

}  // namespace ed::observables
