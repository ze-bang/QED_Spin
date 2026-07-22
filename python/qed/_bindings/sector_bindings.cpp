// =============================================================================
// python/qed/_bindings/sector_bindings.cpp
//
// `qed._core.sector_operators(directory, ...)` -> list[SectorOperator]
//
// Exposes the per-sector symmetry operator built by
// ``ed::make_sector_operators_tagged`` as an applicable Python object. The
// motivating use is a band far wider than the low-lying window the solve
// workflows target: extracting ~rank(P_ice) states means driving a blocked
// filter from the caller, one column block at a time, which needs the sector
// matvec rather than a finished eigenpair list.
//
// Working in a symmetry sector is not merely a speedup here -- it is what
// makes the block fit. A block of `nvec` vectors over a full fixed-Sz space
// costs `nvec * dim_full`; per sector it costs `nvec * dim_full / |G|`.
//
// The set owns the operators (`std::vector<std::unique_ptr<SectorOperator>>`),
// so each handle keeps a `shared_ptr` to the set alive and refers to its
// operator by index. Handles are therefore safe to outlive the Python
// expression that produced them.
// =============================================================================

#include "sector_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/core/basis_utils.h>     // applyPermutation
#include <ed/core/linear_operator.h> // LinearOperator::MatvecFn

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
#include <ed/core/make_operator.h>  // OperatorSpec, SectorOperatorSet, make_sector_operators_tagged
#include <ed/symmetry/sector_operator.h>

namespace py = pybind11;

using Complex      = std::complex<double>;
using ComplexArray = py::array_t<Complex, py::array::c_style | py::array::forcecast>;

namespace {

// ---------------------------------------------------------------------------
// A single sector, kept alive by a shared_ptr to the owning set.
// ---------------------------------------------------------------------------
class SectorHandle {
public:
    SectorHandle(std::shared_ptr<ed::SectorOperatorSet> set, std::size_t index)
        : set_(std::move(set)), index_(index) {}

    // Bind the device matvec once and reuse it: bind_cuda() uploads the rep
    // data, so re-binding per call would dominate the matvec it enables.
    // Throws on a build without WITH_CUDA, which is the honest failure -- the
    // symmetry lane has no weak CPU fallback for the device path.
    void use_device() {
#ifndef WITH_CUDA
        throw std::runtime_error(
            "SectorOperator.use_device: built without WITH_CUDA.");
#else
        if (device_) return;
        device_ = op().bind_cuda();
        // bind_cuda() returns a MatvecFn taking DEVICE pointers (the contract
        // is stated in streaming_symmetry_gpu_mirror.cu). Handing it host
        // pointers does not error -- the kernel writes nothing and returns
        // instantly, which reads as an enormous speedup producing zeros. So we
        // own device buffers and stage through them.
        const std::size_t bytes = static_cast<std::size_t>(op().dim()) * sizeof(Complex);
        check_cuda_(cudaMalloc(&d_in_, bytes), "cudaMalloc(in)");
        check_cuda_(cudaMalloc(&d_out_, bytes), "cudaMalloc(out)");
#endif
    }

    ~SectorHandle() {
#ifdef WITH_CUDA
        // Interpreter teardown can retire the CUDA context before these
        // handles are destroyed, so cudaFree returns cudaErrorCudartUnloading.
        // Swallow it: throwing (or letting thrust throw) from a destructor at
        // shutdown aborts the process and turns a clean run into a core dump.
        if (d_in_)  { cudaFree(d_in_);  d_in_ = nullptr; }
        if (d_out_) { cudaFree(d_out_); d_out_ = nullptr; }
        cudaGetLastError();
#endif
    }
    SectorHandle(const SectorHandle&) = delete;
    SectorHandle& operator=(const SectorHandle&) = delete;
    SectorHandle(SectorHandle&& other) noexcept
        : set_(std::move(other.set_)), index_(other.index_),
          device_(std::move(other.device_)) {
#ifdef WITH_CUDA
        d_in_ = other.d_in_; d_out_ = other.d_out_;
        other.d_in_ = nullptr; other.d_out_ = nullptr;
#endif
    }

    // Kernel-only cost: upload once, run the device matvec ``repeats`` times
    // without touching host memory. Separates the kernel from the PCIe
    // transfer, which at 1.2 GB per vector is not a rounding error.
    double device_matvec_seconds(int repeats) const {
#ifndef WITH_CUDA
        throw std::runtime_error("built without WITH_CUDA");
#else
        if (!device_) throw std::runtime_error("call use_device() first");
        const auto n = static_cast<std::size_t>(op().dim());
        py::gil_scoped_release release;
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) {
            device_(static_cast<const Complex*>(d_in_), static_cast<Complex*>(d_out_), n);
        }
        check_cuda_(cudaDeviceSynchronize(), "sync");
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count() / repeats;
#endif
    }

    [[nodiscard]] bool on_device() const { return static_cast<bool>(device_); }

    void run(const Complex* in, Complex* out, std::size_t n) const {
        if (!device_) {
            op().apply(in, out, n);
            return;
        }
#ifdef WITH_CUDA
        const std::size_t bytes = n * sizeof(Complex);
        check_cuda_(cudaMemcpy(d_in_, in, bytes, cudaMemcpyHostToDevice), "H2D");
        device_(static_cast<const Complex*>(d_in_), static_cast<Complex*>(d_out_), n);
        check_cuda_(cudaMemcpy(out, d_out_, bytes, cudaMemcpyDeviceToHost), "D2H");
#endif
    }

#ifdef WITH_CUDA
    static void check_cuda_(cudaError_t status, const char* what) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA ") + what + ": " +
                                     cudaGetErrorString(status));
        }
    }
#endif

    [[nodiscard]] ed::symmetry::SectorOperator& op() const {
        return *set_->operators[index_];
    }

    [[nodiscard]] std::uint64_t dimension() const {
        return static_cast<std::uint64_t>(op().dim());
    }

    [[nodiscard]] std::vector<int> quantum_numbers() const {
        const auto raw = set_->tags[index_].sector_index;
        if (raw < set_->all_quantum_numbers.size()) {
            return set_->all_quantum_numbers[raw];
        }
        return {};
    }

    [[nodiscard]] std::size_t sector_index() const {
        return set_->tags[index_].sector_index;
    }

    // Diagnostic: how many orbits are actually materialised. The rep-lazy
    // basis reports dim() from rep_dim_ while basis_states is still empty, so
    // a zero here with a nonzero dimension means the orbit elements and
    // coefficients do not exist yet.
    [[nodiscard]] std::size_t materialized_orbits() const {
        return op().basis().sector().basis_states.size();
    }

    // H * v for a single vector.
    ComplexArray apply(const ComplexArray& vec) const {
        const auto dim = dimension();
        auto in = vec.unchecked<1>();
        if (static_cast<std::uint64_t>(in.shape(0)) != dim) {
            throw std::runtime_error(
                "SectorOperator.apply: expected a vector of length " +
                std::to_string(dim) + ", got " + std::to_string(in.shape(0)));
        }
        ComplexArray out(static_cast<py::ssize_t>(dim));
        {
            py::gil_scoped_release release;
            run(vec.data(), out.mutable_data(), static_cast<std::size_t>(dim));
        }
        return out;
    }

    // Amplitude of each requested full-space basis state on this sector.
    //
    // Returns ``(indices, amplitudes)``: the sector basis index each state
    // lands on -- or -1 when its orbit cancels in this irrep -- and the
    // amplitude <idx|state>.
    //
    // Reads the CSR-free ``RepSectorData`` (reps / inv_norms / characters /
    // perms_flat), NOT ``SectorBasis::sector().basis_states``. On the
    // production rep-lazy path the latter is empty and stays empty even after
    // a matvec (an apply materialises the CSR, which is a different
    // structure), so a projection built on it silently resolves every state to
    // -1. The rep data is what the matvec itself consumes, so this shares the
    // matvec's basis by construction and needs no materialisation.
    //
    // The orbit and its coefficients are rebuilt exactly as
    // ``compute_orbit_for_state`` does -- accumulate conj(chi_g) over the group
    // onto distinct images -- then normalised so the symmetrised vectors are
    // orthonormal. That normalisation differs from the stored ``norm`` by the
    // uniform factor sqrt(|G|), which cancels out of the polar pullback
    // (h = G^-1/2 A G^-1/2 is invariant under a uniform rescaling of the
    // gather), while any PER-ORBIT mismatch would not.
    py::tuple project_states(const py::array_t<std::uint64_t,
                                               py::array::c_style |
                                               py::array::forcecast>& states) const {
        auto requested = states.unchecked<1>();
        const auto count = static_cast<std::size_t>(requested.shape(0));
        py::array_t<std::int64_t> indices(static_cast<py::ssize_t>(count));
        ComplexArray amplitudes(static_cast<py::ssize_t>(count));
        auto* index_out = indices.mutable_data();
        auto* amp_out   = amplitudes.mutable_data();

        const auto& rep = op().basis().ensureRepData();
        if (!rep.usable()) {
            throw std::runtime_error(
                "SectorOperator.project_states: RepSectorData is not usable "
                "for this sector.");
        }
        const auto& reps       = rep.reps;
        const auto& characters = rep.characters;
        const auto& perms      = rep.perms_flat;
        const int   n_sites    = rep.n_sites;
        const std::size_t group = characters.size();
        if (group == 0 || perms.size() != group * static_cast<std::size_t>(n_sites)) {
            throw std::runtime_error(
                "SectorOperator.project_states: inconsistent group data "
                "(characters/perms_flat do not agree).");
        }
        // Binary search below needs ascending reps; the builders promise it,
        // but a silent violation would mis-index every state.
        if (!std::is_sorted(reps.begin(), reps.end())) {
            throw std::runtime_error(
                "SectorOperator.project_states: orbit representatives are not "
                "ascending, so the index lookup would be wrong.");
        }

        auto permute = [&](std::uint64_t state, std::size_t g) {
            std::uint64_t out = 0;
            const int* row = perms.data() + g * static_cast<std::size_t>(n_sites);
            for (int i = 0; i < n_sites; ++i) {
                out |= ((state >> row[i]) & 1ULL) << i;
            }
            return out;
        };

        std::vector<std::uint64_t> orbit(group);
        std::vector<Complex>       coeff(group);

        // Scope the release to the compute loop only: the return below
        // allocates a Python tuple, which segfaults if the GIL is still
        // released when the function exits.
        {
        py::gil_scoped_release release;
        for (std::size_t s = 0; s < count; ++s) {
            const std::uint64_t state = requested(static_cast<py::ssize_t>(s));
            index_out[s] = -1;
            amp_out[s]   = Complex(0.0, 0.0);

            std::uint64_t canonical = state;
            for (std::size_t g = 0; g < group; ++g) {
                canonical = std::min(canonical, permute(state, g));
            }
            const auto found = std::lower_bound(reps.begin(), reps.end(), canonical);
            if (found == reps.end() || *found != canonical) {
                continue;  // orbit cancels in this irrep
            }
            index_out[s] = static_cast<std::int64_t>(found - reps.begin());

            // Rebuild the symmetrised combination on this orbit.
            std::size_t distinct = 0;
            for (std::size_t g = 0; g < group; ++g) {
                const std::uint64_t image = permute(canonical, g);
                const Complex contribution = std::conj(characters[g]);
                bool seen = false;
                for (std::size_t j = 0; j < distinct; ++j) {
                    if (orbit[j] == image) { coeff[j] += contribution; seen = true; break; }
                }
                if (!seen) { orbit[distinct] = image; coeff[distinct] = contribution; ++distinct; }
            }
            double total = 0.0;
            for (std::size_t j = 0; j < distinct; ++j) total += std::norm(coeff[j]);
            if (total <= 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < distinct; ++j) {
                if (orbit[j] == state) {
                    amp_out[s] = std::conj(coeff[j]) / std::sqrt(total);
                    break;
                }
            }
        }
        }  // GIL reacquired here, before the tuple is built
        return py::make_tuple(indices, amplitudes);
    }

    // H * B for a block of vectors, ONE VECTOR PER ROW.
    //
    // Row-major layout is deliberate: it makes each vector contiguous, so the
    // block loop hands the kernel exactly the same memory pattern as the
    // single-vector path with no gather. A column-major (dim, nvec) block
    // would need a strided copy per column, which at these dimensions is the
    // same cost as the matvec itself.
    ComplexArray apply_block(const ComplexArray& block) const {
        const auto dim = dimension();
        auto in = block.unchecked<2>();
        if (static_cast<std::uint64_t>(in.shape(1)) != dim) {
            throw std::runtime_error(
                "SectorOperator.apply_block: expected shape (nvec, " +
                std::to_string(dim) + ") with one vector per row, got (" +
                std::to_string(in.shape(0)) + ", " +
                std::to_string(in.shape(1)) + ")");
        }
        const auto nvec = static_cast<std::size_t>(in.shape(0));
        ComplexArray out({static_cast<py::ssize_t>(nvec),
                          static_cast<py::ssize_t>(dim)});
        const Complex* src = block.data();
        Complex*       dst = out.mutable_data();
        {
            py::gil_scoped_release release;
            // The kernel is itself threaded over the sector, so the column
            // loop stays serial; parallelising here would oversubscribe.
            for (std::size_t v = 0; v < nvec; ++v) {
                run(src + v * dim, dst + v * dim, static_cast<std::size_t>(dim));
            }
        }
        return out;
    }

private:
    std::shared_ptr<ed::SectorOperatorSet> set_;
    std::size_t                            index_;
    ed::LinearOperator::MatvecFn           device_{};
#ifdef WITH_CUDA
    void*                                  d_in_{nullptr};
    void*                                  d_out_{nullptr};
#endif
};

}  // namespace

void bind_sectors(py::module_& m) {
    py::class_<SectorHandle>(m, "SectorOperator", R"pbdoc(
        One symmetry sector of a Hamiltonian, exposed as a matvec.

        Obtained from :func:`qed._core.sector_operators`. Holds a reference to
        the owning sector set, so it stays valid independently of the call that
        produced it.
        )pbdoc")
        .def_property_readonly("dimension", &SectorHandle::dimension,
                               "Dimension of this symmetry sector.")
        .def_property_readonly("quantum_numbers", &SectorHandle::quantum_numbers,
                               "Irrep labels (e.g. momentum indices) of this sector.")
        .def_property_readonly("sector_index", &SectorHandle::sector_index,
                               "Raw irrep index within the symmetry group.")
        .def_property_readonly("materialized_orbits", &SectorHandle::materialized_orbits,
                               "Number of orbits with materialised elements/coefficients. "
                               "Zero alongside a nonzero dimension means the rep-lazy "
                               "basis has not built the orbit data.")
        .def("apply", &SectorHandle::apply, py::arg("vec"),
             "Compute H * v for a 1-D complex128 array of length ``dimension``.")
        .def("apply_block", &SectorHandle::apply_block, py::arg("block"),
             "Compute H * B for a complex128 array of shape ``(nvec, dimension)`` "
             "-- one vector per row.")
        .def("use_device", &SectorHandle::use_device,
             "Bind the CUDA matvec for this sector and route apply/apply_block "
             "through it. Raises on a build without WITH_CUDA.")
        .def("device_matvec_seconds", &SectorHandle::device_matvec_seconds,
             py::arg("repeats") = 10,
             "Mean seconds per device matvec with the vectors already resident, "
             "excluding host<->device transfer.")
        .def_property_readonly("on_device", &SectorHandle::on_device,
                               "True once use_device() has bound the device matvec.")
        .def("project_states", &SectorHandle::project_states, py::arg("states"),
             R"pbdoc(
             Amplitudes of full-space basis states on this sector.

             Takes a uint64 array of full-space bit states and returns
             ``(indices, amplitudes)``: the sector basis index each state lands
             on (``-1`` if its orbit cancels in this irrep) and the amplitude
             ``<idx|state>``. Use it to gather a distinguished full-space
             subspace into the symmetry-adapted basis without reproducing the
             orbit normalisation convention by hand.
             )pbdoc");

    m.def(
        "sector_operators",
        [](const std::string& directory,
           std::uint64_t      num_sites,
           double             spin_l,
           py::object         fixed_sz_n_up,
           py::object         basis_cache_dir,
           int                mpi_rank,
           int                mpi_size) {
            ed::OperatorSpec spec;
            spec.source             = ed::DirectoryPath{directory};
            spec.num_sites          = num_sites;
            spec.spin_l             = static_cast<float>(spin_l);
            spec.streaming_symmetry = true;
            if (!fixed_sz_n_up.is_none()) {
                spec.fixed_sz = fixed_sz_n_up.cast<int>();
            }
            if (!basis_cache_dir.is_none()) {
                spec.basis_cache_dir = basis_cache_dir.cast<std::string>();
            }

            auto set = std::make_shared<ed::SectorOperatorSet>();
            {
                py::gil_scoped_release release;
                *set = ed::make_sector_operators_tagged(spec, mpi_rank, mpi_size);
            }

            py::list handles;
            for (std::size_t i = 0; i < set->operators.size(); ++i) {
                if (set->operators[i]) {
                    handles.append(SectorHandle(set, i));
                }
            }
            return handles;
        },
        py::arg("directory"), py::arg("num_sites"), py::arg("spin_l") = 0.5,
        py::arg("fixed_sz_n_up") = py::none(),
        py::arg("basis_cache_dir") = py::none(),
        py::arg("mpi_rank") = 0, py::arg("mpi_size") = 1,
        R"pbdoc(
        Build the symmetry-adapted sector operators for a Hamiltonian directory.

        ``directory`` must hold the usual ``Trans.dat`` / ``InterAll.dat`` plus
        the symmetry group metadata consumed by
        ``ed::make_sector_operators_tagged``. Returns one
        :class:`SectorOperator` per non-empty irrep.

        The symmetry-adapted basis depends only on the group and the site
        count, not on the term values, so passing the same ``basis_cache_dir``
        across a parameter sweep builds it once.
        )pbdoc");
}
