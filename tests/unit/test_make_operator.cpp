// =============================================================================
// tests/unit/test_make_operator.cpp
//
// Wave A acceptance (Full unified-interface collapse, May 2026):
// targeted unit coverage for the `ed::make_operator(OperatorSpec)`
// factory paths and the new `bind_<Backend>` override matrix.
//
// Coverage matrix:
//
//   * make_operator(InMemoryOperator) round-trips an externally-built
//     Operator unchanged.
//   * make_operator(FilePaths) on a tiny on-disk InterAll deck
//     populates the same term storage as a hand-built Operator.
//   * make_operator(DirectoryPath) routes through the file-paths lane
//     and ends in an equivalent operator.
//   * make_operator(spec.fixed_sz=*) yields a FixedSzOperator with the
//     expected sector dimension.
//   * make_operator with streaming_symmetry + distributed combinations
//     that require axes the simple lane does not provide throws the
//     documented error.
//   * The default-lane bind_* overrides return live callables, while
//     the wrong-backend overrides throw with the documented message.
//
// The streaming-symmetry + distributed lanes are covered by their
// existing per-feature tests (test_streaming_symmetry, the
// test_distributed_* family); this file targets the factory glue.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/make_operator.h>
#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/linear_operator.h>
#include <ed/matvec/backends/cpu_backend.h>

#include <complex>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace {

/// Write a tiny InterAll.dat with a single Sz_i Sz_j coupling on a
/// pair of sites. The HPhi header reader (open_hphi_file_) consumes
/// one separator, the label line, then three more separators before
/// the first data row. Each data row has six fields:
///     Op_i indx_i Op_j indx_j re im
/// where Op = 0 (S+), 1 (S-), 2 (Sz).
void write_interall_sz_sz(const std::filesystem::path& path,
                          int site1, int site2,
                          double j_coupling = 1.0) {
    std::ofstream f(path);
    f << "==== two-body interaction ====\n";
    f << "num_terms 1\n";
    f << "========================\n";
    f << "========================\n";
    f << "========================\n";
    f << "2 " << site1 << " 2 " << site2 << " " << j_coupling
      << " 0.0\n";
}

}  // namespace

TEST_CASE("make_operator: InMemoryOperator source forwards the pointer "
          "unchanged",
          "[make_operator][in_memory]") {
    auto src = std::make_unique<Operator>(static_cast<uint64_t>(4), 0.5f);
    Operator* raw = src.get();

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{std::move(src)};
    spec.num_sites = 4;
    spec.spin_l    = 0.5f;

    auto op = ed::make_operator(std::move(spec));
    REQUIRE(op);
    // The owning pointer should now point at the SAME object the caller
    // handed in (no copy, no deep clone).
    REQUIRE(op.get() == raw);
}

TEST_CASE("make_operator: FilePaths source populates term storage",
          "[make_operator][file_paths]") {
    namespace fs = std::filesystem;
    fs::path tmp =
        fs::temp_directory_path() / "ed_make_operator_files_test";
    fs::create_directories(tmp);

    fs::path inter = tmp / "InterAll.dat";
    write_interall_sz_sz(inter, /*site1=*/0, /*site2=*/1);

    ed::OperatorSpec spec;
    spec.source    = ed::FilePaths{inter.string(), "", "", ""};
    spec.num_sites = 2;
    spec.spin_l    = 0.5f;

    auto op = ed::make_operator(std::move(spec));
    REQUIRE(op);

    // The 4-state Hilbert space has dim 4; |00> picks up the Sz_0 Sz_1
    // = (+1/2)(+1/2) = +1/4 diagonal contribution.
    std::vector<std::complex<double>> x(4, std::complex<double>{0.0, 0.0});
    std::vector<std::complex<double>> y(4, std::complex<double>{0.0, 0.0});
    x[0] = std::complex<double>{1.0, 0.0};

    auto matvec = op->bind<ed::matvec::CpuBackend>();
    REQUIRE(matvec);
    matvec(x.data(), y.data(), x.size());

    REQUIRE(std::abs(y[0].real() - 0.25) < 1e-12);

    fs::remove_all(tmp);
}

TEST_CASE("make_operator: DirectoryPath source loads InterAll from "
          "the supplied directory",
          "[make_operator][directory_path]") {
    namespace fs = std::filesystem;
    fs::path tmp =
        fs::temp_directory_path() / "ed_make_operator_dir_test";
    fs::create_directories(tmp);
    write_interall_sz_sz(tmp / "InterAll.dat", 0, 1);

    ed::OperatorSpec spec;
    spec.source    = ed::DirectoryPath{tmp.string()};
    spec.num_sites = 2;

    auto op = ed::make_operator(std::move(spec));
    REQUIRE(op);
    REQUIRE(op->dim() == 4);

    fs::remove_all(tmp);
}

TEST_CASE("make_operator: fixed_sz axis yields a FixedSzOperator with "
          "the half-filled sector dimension",
          "[make_operator][fixed_sz]") {
    namespace fs = std::filesystem;
    fs::path tmp =
        fs::temp_directory_path() / "ed_make_operator_fixed_sz_test";
    fs::create_directories(tmp);
    write_interall_sz_sz(tmp / "InterAll.dat", 0, 1);

    ed::OperatorSpec spec;
    spec.source    = ed::DirectoryPath{tmp.string()};
    spec.num_sites = 4;
    spec.fixed_sz  = 2;  // n_up = 2 -> Sz=0 sector for N=4

    auto op = ed::make_operator(std::move(spec));
    REQUIRE(op);
    // 4-choose-2 = 6 basis states.
    REQUIRE(op->dim() == 6);

    fs::remove_all(tmp);
}

TEST_CASE("make_operator: streaming_symmetry without DirectoryPath throws "
          "the documented error",
          "[make_operator][streaming_symmetry][error]") {
    ed::OperatorSpec spec;
    spec.source             = ed::FilePaths{};
    spec.num_sites          = 4;
    spec.streaming_symmetry = true;

    REQUIRE_THROWS_AS(ed::make_operator(std::move(spec)),
                       std::runtime_error);
}

TEST_CASE("make_operator: InMemoryOperator + streaming_symmetry rejected",
          "[make_operator][in_memory][error]") {
    ed::OperatorSpec spec;
    spec.source             =
        ed::InMemoryOperator{std::make_unique<Operator>(uint64_t{4}, 0.5f)};
    spec.num_sites          = 4;
    spec.streaming_symmetry = true;

    REQUIRE_THROWS_AS(ed::make_operator(std::move(spec)),
                       std::runtime_error);
}

// ---------------------------------------------------------------------------
// bind_<Backend> override matrix (Wave A2).
// ---------------------------------------------------------------------------

TEST_CASE("LinearOperator: plain Operator's bind_cpu returns a "
          "non-empty callable",
          "[bind_overrides][cpu]") {
    auto H =
        ed_tests::build_heisenberg_chain(4, /*J=*/1.0, /*periodic=*/true);
    const ed::LinearOperator& base = *H;
    auto mv = base.bind<ed::matvec::CpuBackend>();
    REQUIRE(mv);

    const std::size_t dim = static_cast<std::size_t>(1ull << 4);
    std::vector<std::complex<double>> x(dim), y(dim);
    x[0] = std::complex<double>{1.0, 0.0};
    REQUIRE_NOTHROW(mv(x.data(), y.data(), dim));
}

#ifdef WITH_CUDA
// The GPUOperator throw-on-wrong-backend overrides are exercised here.
// We rely on the fact that calling `bind_cpu()` on a GPUOperator
// should throw at construction time of the callable, not on the
// first apply -- because GPUOperator decides which lane is legal
// at bind time.
#include <ed/gpu/gpu_operator.cuh>
TEST_CASE("LinearOperator: GPUOperator's bind_cpu throws the documented "
          "device-only message",
          "[bind_overrides][gpu][error]") {
    GPUOperator gpu(4, 0.5f);
    REQUIRE_THROWS_AS(gpu.bind<ed::matvec::CpuBackend>(),
                       std::runtime_error);
}
#endif
