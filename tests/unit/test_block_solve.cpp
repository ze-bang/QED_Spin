// =============================================================================
// tests/unit/test_block_solve.cpp
//
// The per-block solve driver:
//
//   [place]   place(): THE device decision for every block of every verb, with an
//             injected DeviceProbe that counts its calls; with_backend().
//   [thermal] a sampled block up to dense_max_dim is diagonalised on the host (and
//             counted host_dense); dense_max_dim = 0, a spin tower or observables sample it.
//   [lanes]   the Backend-templated block lanes (scan, Krylov-Schur, GS vector, estimate)
//             on toy blocks against Eigen; [lanes][cuda] the same lanes on CudaBackend.
//   [dense]   solve_block_full and solve_block_dense.
//   [members] a level's multiplet gathered into momentum sectors (members_of) against multiplet().
//   [linear_operator] bind<Backend>, has_device_kernel; a host-only operator refuses
//             bind_cuda.
// =============================================================================
#include "common/catch2_harness.h"
#include "common/dense_operator.h"
#include "common/test_harness.h"
#include "engine/internal.h"
#include "engine/walk.h"

#include <ed/core/device.h>
#include <ed/core/errors.h>
#include <ed/core/select_backend.h>
#include <ed/sectors/thermal.h>
#include <ed/ops/casimir_projector.h>
#include <ed/parallel/thread_budget.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using ed::Device;
using ed::Lane;
using ed::Task;

// The Scalar scaffold: every lane runs the complex-double instantiation, and the host-backend
// trait (not the concrete type) decides where vectors live.
static_assert(std::is_same_v<ed::matvec::CpuBackend::scalar_type, std::complex<double>>);
static_assert(std::is_base_of_v<ed::matvec::Backend, ed::matvec::CpuBackend>);
static_assert(ed::matvec::is_cpu_backend_v<ed::matvec::CpuBackend>);
static_assert(std::is_same_v<decltype(ed::krylov::lanczos_kernel(
                                  std::declval<const ed::matvec::CpuBackend&>(),
                                  std::declval<ed::LinearOperator::MatvecFn&>(), std::size_t{},
                                  std::declval<const std::complex<double>*>(),
                                  std::declval<const ed::krylov::LanczosKernelOptions&>())),
                             ed::krylov::LanczosKernelResult>);
#ifdef WITH_CUDA
static_assert(std::is_base_of_v<ed::matvec::Backend, ed::matvec::CudaBackend>);
static_assert(!ed::matvec::is_cpu_backend_v<ed::matvec::CudaBackend>);
#endif

namespace {

// A fake machine: whether a device is visible and how much memory it reports (nullopt: the
// query fails). Counts every call so a test can say "never asked".
struct FakeMachine {
    static inline bool device = true;
    static inline std::optional<std::size_t> free = std::size_t{1} << 40;
    static inline int available_calls = 0, free_calls = 0, fresh_calls = 0;
    static bool available() noexcept { ++available_calls; return device; }
    static std::optional<std::size_t> free_bytes(bool fresh) noexcept {
        ++free_calls;
        if (fresh) ++fresh_calls;
        return free;
    }
    static void reset(bool dev = true, std::optional<std::size_t> f = std::size_t{1} << 40) {
        device = dev; free = f;
        available_calls = free_calls = fresh_calls = 0;
    }
    static ed::DeviceProbe probe() { return {&FakeMachine::available, &FakeMachine::free_bytes}; }
};

constexpr Task kTasks[] = {Task::Eigs, Task::Sampled, Task::Oftlm, Task::DenseBatch,
                           Task::DynamicsCf, Task::DynamicsFtlm};

ed::BlockRequest req(Task t, std::uint64_t dim, bool kernel = true, std::uint64_t want = 1) {
    ed::BlockRequest r;
    r.task = t; r.dim = dim; r.want = want; r.device_kernel = kernel; r.verb = "eigs";
    return r;
}

template <class E>
std::string message_of(Device d, const ed::BlockRequest& r) {
    try {
        (void)ed::place(d, r, FakeMachine::probe());
    } catch (const E& e) {
        return e.what();
    }
    FAIL("place() did not throw the expected class");
    return {};
}

}  // namespace

TEST_CASE("place: the auto table", "[place]") {
    REQUIRE(ed::auto_row(Task::Eigs).floor == 16384);
    REQUIRE(ed::auto_row(Task::Eigs).fit);
    REQUIRE(ed::auto_row(Task::Sampled).floor == 16384);
    REQUIRE(ed::auto_row(Task::Sampled).fit);
    REQUIRE_FALSE(ed::auto_row(Task::Oftlm).fit);
    REQUIRE(ed::auto_row(Task::DenseBatch).floor == 0);
    REQUIRE_FALSE(ed::auto_row(Task::DenseBatch).fit);
    REQUIRE(ed::auto_row(Task::DynamicsCf).floor == 16384);
    REQUIRE_FALSE(ed::auto_row(Task::DynamicsCf).fit);
    REQUIRE(ed::auto_row(Task::DynamicsFtlm).floor == 65536);
    REQUIRE_FALSE(ed::auto_row(Task::DynamicsFtlm).fit);
    REQUIRE(ed::kHostGatherFloor == (std::uint64_t{1} << 20));
    REQUIRE(ed::kHostPoolMaxDim == (std::uint64_t{1} << 16));
    REQUIRE(ed::kDeviceDenseMaxDim == 32);
}

TEST_CASE("place: Cpu never probes the device", "[place]") {
    for (Task t : kTasks)
        for (std::uint64_t dim : {std::uint64_t{1}, std::uint64_t{40}, std::uint64_t{1} << 30})
            for (bool dense : {false, true}) {
                FakeMachine::reset();
                auto r = req(t, dim);
                r.dense = dense;
                const Lane lane = ed::place(Device::Cpu, r, FakeMachine::probe());
                REQUIRE_FALSE(ed::on_device(lane));
                REQUIRE(lane == (t == Task::DenseBatch || dense ? Lane::HostDense : Lane::HostKrylov));
                REQUIRE(FakeMachine::available_calls == 0);
                REQUIRE(FakeMachine::free_calls == 0);
            }
}

TEST_CASE("place: a dense block runs on the host under every device, unprobed", "[place]") {
    for (Device d : {Device::Cpu, Device::Gpu, Device::Auto})
        for (bool kernel : {false, true}) {
            FakeMachine::reset(/*dev=*/false, std::nullopt);
            auto r = req(Task::Eigs, 1u << 20, kernel);
            r.dense = true;
            REQUIRE(ed::place(d, r, FakeMachine::probe()) == Lane::HostDense);
            REQUIRE(FakeMachine::available_calls == 0);
            REQUIRE(FakeMachine::free_calls == 0);
        }
}

TEST_CASE("place: Gpu refuses in order and says why", "[place]") {
    FakeMachine::reset(/*dev=*/false);
    REQUIRE(message_of<ed::DeviceUnavailable>(Device::Gpu, req(Task::Eigs, 1u << 20))
            == "eigs: device='gpu', but no usable CUDA device is visible");

    FakeMachine::reset();
    auto o = req(Task::Oftlm, 1u << 20);
    o.verb = "thermal";
    REQUIRE(message_of<ed::DeviceUnsupported>(Device::Gpu, o)
            == "thermal: OFTLM (exact_states > 0) runs on the host only; with device='gpu' use FTLM "
               "without exact_states, or device='auto' or 'cpu'");

    FakeMachine::reset();
    auto w = req(Task::Eigs, 924, /*kernel=*/false);
    w.what = [] { return std::string("the block of star 3, irrep 1, n_up 6 (dim 924)"); };
    w.why  = "is a sector of an irrep of dimension > 1, which has no device kernel";
    REQUIRE(message_of<ed::DeviceUnsupported>(Device::Gpu, w)
            == "eigs: device='gpu', but the block of star 3, irrep 1, n_up 6 (dim 924) is a sector of an irrep "
               "of dimension > 1, which has no device kernel; use device='auto' or 'cpu'");
    REQUIRE(FakeMachine::free_calls == 0);
    REQUIRE(message_of<ed::DeviceUnsupported>(Device::Gpu, req(Task::Sampled, 500, false))
            == "eigs: device='gpu', but a block of dim 500 has no device kernel; use device='auto' or 'cpu'");

    FakeMachine::reset(true, std::nullopt);
    REQUIRE(message_of<ed::DeviceUnavailable>(Device::Gpu, req(Task::Eigs, 1u << 20))
            == "device='gpu', but the device's memory cannot be queried (no CUDA context could be created)");
    REQUIRE(FakeMachine::fresh_calls == 1);

    // One FTLM sample on the device holds 4 vectors of 16 B: 64 B per state (core/footprint.h).
    FakeMachine::reset(true, (std::size_t{64} << 20) - 1);
    REQUIRE(message_of<ed::ResourceLimit>(Device::Gpu, req(Task::Sampled, std::uint64_t{1} << 20))
            == "device='gpu', but a block of dim 1048576 needs 64 MiB of device memory and 63 MiB are free");
    FakeMachine::reset(true, std::size_t{64} << 20);
    REQUIRE(ed::place(Device::Gpu, req(Task::Sampled, std::uint64_t{1} << 20), FakeMachine::probe())
            == Lane::DeviceKrylov);
    REQUIRE(FakeMachine::fresh_calls == 1);
}

TEST_CASE("place: tasks without a memory row never query memory", "[place]") {
    for (Task t : {Task::DenseBatch, Task::DynamicsCf, Task::DynamicsFtlm})
        for (Device d : {Device::Gpu, Device::Auto}) {
            FakeMachine::reset(true, std::nullopt);
            REQUIRE(ed::on_device(ed::place(d, req(t, std::uint64_t{1} << 20), FakeMachine::probe())));
            REQUIRE(FakeMachine::free_calls == 0);
        }
}

TEST_CASE("place: Auto floors", "[place]") {
    for (Task t : {Task::Eigs, Task::Sampled, Task::DynamicsCf}) {
        FakeMachine::reset();
        REQUIRE(ed::place(Device::Auto, req(t, 16383), FakeMachine::probe()) == Lane::HostKrylov);
        REQUIRE(FakeMachine::available_calls == 0);   // below the floor: CUDA never touched
        REQUIRE(ed::place(Device::Auto, req(t, 16384), FakeMachine::probe()) == Lane::DeviceKrylov);
    }
    FakeMachine::reset();
    REQUIRE(ed::place(Device::Auto, req(Task::DynamicsFtlm, 65535), FakeMachine::probe()) == Lane::HostKrylov);
    REQUIRE(ed::place(Device::Auto, req(Task::DynamicsFtlm, 65536), FakeMachine::probe()) == Lane::DeviceKrylov);
    // Oftlm never goes to the device, whatever the size.
    REQUIRE(ed::place(Device::Auto, req(Task::Oftlm, std::uint64_t{1} << 30), FakeMachine::probe())
            == Lane::HostKrylov);
    // No kernel, or no device: the host, without raising.
    REQUIRE(ed::place(Device::Auto, req(Task::Eigs, 1u << 20, false), FakeMachine::probe()) == Lane::HostKrylov);
    FakeMachine::reset(false);
    REQUIRE(ed::place(Device::Auto, req(Task::Eigs, 1u << 20), FakeMachine::probe()) == Lane::HostKrylov);
}

TEST_CASE("place: Auto with too little device memory stays on the host", "[place]") {
    // The two-pass GS vector: 5 device vectors, 80 B per state.
    FakeMachine::reset(true, (std::size_t{80} << 20) - 1);
    REQUIRE(ed::place(Device::Auto, req(Task::Eigs, std::uint64_t{1} << 20), FakeMachine::probe())
            == Lane::HostKrylov);
    FakeMachine::reset(true, std::size_t{80} << 20);
    REQUIRE(ed::place(Device::Auto, req(Task::Eigs, std::uint64_t{1} << 20), FakeMachine::probe())
            == Lane::DeviceKrylov);
    FakeMachine::reset(true, (std::size_t{80} << 20) - 1);
    REQUIRE(FakeMachine::fresh_calls == 0);   // the cached query
    FakeMachine::reset(true, std::nullopt);
    REQUIRE(ed::place(Device::Auto, req(Task::Sampled, std::uint64_t{1} << 20), FakeMachine::probe())
            == Lane::HostKrylov);
}

TEST_CASE("place: the request's device working set is what must fit", "[place]") {
    auto r = req(Task::Sampled, std::uint64_t{1} << 20);
    r.device_bytes = std::uint64_t{1} << 30;   // e.g. FTLM keeping its basis
    FakeMachine::reset(true, (std::size_t{1} << 30) - 1);
    REQUIRE(ed::place(Device::Auto, r, FakeMachine::probe()) == Lane::HostKrylov);
    REQUIRE_THROWS_AS(ed::place(Device::Gpu, r, FakeMachine::probe()), ed::ResourceLimit);
    FakeMachine::reset(true, std::size_t{1} << 30);
    REQUIRE(ed::place(Device::Auto, r, FakeMachine::probe()) == Lane::DeviceKrylov);
    // Krylov-Schur at its smallest cycle (want + 8), keeping p = k + max(k/2, 8) = 12 Ritz vectors
    // across a restart: (2 (k + 8) + p + 2 k + 8) vectors.
    const auto k = req(Task::Eigs, std::uint64_t{1} << 20, true, 4);
    REQUIRE(ed::device_need(k) == std::uint64_t{(2 * 12 + 12 + 2 * 4 + 8) * 16} << 20);

    // ED_MEM_GUARD_OFF: no memory check, so no query either.
    const char* old = std::getenv("ED_MEM_GUARD_OFF");
    const std::string saved = old ? old : "";
    setenv("ED_MEM_GUARD_OFF", "1", 1);
    FakeMachine::reset(true, std::size_t{1});
    REQUIRE(ed::place(Device::Gpu, r, FakeMachine::probe()) == Lane::DeviceKrylov);
    REQUIRE(ed::place(Device::Auto, r, FakeMachine::probe()) == Lane::DeviceKrylov);
    REQUIRE(FakeMachine::free_calls == 0);
    if (old) setenv("ED_MEM_GUARD_OFF", saved.c_str(), 1); else unsetenv("ED_MEM_GUARD_OFF");
}

TEST_CASE("place: DenseBatch", "[place]") {
    FakeMachine::reset();
    REQUIRE(ed::place(Device::Cpu, req(Task::DenseBatch, 10), FakeMachine::probe()) == Lane::HostDense);
    // Auto: the host pool below the measured crossover, the device from it; Gpu: the device always.
    REQUIRE(ed::place(Device::Auto, req(Task::DenseBatch, 10), FakeMachine::probe()) == Lane::HostDense);
    REQUIRE(ed::place(Device::Auto, req(Task::DenseBatch, ed::kDeviceDenseMinDim - 1), FakeMachine::probe())
            == Lane::HostDense);
    REQUIRE(ed::place(Device::Auto, req(Task::DenseBatch, ed::kDeviceDenseMinDim), FakeMachine::probe())
            == Lane::DeviceDense);
    REQUIRE(ed::place(Device::Gpu, req(Task::DenseBatch, 10, false), FakeMachine::probe()) == Lane::DeviceDense);
    FakeMachine::reset(false);
    REQUIRE(ed::place(Device::Auto, req(Task::DenseBatch, 10), FakeMachine::probe()) == Lane::HostDense);
    REQUIRE_THROWS_AS(ed::place(Device::Gpu, req(Task::DenseBatch, 10), FakeMachine::probe()),
                      ed::DeviceUnavailable);
}

TEST_CASE("place: the transitional small-eigs rule", "[place]") {
    for (Device d : {Device::Gpu, Device::Auto}) {
        FakeMachine::reset();
        if (d == Device::Auto) {   // above the auto floor so the block is device-bound first
            REQUIRE(ed::place(d, req(Task::Eigs, 16384, true, 8192), FakeMachine::probe()) == Lane::HostDense);
            REQUIRE(ed::place(d, req(Task::Eigs, 16384, true, 8191), FakeMachine::probe()) == Lane::DeviceKrylov);
            continue;
        }
        REQUIRE(ed::place(d, req(Task::Eigs, 32), FakeMachine::probe()) == Lane::HostDense);
        REQUIRE(ed::place(d, req(Task::Eigs, 33), FakeMachine::probe()) == Lane::DeviceKrylov);
        REQUIRE(ed::place(d, req(Task::Eigs, 40, true, 20), FakeMachine::probe()) == Lane::HostDense);
        REQUIRE(ed::place(d, req(Task::Eigs, 40, true, 19), FakeMachine::probe()) == Lane::DeviceKrylov);
        // Only Eigs: a small sampled block stays on the device.
        REQUIRE(ed::place(d, req(Task::Sampled, 20), FakeMachine::probe()) == Lane::DeviceKrylov);
    }
}

TEST_CASE("place: the system probe", "[place]") {
    const Lane lane = ed::place(Device::Auto, req(Task::Eigs, std::uint64_t{1} << 20));
#ifndef WITH_CUDA
    REQUIRE(lane == Lane::HostKrylov);
    REQUIRE_THROWS_AS(ed::place(Device::Gpu, req(Task::Eigs, std::uint64_t{1} << 20)), ed::DeviceUnavailable);
#else
    REQUIRE(lane == (ed::have_cuda() ? Lane::DeviceKrylov : Lane::HostKrylov));
#endif
}

TEST_CASE("with_backend: a fresh backend of the lane", "[place]") {
    const bool cpu = ed::with_backend(Lane::HostKrylov, [](auto& be) {
        return std::is_same_v<std::decay_t<decltype(be)>, ed::matvec::CpuBackend>;
    });
    REQUIRE(cpu);
    REQUIRE(ed::with_backend(Lane::HostDense, [](auto& be) {
        return std::is_same_v<std::decay_t<decltype(be)>, ed::matvec::CpuBackend>;
    }));
#ifndef WITH_CUDA
    REQUIRE_THROWS_AS(ed::with_backend(Lane::DeviceKrylov, [](auto&) { return 0; }), std::logic_error);
#endif
}

// -----------------------------------------------------------------------------
// [thermal]: a sampled block at most dense_max_dim states, with no spin tower and no
// observables, is diagonalised on the host; dense_max_dim = 0 restores the kernels.
// -----------------------------------------------------------------------------
namespace {

constexpr int kRing = 6;   // the periodic Heisenberg ring: one 64-state block without symmetry

ed::sectors::Spec one_block() {
    ed::sectors::Spec s;
    s.use_sz = false;
    s.spin_flip = 0;
    s.time_reversal = 0;
    return s;
}

const std::vector<double> kT = {0.05, 0.25, 1.0, 3.0, 10.0};

std::vector<double> ring_spectrum() {
    auto H = ed_tests::build_heisenberg_chain(kRing, 1.0, /*periodic=*/true);
    return ed_tests::reference_from_operator(*H, 1ULL << kRing).eigs;
}

std::vector<double> exact_energies(const std::vector<double>& eigs) {
    const double e0 = *std::min_element(eigs.begin(), eigs.end());
    std::vector<double> E;
    for (double t : kT) {
        double z = 0.0, num = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-(e - e0) / t);
            z += w; num += e * w;
        }
        E.push_back(num / z);
    }
    return E;
}

ed::sectors::ThermalSpec few_samples(ed::sectors::ThermalSpec::Method m, std::size_t exact_states = 0) {
    ed::sectors::ThermalSpec t;
    t.method       = m;
    t.temperatures = kT;
    t.samples      = 4;     // far too few to be accurate...
    t.krylov       = 8;     // ...with a far too short Krylov space
    t.exact_states = exact_states;
    t.seed         = 12345;
    return t;
}

}  // namespace

TEST_CASE("thermal: a block up to dense_max_dim is exact for every sampling method", "[thermal]") {
    using M = ed::sectors::ThermalSpec::Method;
    auto H = ed_tests::build_heisenberg_chain(kRing, 1.0, /*periodic=*/true);
    const auto E = exact_energies(ring_spectrum());
    // 4 samples of 8 Lanczos steps cannot reach 1e-10 at any temperature: the dense path ran.
    for (const auto& [m, exact_states] : {std::pair{M::FTLM, std::size_t{0}}, std::pair{M::FTLM, std::size_t{2}},
                                          std::pair{M::mTPQ, std::size_t{0}}}) {
        const auto r = ed::sectors::thermal(*H, one_block(), few_samples(m, exact_states));
        INFO("method " << static_cast<int>(m) << ", exact_states " << exact_states);
        REQUIRE(r.placement.host_dense == 1);
        REQUIRE(r.placement.host_krylov == 0);
        REQUIRE(r.E.size() == kT.size());
        for (std::size_t i = 0; i < kT.size(); ++i) REQUIRE(std::abs(r.E[i] - E[i]) < 1e-10);
    }
}

TEST_CASE("thermal: dense_max_dim = 0 samples the block", "[thermal]") {
    auto H = ed_tests::build_heisenberg_chain(kRing, 1.0, /*periodic=*/true);
    const auto eigs = ring_spectrum();
    const auto E = exact_energies(eigs);
    auto t = few_samples(ed::sectors::ThermalSpec::Method::FTLM);
    t.dense_max_dim = 0;
    const auto r = ed::sectors::thermal(*H, one_block(), t);
    REQUIRE(r.placement.host_krylov == 1);
    REQUIRE(r.placement.host_dense == 0);
    double worst = 0.0;
    for (std::size_t i = 0; i < kT.size(); ++i) {
        worst = std::max(worst, std::abs(r.E[i] - E[i]));
        REQUIRE(r.E[i] >= eigs.front() - 1e-9);   // a thermal average sits above the ground state
    }
    INFO("worst |dE| when sampling = " << worst);
    REQUIRE(worst > 1e-10);
}

TEST_CASE("thermal: a spin tower or observables take the exact small-block solve too", "[thermal]") {
    // P6.6 (audit P4-thermal-12): below dense_max_dim a block is solved exactly whatever is asked of it
    // -- a spin tower on Q^dag H Q, observables through the eigenvectors -- instead of sampled.
    auto H = ed_tests::build_heisenberg_chain(kRing, 1.0, /*periodic=*/true);
    SECTION("total spin") {
        auto s = one_block();
        s.use_sz = true;
        s.two_S = 0;   // the singlets
        const auto r = ed::sectors::thermal(*H, s, few_samples(ed::sectors::ThermalSpec::Method::FTLM));
        REQUIRE(r.placement.host_dense >= 1);
        REQUIRE(r.placement.host_krylov == 0);
        ed::sectors::ThermalSpec ex;
        ex.method = ed::sectors::ThermalSpec::Method::Exact;
        ex.temperatures = kT;
        const auto x = ed::sectors::thermal(*H, s, ex);
        for (std::size_t i = 0; i < kT.size(); ++i) {
            REQUIRE(std::abs(r.E[i] - x.E[i]) <= 1e-9 * std::max(1.0, std::abs(x.E[i])));
            REQUIRE(std::abs(r.lnZ[i] - x.lnZ[i]) <= 1e-9 * std::max(1.0, std::abs(x.lnZ[i])));
        }
    }
    SECTION("observables") {
        auto t = few_samples(ed::sectors::ThermalSpec::Method::FTLM);
        t.observables = {H.get()};
        const auto r = ed::sectors::thermal(*H, one_block(), t);
        REQUIRE(r.placement.host_dense == 1);
        REQUIRE(r.placement.host_krylov == 0);
        REQUIRE(r.O.size() == 1);
        for (std::size_t i = 0; i < kT.size(); ++i)   // <H>(T) is E(T)
            REQUIRE(std::abs(r.O[0][i] - Complex(r.E[i], 0.0)) <= 1e-9 * std::max(1.0, std::abs(r.E[i])));
    }
}

// -----------------------------------------------------------------------------
// [lanes]: the Backend-templated block lanes on toy blocks against Eigen. Every block is
// above the lanes' own n <= 2 dense guard, so these are the Krylov paths the verbs run
// above their dense crossover (and, on a device, at every size).
// -----------------------------------------------------------------------------
namespace {

namespace lg = ed::solvers::lg_detail;

// XXZ chain: Jxy (S+S- + S-S+)/2 + Jz SzSz on nearest neighbours, plus a uniform field hz.
std::shared_ptr<Operator> xxz_chain(int N, bool periodic, double Jxy, double Jz, double hz) {
    auto H = std::make_shared<Operator>(static_cast<std::uint64_t>(N), 0.5f);
    for (int i = 0; i < (periodic ? N : N - 1); ++i) {
        const auto a = static_cast<std::uint64_t>(i), b = static_cast<std::uint64_t>((i + 1) % N);
        H->addTwoBodyTerm(2, a, 2, b, Complex(Jz, 0));
        H->addTwoBodyTerm(0, a, 1, b, Complex(0.5 * Jxy, 0));
        H->addTwoBodyTerm(1, a, 0, b, Complex(0.5 * Jxy, 0));
    }
    for (int i = 0; hz != 0.0 && i < N; ++i) H->addOneBodyTerm(2, static_cast<std::uint64_t>(i), Complex(hz, 0));
    return H;
}

struct ToyBlock {
    std::string name;
    std::shared_ptr<const ed::LinearOperator> op;
    Eigen::MatrixXcd dense;
};

// Open chain, ring, Ising-heavy and field-XXZ models; full spaces and Sz sectors; dims 3..256.
std::vector<ToyBlock> toy_blocks() {
    struct Model { const char* name; int N; bool periodic; double Jxy, Jz, hz; };
    const Model models[] = {{"open", 6, false, 1.0, 1.0, 0.0},    {"ring", 8, true, 1.0, 1.0, 0.0},
                            {"ising", 7, true, 0.2, 1.0, 0.0},    {"field", 8, true, 1.0, 0.5, 0.31}};
    std::vector<ToyBlock> out;
    for (const Model& m : models) {
        auto H = xxz_chain(m.N, m.periodic, m.Jxy, m.Jz, m.hz);
        const std::uint64_t full = std::uint64_t{1} << m.N;
        out.push_back({std::string(m.name) + "/full", H, ed_tests::reference_from_operator(*H, full).H});
        for (int n_up : {1, m.N / 2}) {
            auto S = std::make_shared<ed_tests::SzSectorOperator>(H, n_up);
            Eigen::MatrixXcd D = ed_tests::apply_to_dense([&S](const Complex* in, Complex* o, int n) {
                S->apply(in, o, static_cast<std::size_t>(n));
            }, S->dim());
            out.push_back({std::string(m.name) + "/n_up=" + std::to_string(n_up), S, std::move(D)});
        }
    }
    return out;
}

std::vector<double> eigen_values(const Eigen::MatrixXcd& D) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(D);
    return {es.eigenvalues().data(), es.eigenvalues().data() + es.eigenvalues().size()};
}

std::shared_ptr<const ed::LinearOperator> sz_sector(int N, bool periodic, int n_up) {
    return std::make_shared<ed_tests::SzSectorOperator>(xxz_chain(N, periodic, 1.0, 1.0, 0.0), n_up);
}

double residual(const ed::LinearOperator& H, double e, const std::vector<Complex>& v) {
    std::vector<Complex> hv(v.size());
    H.apply(v.data(), hv.data(), v.size());
    double r = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) r += std::norm(hv[i] - e * v[i]);
    return std::sqrt(r);
}

}  // namespace

TEST_CASE("lanes: lowest values of toy blocks match Eigen", "[lanes]") {
    ed::matvec::CpuBackend be;
    for (const ToyBlock& b : toy_blocks()) {
        const auto ref = eigen_values(b.dense);
        const std::size_t dim = ref.size();
        REQUIRE(dim >= 3);
        for (std::size_t k = 1; k <= std::min<std::size_t>(dim, 8) + 1; ++k) {
            INFO(b.name << " dim " << dim << " k " << k);
            const auto sol = lg::solve_block_lowest(be, *b.op, k);
            CHECK(sol.converged);              // CHECK: one run lists every failing (block, k)
            CHECK(sol.values.size() == std::min(k, dim));
            for (std::size_t i = 0; i < std::min(sol.values.size(), ref.size()); ++i)
                CHECK(std::abs(sol.values[i] - ref[i]) < 1e-8);
        }
    }
}

TEST_CASE("lanes: the Bethe ground state of the 10-ring", "[lanes]") {
    ed::matvec::CpuBackend be;
    const auto H = sz_sector(10, true, 5);   // 252 states
    const auto sol = lg::solve_block_lowest(be, *H, 1);
    REQUIRE(sol.converged);
    REQUIRE(sol.values.size() == 1);
    REQUIRE(std::abs(sol.values[0] - (-4.515446354492155)) < 1e-8);
    REQUIRE(sol.applies > 0);
}

TEST_CASE("lanes: Krylov-Schur pairs are orthonormal eigenpairs", "[lanes]") {
    ed::matvec::CpuBackend be;
    const auto H = sz_sector(8, false, 4);   // the 8-site open chain at sz = 0: 70 states
    const auto ref = ed_tests::reference_from_operator(*xxz_chain(8, false, 1.0, 1.0, 0.0), 256).eigs;
    const auto sol = lg::solve_block_eigenpairs(be, *H, 5);
    REQUIRE(sol.converged);
    REQUIRE(sol.values.size() == 5);
    REQUIRE(sol.vectors.size() == 5);
    const auto sector = eigen_values(ed_tests::apply_to_dense([&H](const Complex* in, Complex* o, int n) {
        H->apply(in, o, static_cast<std::size_t>(n));
    }, H->dim()));
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE(std::abs(sol.values[i] - sector[i]) < 1e-9);
        REQUIRE(residual(*H, sol.values[i], sol.vectors[i]) < 1e-7);
        for (std::size_t j = 0; j < 5; ++j) {
            Complex g(0, 0);
            for (std::size_t a = 0; a < H->dim(); ++a) g += std::conj(sol.vectors[i][a]) * sol.vectors[j][a];
            REQUIRE(std::abs(g - Complex(i == j ? 1.0 : 0.0, 0.0)) < 1e-12);
        }
    }
    REQUIRE(sector.front() >= ref.front() - 1e-12);
}

TEST_CASE("lanes: the replayed GS vector is certified at small n", "[lanes]") {
    ed::matvec::CpuBackend be;
    for (int N : {6, 8, 10, 12}) {                 // 20, 70, 252, 924 states
        const auto H = sz_sector(N, true, N / 2);
        INFO("N " << N << ", dim " << H->dim());
        const auto g = lg::solve_gs_vector(be, *H, /*kept_basis_max_dim=*/0);
        REQUIRE(g.certified);
        REQUIRE(g.residual <= 1e-8);
        REQUIRE(residual(*H, g.energy, g.vector) <= 1e-8);
        // The kept basis finds the same level.
        const auto k = lg::solve_gs_vector(be, *H);
        REQUIRE(k.certified);
        REQUIRE(std::abs(k.energy - g.energy) < 1e-9);
    }
}

TEST_CASE("lanes: the GS vector stops on the Paige bound; a kept basis saves the replay", "[lanes]") {
    ed::matvec::CpuBackend be;
    const auto H = sz_sector(14, true, 7);   // 3432 states, a unique ground state
    const auto kept = lg::solve_gs_vector(be, *H);
    const auto replay = lg::solve_gs_vector(be, *H, /*kept_basis_max_dim=*/0);
    REQUIRE(kept.certified);
    REQUIRE(replay.certified);
    REQUIRE(std::abs(kept.energy - replay.energy) < 1e-12 * std::abs(kept.energy));
    Complex o(0, 0);
    for (std::size_t a = 0; a < H->dim(); ++a) o += std::conj(kept.vector[a]) * replay.vector[a];
    REQUIRE(1.0 - std::abs(o) < 1e-10);
    // The gate ends the recurrence (the lane used to run a fixed 200 steps); the replay applies
    // H once more per step.
    REQUIRE(kept.applies < 200);
    REQUIRE(replay.applies > kept.applies);
}

TEST_CASE("lanes: a starved budget is reported, not thrown", "[lanes]") {
    ed::matvec::CpuBackend be;
    const auto H = sz_sector(10, true, 5);
    REQUIRE_FALSE(lg::solve_block_lowest(be, *H, 1, /*max_iter=*/4).converged);
    REQUIRE_FALSE(lg::solve_block_lowest(be, *H, 3, /*max_iter=*/4).converged);
    REQUIRE_FALSE(lg::solve_block_eigenpairs(be, *H, 3, /*max_iter=*/4).converged);
    const auto g = lg::solve_gs_vector(be, *H, lg::LanePolicy<ed::matvec::CpuBackend>::gs_kept_basis_max_dim, 4);
    REQUIRE_FALSE(g.certified);
    REQUIRE_FALSE(lg::solve_block_eigenpairs(be, *H, 1, /*max_iter=*/4).converged);
}

TEST_CASE("lanes: an operator's ResourceLimit propagates", "[lanes]") {
    ed::matvec::CpuBackend be;
    for (std::size_t k : {std::size_t{1}, std::size_t{3}}) {
        auto T = std::make_shared<ed_tests::ThrowingOperator>(sz_sector(10, true, 5), 7);
        REQUIRE_THROWS_AS(lg::solve_block_eigenpairs(be, *T, k), ed::ResourceLimit);
    }
}

TEST_CASE("lanes: the pruning estimate is deterministic and an upper bound", "[lanes]") {
    ed::matvec::CpuBackend be;
    const auto H = sz_sector(12, true, 6);
    const auto a = lg::estimate_lowest(be, *H), b = lg::estimate_lowest(be, *H);
    REQUIRE(a.theta == b.theta);
    REQUIRE(a.applies > 0);
    REQUIRE(a.applies <= 40);
    REQUIRE(a.theta >= lg::solve_block_lowest(be, *H, 1).values.front() - 1e-12);
}

#ifdef WITH_CUDA
TEST_CASE("lanes: CudaBackend runs the same lanes as CpuBackend", "[lanes][cuda]") {
    if (!ed::have_cuda()) {
        SUCCEED("no CUDA device: skipped");
        return;
    }
    ed::matvec::CpuBackend cpu;
    ed::matvec::CudaBackend gpu;
    for (const ToyBlock& b : toy_blocks()) {
        ed_tests::DenseOperator D(b.dense);
        REQUIRE(D.has_device_kernel());
        for (std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
            INFO(b.name << " k " << k);
            const auto c = lg::solve_block_lowest(cpu, D, k), g = lg::solve_block_lowest(gpu, D, k);
            REQUIRE(c.converged == g.converged);
            REQUIRE(c.values.size() == g.values.size());
            for (std::size_t i = 0; i < c.values.size(); ++i) REQUIRE(std::abs(c.values[i] - g.values[i]) < 1e-10);
        }
        const auto c1 = lg::solve_block_eigenpairs(cpu, D, 1), g1 = lg::solve_block_eigenpairs(gpu, D, 1);
        REQUIRE(c1.converged == g1.converged);
        if (c1.converged) {
            REQUIRE(std::abs(c1.values[0] - g1.values[0]) < 1e-10);
            const auto ref = eigen_values(b.dense);
            if (ref.size() < 2 || ref[1] - ref[0] > 1e-6) {
                // A unique ground state: the same vector up to a phase.
                Complex o(0, 0);
                for (std::size_t a = 0; a < D.dim(); ++a) o += std::conj(c1.vectors[0][a]) * g1.vectors[0][a];
                REQUIRE(1.0 - std::abs(o) < 1e-10);
            } else {
                // A degenerate E0 (the odd Ising ring): each lane may return any vector of the
                // eigenspace, so require each to be an eigenvector.
                REQUIRE(residual(D, c1.values[0], c1.vectors[0]) < 1e-8);
                REQUIRE(residual(D, g1.values[0], g1.vectors[0]) < 1e-8);
            }
        }
        // The kept-basis lane on the device too.
        const auto kb = lg::solve_gs_vector(gpu, D, /*kept_basis_max_dim=*/D.dim());
        REQUIRE(kb.certified == c1.converged);
    }
    ed_tests::DenseOperator D(ed_tests::apply_to_dense([H = sz_sector(10, true, 5)](const Complex* in, Complex* o, int n) {
        H->apply(in, o, static_cast<std::size_t>(n));
    }, 252));
    // Six levels of 252 states take several thick-restart cycles (m = 48): the restart rewrites the
    // basis columns in place, which the device's staged copies must follow.
    const auto c6 = lg::solve_block_lowest(cpu, D, 6), g6 = lg::solve_block_lowest(gpu, D, 6);
    REQUIRE(c6.converged);
    REQUIRE(g6.converged);
    REQUIRE(c6.values.size() == g6.values.size());
    for (std::size_t i = 0; i < c6.values.size(); ++i) REQUIRE(std::abs(c6.values[i] - g6.values[i]) < 1e-9);
    const auto gk = lg::solve_gs_vector(gpu, D, /*kept_basis_max_dim=*/D.dim());
    REQUIRE(gk.certified);
    REQUIRE(std::abs(gk.energy - c6.values[0]) < 1e-9);
    REQUIRE_FALSE(lg::solve_block_lowest(gpu, D, 1, 4).converged);
    REQUIRE_FALSE(lg::solve_block_lowest(gpu, D, 3, 4).converged);
    REQUIRE_FALSE(lg::solve_block_eigenpairs(gpu, D, 1, 4).converged);
    auto T = std::make_shared<ed_tests::ThrowingOperator>(std::make_shared<ed_tests::DenseOperator>(D.matrix()), 7);
    REQUIRE_THROWS_AS(lg::solve_block_eigenpairs(gpu, *T, 1), ed::ResourceLimit);
}
#endif

// -----------------------------------------------------------------------------
// [dense]: the dense lanes (solve_block_full, solve_block_dense)
// against the reference spectrum.
// -----------------------------------------------------------------------------
TEST_CASE("dense: solve_block_full matches the dense reference", "[dense]") {
    for (std::uint64_t N : {4u, 6u, 8u}) {
        for (double hz : {0.0, 0.37}) {           // with a Zeeman field: no SU(2) degeneracy
            auto H = xxz_chain(static_cast<int>(N), false, 1.0, 1.0, hz);
            const std::uint64_t dim = std::uint64_t{1} << N;
            const auto ref = ed_tests::reference_from_operator(*H, dim);
            const auto ev = lg::solve_block_full(*H);
            INFO("N " << N << " hz " << hz);
            ed_tests::require_eigs_close(ev, ref.eigs, ref.eigs.size(), 1e-9, "solve_block_full");
        }
    }
}

TEST_CASE("dense: solve_block_dense and solve_block_full", "[dense]") {
    auto H = xxz_chain(6, true, 1.0, 0.7, 0.21);
    const auto ref = ed_tests::reference_from_operator(*H, 64).eigs;
    const auto full = lg::solve_block_full(*H);
    ed_tests::require_eigs_close(full, ref, ref.size(), 1e-10, "solve_block_full");
    for (bool vectors : {false, true}) {
        const auto sol = lg::solve_block_dense(*H, 5, vectors);
        REQUIRE(sol.converged);
        REQUIRE(sol.values.size() == 5);
        for (std::size_t i = 0; i < 5; ++i) REQUIRE(std::abs(sol.values[i] - ref[i]) < 1e-10);
        REQUIRE(sol.vectors.size() == (vectors ? 5u : 0u));
        for (std::size_t i = 0; vectors && i < 5; ++i) REQUIRE(residual(*H, sol.values[i], sol.vectors[i]) < 1e-9);
        // More levels than the block holds: the whole block.
        REQUIRE(lg::solve_block_dense(*H, 100, vectors).values.size() == 64);
    }
}

TEST_CASE("dense: a block past the LAPACK index range is refused before it is built", "[dense]") {
    // 32-bit LAPACK addresses n x n only up to n = 46340 (C10-krylov-01: silently wrong spectra).
    if (ed::core::lapack_max_dense_n() > 46340) { SUCCEED("64-bit LAPACK"); return; }
    struct Huge final : ed::LinearOperator {
        std::size_t dim() const override { return 46341; }
        void apply(const Complex*, Complex*, std::size_t) const override {
            throw std::logic_error("a refused block was applied");
        }
    } huge;
    REQUIRE_THROWS_AS(lg::solve_block_full(huge), ed::Unsupported);
    // The dense crossovers stay below the limit: such a block takes the Krylov lanes.
    REQUIRE(lg::lowest_dense_floor(1, 1000000, false) == 46340);
    REQUIRE(lg::lowest_dense_floor(1, 100, true) == 100);
}

TEST_CASE("narrowing: checked_narrow throws instead of wrapping", "[dense]") {
    REQUIRE(ed::core::checked_narrow<int>(std::uint64_t{2147483647}, "x") == 2147483647);
    REQUIRE_THROWS_AS(ed::core::checked_narrow<int>(std::uint64_t{2147483648}, "x"), ed::ResourceLimit);
    REQUIRE_THROWS_AS(ed::core::checked_narrow<std::uint32_t>(-1, "x"), ed::ResourceLimit);
    REQUIRE(ed::core::checked_narrow<std::int32_t>(-5LL, "x") == -5);
}

// -----------------------------------------------------------------------------
// [linear_operator]: the one operator interface.
// -----------------------------------------------------------------------------
TEST_CASE("linear_operator: bind<CpuBackend> is the host apply", "[linear_operator]") {
    auto H = ed_tests::build_heisenberg_chain(4, 1.0, /*periodic=*/true);
    const ed::LinearOperator& base = *H;
    auto mv = base.bind<ed::matvec::CpuBackend>();
    REQUIRE(mv);
    std::vector<Complex> x(16), y(16), z(16);
    for (std::size_t i = 0; i < 16; ++i) x[i] = Complex(0.1 * i, -0.05 * i * i);
    mv(x.data(), y.data(), 16);
    base.apply(x.data(), z.data(), 16);
    REQUIRE(y == z);
}

TEST_CASE("linear_operator: device capability", "[linear_operator]") {
    auto H = ed_tests::build_heisenberg_chain(4, 1.0, /*periodic=*/true);
    REQUIRE_FALSE(H->has_device_kernel());                  // ::Operator has none
    auto S = std::make_shared<ed_tests::SzSectorOperator>(
        std::shared_ptr<const Operator>(ed_tests::build_heisenberg_chain(4, 1.0, true)), 2);
    REQUIRE_FALSE(S->has_device_kernel());
    const Eigen::MatrixXcd Hd = ed_tests::reference_from_operator(*H, 16).H;
    auto dense_h = std::make_shared<ed_tests::DenseOperator>(Hd);
#ifdef WITH_CUDA
    REQUIRE(dense_h->has_device_kernel());
#else
    REQUIRE_FALSE(dense_h->has_device_kernel());
#endif
}

#ifdef WITH_CUDA
TEST_CASE("linear_operator: a host-only operator refuses a device binding", "[linear_operator][cuda]") {
    auto H = ed_tests::build_heisenberg_chain(4, 1.0, /*periodic=*/true);
    REQUIRE_THROWS_AS(H->bind<ed::matvec::CudaBackend>(), ed::DeviceUnsupported);
}
#endif

// The host DenseBatch queues its blocks (within host_budget()) and solves them concurrently (one
// serial LAPACK call per thread, largest first): the spectra are those of the single solves.
TEST_CASE("dense: the host DenseBatch solves its queue concurrently, as single solves do", "[dense]") {
    std::vector<std::shared_ptr<const ed::LinearOperator>> ops;
    for (int N : {8, 10, 12})                       // 33 blocks of 1 to 924 states
        for (int n_up = 0; n_up <= N; ++n_up) ops.push_back(sz_sector(N, n_up % 2 == 0, n_up));
    ed::sectors::detail::DenseBatch batch(ed::Device::Cpu, "test");
    std::vector<std::size_t> ids;
    for (const auto& H : ops) ids.push_back(batch.add(*H));
    batch.solve();
    for (std::size_t i = 0; i < ops.size(); ++i) {
        INFO("block " << i << ", dim " << ops[i]->dim());
        const auto ref = lg::solve_block_full(*ops[i]);
        const auto& got = batch.spectrum(ids[i]);
        REQUIRE(batch.lane(ids[i]) == ed::Lane::HostDense);
        REQUIRE(got.size() == ref.size());
        for (std::size_t j = 0; j < ref.size(); ++j)
            REQUIRE(std::abs(got[j] - ref[j]) <= 1e-12 * std::max(1.0, std::abs(ref[j])));
    }
}

// -----------------------------------------------------------------------------
// [members]: a level's multiplet gathered into momentum sectors (walk.h members_of), which dynamics
// uses for the ground manifold of the folded solve. Every member must be an eigenvector of H in its
// momentum sector, the members must number the level's multiplicity, and in each Sz sector they must
// span what multiplet() builds in the computational basis.
// -----------------------------------------------------------------------------
namespace {

constexpr int kL = 4;   // the 4 x 4 square torus

int site(int x, int y) { return ((x % kL + kL) % kL) + kL * ((y % kL + kL) % kL); }

ed::sectors::Perm square_map(int a, int b, int c, int d, int tx, int ty) {   // (x, y) -> (a x + b y + tx, c x + d y + ty)
    ed::sectors::Perm p(kL * kL);
    for (int y = 0; y < kL; ++y)
        for (int x = 0; x < kL; ++x) p[static_cast<std::size_t>(site(x, y))] = site(a * x + b * y + tx, c * x + d * y + ty);
    return p;
}

std::shared_ptr<Operator> square_j1j2(double J2) {
    auto H = std::make_shared<Operator>(static_cast<std::uint64_t>(kL * kL), 0.5f);
    auto bond = [&](int i, int j, double J) {
        const auto a = static_cast<std::uint64_t>(i), b = static_cast<std::uint64_t>(j);
        H->addTwoBodyTerm(2, a, 2, b, Complex(J, 0));
        H->addTwoBodyTerm(0, a, 1, b, Complex(0.5 * J, 0));
        H->addTwoBodyTerm(1, a, 0, b, Complex(0.5 * J, 0));
    };
    for (int y = 0; y < kL; ++y)
        for (int x = 0; x < kL; ++x) {
            bond(site(x, y), site(x + 1, y), 1.0);
            bond(site(x, y), site(x, y + 1), 1.0);
            bond(site(x, y), site(x + 1, y + 1), J2);
            bond(site(x, y), site(x + 1, y - 1), J2);
        }
    return H;
}

}  // namespace

TEST_CASE("members: a level's multiplet in momentum sectors", "[members]") {
    // One thread: the full-basis multiplet() check runs thousands of short parallel loops, which under
    // ctest's oversubscription (-j cores x cores threads) cost minutes for a test of a second.
    ed::parallel::ThreadBudgetScope one_thread(1);
    auto H = square_j1j2(0.5);
    const int N = kL * kL;
    ed::sectors::Spec s;
    for (int ty = 0; ty < kL; ++ty)
        for (int tx = 0; tx < kL; ++tx) s.abelian.push_back(square_map(1, 0, 0, 1, tx, ty));
    using ed::sectors::Perm;
    const Perm c4 = square_map(0, -1, 1, 0, 0, 0), c2 = square_map(-1, 0, 0, -1, 0, 0), c4i = square_map(0, 1, -1, 0, 0, 0);
    const Perm sx = square_map(1, 0, 0, -1, 0, 0), sy = square_map(-1, 0, 0, 1, 0, 0);
    const Perm sd = square_map(0, 1, 1, 0, 0, 0), sa = square_map(0, -1, -1, 0, 0, 0);
    struct Case { const char* name; std::vector<Perm> residues; int n_up; };
    // C4v: real irreps, two-dimensional E at Gamma and M. C4 alone: complex irreps, folded by K.
    // n_up = 8: the flip-projected half filling; n_up = -1: every Sz, n_up and N - n_up mirrored.
    const Case cases[] = {{"C4v, n_up 8", {c4, c2, c4i, sx, sy, sd, sa}, N / 2},
                          {"C4, n_up 7", {c4, c2, c4i}, N / 2 - 1},
                          {"C4v, every Sz", {c4, c2, c4i, sx, sy, sd, sa}, -1}};
    for (const Case& c : cases) {
        INFO(c.name);
        ed::sectors::Spec sc = s;
        sc.residues = c.residues;
        sc.n_up = c.n_up;
        ed::sectors::EigsOptions eo;
        eo.k = 40;
        eo.vectors = true;
        const auto r = ed::sectors::eigs(*H, sc, eo);
        ed::sectors::detail::MemberSectors ms{*H, s.abelian, N, {}};
        bool saw_d2 = false, saw_star = false, saw_mirror = false, saw_fold = false;
        for (const auto& L : r.levels) {
            const auto& v = r.vectors[static_cast<std::size_t>(L.vector)];
            const std::uint64_t count = L.tag.multiplicity * static_cast<std::uint64_t>(L.mirror);
            INFO("level E " << L.energy << " k0 " << L.tag.k0 << " irrep " << L.tag.irrep << " d " << L.tag.irrep_dim
                 << " star " << L.tag.star_size << " mirror " << L.mirror << " tr_folded " << L.tag.tr_folded);
            const auto members = ed::sectors::detail::members_of(L, v, count, sc, ms);
            REQUIRE(members.size() == count);
            saw_d2 = saw_d2 || L.tag.irrep_dim == 2;
            saw_star = saw_star || L.tag.star_size > 1;
            saw_mirror = saw_mirror || L.mirror == 2;
            saw_fold = saw_fold || L.tag.tr_folded;
            for (std::size_t i = 0; i < members.size(); ++i) {
                const auto& m = members[i];
                REQUIRE(m.v.basis->group_size == static_cast<int>(s.abelian.size()));
                lg::RepSectorMatVec Hm(*H, m.v.basis);
                REQUIRE(residual(Hm, L.energy, m.v.amplitudes) <= 1e-9);
                for (std::size_t j = 0; j < i; ++j) {
                    if (members[j].v.basis != m.v.basis) continue;
                    Complex dot(0, 0);
                    for (std::size_t a = 0; a < m.v.amplitudes.size(); ++a)
                        dot += std::conj(members[j].v.amplitudes[a]) * m.v.amplitudes[a];
                    REQUIRE(std::abs(dot) <= 1e-10);
                }
            }
            // Per Sz sector, the same span as multiplet() in the computational basis: the overlap matrix
            // of the two orthonormal sets is unitary.
            std::set<int> sz;
            for (const auto& m : members) sz.insert(m.sub.n_up);
            for (int n_up : sz) {
                std::vector<std::vector<Complex>> a;
                for (const auto& m : members)
                    if (m.sub.n_up == n_up) a.push_back(ed::sectors::expand(*m.v.basis, m.v.amplitudes, n_up));
                const auto b = ed::sectors::multiplet(sc, N, L, v, n_up);
                REQUIRE(a.size() == b.size());
                Eigen::MatrixXcd G(static_cast<Eigen::Index>(a.size()), static_cast<Eigen::Index>(b.size()));
                for (std::size_t i = 0; i < a.size(); ++i)
                    for (std::size_t j = 0; j < b.size(); ++j) {
                        Complex dot(0, 0);
                        for (std::size_t x = 0; x < a[i].size(); ++x) dot += std::conj(a[i][x]) * b[j][x];
                        G(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = dot;
                    }
                const Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(G.cols(), G.cols());
                REQUIRE((G.adjoint() * G - I).norm() <= 1e-9);
            }
        }
        // The cases reach every kind of member: a two-dimensional irrep, a star, a mirror, a K fold.
        if (c.residues.size() == 7 && c.n_up == N / 2) REQUIRE((saw_d2 && saw_star));
        if (c.residues.size() == 3) REQUIRE(saw_fold);
        if (c.n_up < 0) REQUIRE(saw_mirror);
    }
}
