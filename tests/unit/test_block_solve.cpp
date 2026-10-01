// =============================================================================
// tests/unit/test_block_solve.cpp
//
// The per-block solve driver:
//
//   [place]   place(): THE device decision for every block of every verb, with an
//             injected DeviceProbe that counts its calls; with_backend().
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/core/device.h>
#include <ed/core/errors.h>
#include <ed/core/select_backend.h>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

using ed::Device;
using ed::Lane;
using ed::Task;

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
    REQUIRE(ed::auto_row(Task::Eigs).fit_vectors == 8);
    REQUIRE(ed::auto_row(Task::Sampled).floor == 16384);
    REQUIRE(ed::auto_row(Task::Sampled).fit_vectors == 8);
    REQUIRE(ed::auto_row(Task::Oftlm).fit_vectors == 0);
    REQUIRE(ed::auto_row(Task::DenseBatch).floor == 0);
    REQUIRE(ed::auto_row(Task::DenseBatch).fit_vectors == 0);
    REQUIRE(ed::auto_row(Task::DynamicsCf).floor == 16384);
    REQUIRE(ed::auto_row(Task::DynamicsCf).fit_vectors == 0);
    REQUIRE(ed::auto_row(Task::DynamicsFtlm).floor == 65536);
    REQUIRE(ed::auto_row(Task::DynamicsFtlm).fit_vectors == 0);
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
    w.why  = "is an isotypic (W) block, which has no device kernel";
    REQUIRE(message_of<ed::DeviceUnsupported>(Device::Gpu, w)
            == "eigs: device='gpu', but the block of star 3, irrep 1, n_up 6 (dim 924) is an isotypic (W) "
               "block, which has no device kernel; use device='auto' or 'cpu'");
    REQUIRE(FakeMachine::free_calls == 0);
    REQUIRE(message_of<ed::DeviceUnsupported>(Device::Gpu, req(Task::Sampled, 500, false))
            == "eigs: device='gpu', but a block of dim 500 has no device kernel; use device='auto' or 'cpu'");

    FakeMachine::reset(true, std::nullopt);
    REQUIRE(message_of<ed::DeviceUnavailable>(Device::Gpu, req(Task::Eigs, 1u << 20))
            == "device='gpu', but the device's memory cannot be queried (no CUDA context could be created)");
    REQUIRE(FakeMachine::fresh_calls == 1);

    // 8 vectors of 16 B: 128 B per state.
    FakeMachine::reset(true, (std::size_t{128} << 20) - 1);
    REQUIRE(message_of<ed::ResourceLimit>(Device::Gpu, req(Task::Sampled, std::uint64_t{1} << 20))
            == "device='gpu', but a block of dim 1048576 needs 128 MiB of device memory and 127 MiB are free");
    FakeMachine::reset(true, std::size_t{128} << 20);
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
    FakeMachine::reset(true, (std::size_t{128} << 20) - 1);
    REQUIRE(ed::place(Device::Auto, req(Task::Eigs, std::uint64_t{1} << 20), FakeMachine::probe())
            == Lane::HostKrylov);
    REQUIRE(FakeMachine::fresh_calls == 0);   // the cached query
    FakeMachine::reset(true, std::nullopt);
    REQUIRE(ed::place(Device::Auto, req(Task::Sampled, std::uint64_t{1} << 20), FakeMachine::probe())
            == Lane::HostKrylov);
}

TEST_CASE("place: DenseBatch", "[place]") {
    FakeMachine::reset();
    REQUIRE(ed::place(Device::Cpu, req(Task::DenseBatch, 10), FakeMachine::probe()) == Lane::HostDense);
    REQUIRE(ed::place(Device::Auto, req(Task::DenseBatch, 10), FakeMachine::probe()) == Lane::DeviceDense);
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
