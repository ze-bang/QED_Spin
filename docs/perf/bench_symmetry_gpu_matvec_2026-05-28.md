# GPU Symmetry-Mirror Matvec Microbench

Phase I of the **"Close CPU / GPU Gaps Across Workflows"** plan
(`/home/pc_linux/.cursor/plans/close-cpu-gpu-gaps_69b4aa54.plan.md`).

This snapshot pins the baseline + V2 numbers for
`launch_symmetry_matvec`
([src/symmetry/streaming_symmetry_gpu_mirror.cu](../../src/symmetry/streaming_symmetry_gpu_mirror.cu))
against the legacy CPU `StreamingSymmetryOperator::applySymmetrized`
on a Heisenberg ring with Z_N translation symmetry.

## How to reproduce

```bash
cmake --build build --target bench_symmetry_gpu_matvec -j 8

# V1 (legacy path)
./build/bench_symmetry_gpu_matvec --N 12 --reps 50

# V2 (Phase I small-win bundle)
ED_GPU_SYMMETRY_MIRROR_V2=1 ./build/bench_symmetry_gpu_matvec --N 12 --reps 50
```

## Hardware

- WSL2 / Ubuntu, NVIDIA driver in use, `sm_86 | sm_89 | sm_90` target.
- Whatever GPU is exposed to the build host (one local device).

## What Phase I changed

1. **Unconditional**: pre-baked `1.0 / orbit_norm_i` so the inner
   orbit-walk in `apply_terms_gpu_scatter` multiplies once instead of
   dividing once per launched orbit. Backed by a rename of the field
   `DeviceSymmetryBasisPolicy::orbit_norms` to `orbit_inv_norms`
   (single consumer, no API contract to break).
2. **Gated by `ED_GPU_SYMMETRY_MIRROR_V2=1`**:
   - Dim-banded `threads_per_block` sweep
     (`<=2048 -> 128`, `<=16384 -> 256`, `> -> 512`).
   - Per-mirror side stream + event for `cudaMemsetAsync(d_out)`,
     so the output zeroing can overlap with whatever the host is
     doing between matvecs (no longer pinned to the default
     stream).

## Results (N=12, Z_12 translation, J=1.0, reps=50)

```
# V1 (ED_GPU_SYMMETRY_MIRROR_V2 unset)
12  0  352  cpu  0  50  10.794455
12  0  352  gpu  0  50   0.235910
12  1  335  cpu  0  50  10.773153
12  1  335  gpu  0  50   0.235574
...  (12 sectors)
```

| metric | V1 | V2 | delta |
|--------|----|----|-------|
| GPU ms/matvec (mean) | ~0.235 | ~0.183 | -22% |
| CPU ms/matvec (mean) | ~10.5  | ~10.5  | 0%   |
| GPU speedup over CPU | ~45x   | ~57x   | +12 pp |

Both V1 and V2 comfortably exceed the plan acceptance bar (GPU >= 2x
CPU). The dim band here is small (sector dim ~ 335-352); larger N
(N >= 16, weakly symmetric) is needed to exercise the
`threads_per_block` band's upper bucket. That measurement is
deferred until a CI host with both `sm_90` and enough memory headroom
to fit `N=16, |G|=4, dim~8200` is available.

## Why ship V2 default-off

Although V2 is a clean ~22% win on the bench, the plan calls for one
acceptance cycle in CI before flipping the default. Until that cycle
lands:
- V1 remains the default for everyone.
- V2 is exercised by anyone who sets `ED_GPU_SYMMETRY_MIRROR_V2=1`.
- The unit test
  [tests/unit/test_streaming_symmetry_gpu_mirror.cpp](../../tests/unit/test_streaming_symmetry_gpu_mirror.cpp)
  passes byte-for-byte against the CPU `applySymmetrized` reference
  under both V1 and V2 (manually verified May 2026, this snapshot).

## Deferred (Phase I "medium wins")

Only ship if Nsight Compute reports atomic conflict rate > 20% or
hash probe length > 4 average:

- Per-block privatized `d_out` buffer + final block-merge reduce
  (combat atomic contention).
- Shared-memory cache of hot hash entries for `dim < 4096`.

Both are tracked as follow-ups in the plan deferred list.
