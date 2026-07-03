# `examples/` — the tour

Six short, heavily-commented, runnable scripts cover every real knob
the library exposes. Each runs standalone in seconds:

```bash
python3 examples/tour/01_ground_state.py
```

| script | verb | what it covers |
|---|---|---|
| [`tour/01_ground_state.py`](tour/01_ground_state.py) | `qed.solve` | `symmetry="auto"`, per-symmetry toggles (`spin_flip=` / `time_reversal=` / `point_group=` with auto/on/off/require), solvers, devices, per-sector attribution, eigenvectors |
| [`tour/02_finite_temperature.py`](tour/02_finite_temperature.py) | `qed.thermal` | mTPQ/FTLM/LTLM/KPM, the flat sector pool + flip/TR/star copies, Sz windows, seeds, devices |
| [`tour/03_dynamics_dssf.py`](tour/03_dynamics_dssf.py) | `qed.spectral` | momentum-resolved probes (S^z_Q / S^±_Q), ground-state + finite-T DSSF through the sector machinery, selection rules, broadening/Krylov knobs |
| [`tour/04_symmetry_toolkit.py`](tour/04_symmetry_toolkit.py) | — | `find_symmetries`, `GeneratorSet.describe()`, explicit generator sets, sector selection, the four-state toggles, env escapes |
| [`tour/05_tpq_dssf.py`](tour/05_tpq_dssf.py) | thermal → spectral | finite-temperature DSSF from mTPQ states: persist a snapshot at each β, seed the continued fraction with the warm state (`initial_state=`) |

The tour is CI-guarded: the `linux-tour` lane builds the wheel and
runs every script end-to-end on each push.

For exhaustive per-configuration coverage — every
(backend × symmetry × method) combination — the test suites are the
reference: `python/tests/` + `tests/integration/` (Python) and
`tests/unit/` (C++), plus the verified capability matrix in
`docs/perf/capability_matrix_2026-07-03.md` /
`benchmarks/bench_capability_matrix.py`.
