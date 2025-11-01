# Configuration Reference

This guide supplements the example files in `examples/` by describing the
configuration keys that control the exact diagonalization (ED) driver. The
settings are read from INI-style files (`key = value`) and can be overridden by
passing the equivalent command-line options. Unknown keys are ignored with a
warning so typos are easy to spot in the console output.【F:src/core/ed_config.cpp†L57-L207】

## File Structure

A configuration file is divided into named sections. The parser treats section
headers (`[Section]`) as comments, so the section names are for readability only.
Keys are case-sensitive and must appear exactly as listed below.【F:examples/ed_config_example.txt†L1-L132】

```
[Diagonalization]
method = LANCZOS
num_eigenvalues = 6
...
```

## Diagonalization

| Key | Description |
| --- | --- |
| `method` | Solver to use. Options include iterative Lanczos variants, Davidson/LOBPCG, full diagonalization (`FULL`, `OSS`), tensor-product quantum methods (`mTPQ`, `cTPQ`, `FTLM`, `LTLM`, `HYBRID`), ARPACK modes, and GPU accelerators (`*_GPU`).【F:src/core/ed_config.cpp†L69-L108】【F:src/core/ed_main.cpp†L520-L596】 |
| `num_eigenvalues` | Number of eigenvalues to compute. Use `FULL` to request the complete spectrum.【F:src/core/ed_config.cpp†L96-L98】 |
| `max_iterations` | Maximum iterations for iterative solvers.【F:src/core/ed_config.cpp†L99-L101】 |
| `tolerance` | Convergence tolerance for iterative solvers.【F:src/core/ed_config.cpp†L102-L103】 |
| `compute_eigenvectors` | Set to `true` to compute eigenvectors in addition to eigenvalues.【F:src/core/ed_config.cpp†L104-L105】 |
| `shift`, `block_size`, `max_subspace`, `target_lower`, `target_upper` | Advanced parameters used by shift-invert, block, Davidson, and Chebyshev-filtered workflows.【F:src/core/ed_config.h†L24-L45】 |

## System

| Key | Description |
| --- | --- |
| `num_sites` | Number of lattice sites. Defaults to the value inferred from the Hamiltonian directory when set to `0`.【F:examples/ed_config_example.txt†L25-L43】【F:src/core/ed_config.cpp†L109-L124】 |
| `spin_length` | Spin magnitude per site (e.g., `0.5` for spin-½).【F:src/core/ed_config.cpp†L109-L124】 |
| `sublattice_size` | Controls spin measurement grouping for observables.【F:examples/ed_config_example.txt†L31-L36】 |
| `hamiltonian_dir` | Directory containing the Hamiltonian description files (`InterAll.dat`, `Trans.dat`).【F:examples/ed_config_example.txt†L37-L44】 |
| `interaction_file`, `single_site_file` | Override default Hamiltonian filenames when needed.【F:src/core/ed_config.h†L150-L168】 |
| `use_fixed_sz` / `n_up` | Restrict the Hilbert space to a fixed `S_z` sector with a specified number of up spins.【F:src/core/ed_config.h†L138-L158】【F:src/core/ed_main.cpp†L26-L74】 |

## Workflow

| Key | Description |
| --- | --- |
| `output_dir` | Destination folder for results. The driver writes eigenvalues, thermodynamics, dynamical/static responses, and a copy of the resolved configuration to this directory.【F:src/core/ed_main.cpp†L26-L356】【F:src/core/ed_main.cpp†L433-L680】 |
| `run_standard` | Run the non-symmetrized diagonalization workflow.【F:src/core/ed_main.cpp†L24-L160】 |
| `run_symmetrized` | Run the symmetry-adapted workflow (requires symmetry data).【F:src/core/ed_main.cpp†L70-L120】 |
| `compute_thermo` | Derive thermodynamic observables from the eigenvalue spectrum.【F:src/core/ed_main.cpp†L122-L196】 |
| `compute_dynamical_response` | Enable dynamical response calculations (spectral functions).【F:src/core/ed_main.cpp†L196-L316】 |
| `compute_static_response` | Enable static response / thermal expectation value calculations.【F:src/core/ed_main.cpp†L316-L400】 |
| `skip_ed` | Skip diagonalization (useful when reprocessing existing spectra).【F:examples/ed_config_example.txt†L54-L62】 |

## Thermal

| Key | Description |
| --- | --- |
| `num_samples` | Number of random vectors for TPQ/FTLM/LTLM/hybrid methods.【F:src/core/ed_config.h†L53-L105】 |
| `temp_min`, `temp_max`, `temp_bins` | Temperature grid for thermodynamic averages and spectra. Use logarithmic spacing for dynamical response when multiple bins are requested.【F:src/core/ed_config.h†L53-L105】【F:src/core/ed_main.cpp†L226-L296】 |
| `num_order`, `num_measure_freq`, `delta_tau`, `large_value` | Imaginary-time evolution and stabilization controls for TPQ variants.【F:src/core/ed_config.h†L56-L98】 |
| `ftlm_krylov_dim`, `ftlm_full_reorth`, `ftlm_reorth_freq`, `ftlm_seed`, `ftlm_store_samples`, `ftlm_error_bars` | FTLM-specific settings.【F:src/core/ed_config.h†L73-L93】 |
| `ltlm_krylov_dim`, `ltlm_ground_krylov`, `ltlm_full_reorth`, `ltlm_reorth_freq`, `ltlm_seed`, `ltlm_store_data` | LTLM-specific knobs.【F:src/core/ed_config.h†L93-L113】 |
| `use_hybrid_method`, `hybrid_crossover`, `hybrid_auto_crossover` | Combine LTLM and FTLM by temperature. For new runs prefer `method = HYBRID` instead of the legacy boolean toggle.【F:src/core/ed_config.h†L113-L133】 |

## Observables

| Key | Description |
| --- | --- |
| `calculate` | When `true`, custom observables defined in `observables_*.dat` files are measured during diagonalization.【F:src/core/ed_config.h†L135-L168】 |
| `measure_spin` | Enable sublattice-resolved spin measurements.【F:src/core/ed_config.h†L135-L168】 |
| `omega_min`, `omega_max`, `num_points` | Frequency range for spectral functions and dynamical response.【F:src/core/ed_config.h†L140-L168】 |
| `t_end`, `dt` | Real-time evolution horizon and time step for quench dynamics.【F:src/core/ed_config.h†L140-L168】 |
| `operators` / `names` | Populated internally when operator files are loaded; no direct file syntax required.【F:src/core/ed_config.h†L140-L168】 |

## Dynamical Response

| Key | Description |
| --- | --- |
| `calculate` | Enable the dynamical response workflow (usually set indirectly by `compute_dynamical_response`).【F:src/core/ed_config.h†L170-L208】 |
| `thermal_average`, `num_random_states`, `krylov_dim` | Control random-state sampling and Krylov subspace sizes.【F:src/core/ed_config.h†L170-L208】 |
| `omega_min`, `omega_max`, `num_omega_points`, `broadening` | Define the frequency mesh and Lorentzian broadening.【F:src/core/ed_config.h†L170-L208】 |
| `temp_min`, `temp_max`, `num_temp_bins` | Temperature sweep for thermal spectra (logarithmically spaced in the driver).【F:src/core/ed_config.h†L170-L208】【F:src/core/ed_main.cpp†L226-L296】 |
| `compute_correlation`, `operator_file`, `operator2_file` | Choose between auto-correlation (`O†O`) and two-operator correlation (`O₁†O₂`) calculations by providing one or two operator files.【F:src/core/ed_config.h†L170-L208】【F:src/core/ed_main.cpp†L232-L312】 |
| `output_prefix`, `random_seed` | Customize output filenames and random seeding.【F:src/core/ed_config.h†L170-L208】 |

## Static Response

| Key | Description |
| --- | --- |
| `calculate` | Enable static response measurements (typically toggled through `compute_static_response`).【F:src/core/ed_config.h†L210-L244】 |
| `num_random_states`, `krylov_dim` | Control sampling for thermal averages.【F:src/core/ed_config.h†L210-L244】 |
| `temp_min`, `temp_max`, `num_temp_points` | Temperature grid for static observables.【F:src/core/ed_config.h†L210-L244】 |
| `compute_susceptibility` | Toggle derivative-based susceptibility output.【F:src/core/ed_config.h†L210-L244】 |
| `compute_correlation`, `single_operator_mode` | Choose between ⟨O†O⟩, ⟨O₁†O₂⟩, and single-operator expectation ⟨O⟩.【F:src/core/ed_config.h†L210-L244】【F:src/core/ed_main.cpp†L316-L388】 |
| `operator_file`, `operator2_file`, `output_prefix`, `random_seed` | File locations and naming/seed controls for the static workflow.【F:src/core/ed_config.h†L210-L244】【F:src/core/ed_main.cpp†L316-L388】 |

## Advanced ARPACK

The `[ARPACK]` section refines the multi-pass ARPACK solver used when
`method = ARPACK_ADVANCED`. Each key maps directly to the fields in the
`ArpackConfig` struct; most users can keep the defaults unless convergence is
problematic.【F:src/core/ed_config.h†L170-L208】【F:examples/ed_config_example.txt†L89-L132】 Relevant settings include:

- `verbose` – Print ARPACK progress information.
- `which` – Target spectrum slice (`SM`, `SR`, `LM`, `LR`, …).
- `ncv` – Subspace dimension (auto-selected when `-1`).
- `max_restarts`, `ncv_growth`, `auto_enlarge_ncv` – Adaptive restart controls.
- `two_phase_refine`, `relaxed_tol` – Dual-stage convergence strategy.
- `shift_invert`, `sigma`, `auto_switch_shift_invert`, `switch_sigma` – Shift-invert configuration.
- `adaptive_inner_tol`, `inner_tol_factor`, `inner_tol_min`, `inner_max_iter` – Inner iteration tolerances for shift-invert solvers.

## Command-Line Overrides

Every configuration key has a corresponding command-line flag (prefixed with
`--`). For example, `temp_max = 10.0` in the file can be overridden via
`--temp_max=5.0`. The `ED --help` output lists the available flags grouped by
category, and `ED --method-info=<METHOD>` prints solver-specific defaults.【F:src/core/ed_main.cpp†L433-L596】

When both a config file and command-line options are supplied, command-line
arguments take precedence. The resolved configuration is printed at runtime and
saved to `<output_dir>/ed_config.txt` for reproducibility.【F:src/core/ed_main.cpp†L433-L680】
