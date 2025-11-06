# `nlc_fit.py` User Manual

This guide explains how to build the C++ exact diagonalization executables and how to run the `util/nlc_fit.py` fitting pipeline that compares Numerical Linked Cluster Expansion (NLCE) calculations against experimental specific heat data.

## 1. Workflow Overview

1. Build the C++ ED executable (`ED`) that performs per-cluster diagonalizations.
2. Prepare experimental datasets (plain text) or a JSON configuration that references multiple datasets.
3. Run `util/nlc_fit.py`, which:
   - Generates (optionally) pyrochlore clusters once per working directory.
   - Calls `util/nlce.py` and the C++ binary to evaluate thermodynamics for candidate Hamiltonian parameters.
   - Compares calculated data to experiments using a configurable chi-squared metric.
   - Optimizes couplings and optional nuisance parameters using global or local solvers.
4. Inspect the log, plots, and parameter reports saved in the output directory.

## 2. Prerequisites

### 2.1 System Libraries

- GCC or Clang with C++17 support.
- CMake ≥ 3.18.
- BLAS and LAPACK (OpenBLAS, Intel oneMKL, Accelerate, etc.).
- LAPACKE development headers.
- ARPACK or ARPACK-NG.
- Optional: CUDA Toolkit (for GPU TPQ kernels) and MPI (for multi-node or TPQ workflows).

Ubuntu example:

```bash
sudo apt install g++ cmake libopenblas-dev liblapack-dev liblapacke-dev libarpack2-dev
# Optional
sudo apt install mpi-default-dev nvidia-cuda-toolkit
```

### 2.2 Python Environment

- Python 3.8 or newer.
- Required packages: `numpy`, `scipy`, `matplotlib`, `tqdm`, `networkx`, `pandas` (for some utilities), `scikit-learn` (for advanced sampling), and `jsonschema` (for config validation).
- Optional: `scikit-optimize` (`pip install scikit-optimize`) to unlock Bayesian optimizers.

Consider using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if available
```

## 3. Building the C++ ED Executable

1. Create a build directory and configure with CMake:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_MPI=ON \
  -DWITH_MKL=OFF
```

- Disable optional features by toggling `ON`/`OFF` as needed.
- Add `-DCMAKE_CUDA_ARCHITECTURES="80"` (for example) to target a specific GPU architecture.
- If using Intel oneMKL installed via oneAPI, ensure `MKLROOT` is exported before running CMake.

2. Compile the binaries:

```bash
cmake --build . --target ED --parallel
```

Additional targets:

- `TEST` – observables unit tests.
- `TPQ_DSSF` – time-dependent TPQ with MPI usage.
- `CUDA_EXAMPLE`, `CUDA_TEST` – GPU sample programs (when CUDA enabled).

3. (Optional) Install into a staging prefix:

```bash
cmake --install . --prefix /desired/install/path
```

4. Verify the executable:

```bash
./ED --help
```

Ensure the path used by `nlc_fit.py` (default `./build/ED`) exists or pass `--ed_executable` to point to your binary.

## 4. Input Data Preparation

### 4.1 Plain Text Format

- Columns: temperature (K) and specific heat (J/mol·K) separated by whitespace.
- Example (`specific_heat_Pr2Zr2O7.txt`):

```
# T(K)    C(T)
0.5       1.23
0.6       1.40
...
```

### 4.2 JSON Configuration for Multiple Datasets

Use a structure similar to `experimental_data_config.json`:

```json
{
  "experimental_data": [
    {"file": "specific_heat_Pr2Zr2O7.txt",  "h": 0.0, "field_dir": [0.577, 0.577, 0.577], "weight": 1.0},
    {"file": "specific_heat_Pr2Zr2O7_6T.txt", "h": 6.0, "field_dir": [0.577, 0.577, 0.577], "weight": 0.5}
  ],
  "global_params": {
    "temp_min": 1.0,
    "temp_max": 20.0,
    "max_order": 3
  }
}
```

- `field_dir` accepts unnormalized vectors; the script normalizes internally.
- Optional keys per dataset: `temp_min`, `temp_max`, custom `weight`.
- Global parameters overwrite command-line defaults for every dataset.

## 5. Running `nlc_fit.py`

Basic invocation (single dataset):

```bash
python3 util/nlc_fit.py \
  --exp_data specific_heat_Pr2Zr2O7.txt \
  --output_dir fit_results \
  --work_dir workdir_order3 \
  --ed_executable ./build/ED \
  --max_order 3 --temp_min 1.0 --temp_max 20.0
```

Multi-dataset mode:

```bash
python3 util/nlc_fit.py \
  --exp_config experimental_data_config.json \
  --output_dir fit_results_multi \
  --work_dir nlce_runs \
  --ed_executable ./build/ED \
  --fit_broadening --fit_g_renorm
```

### 5.1 Key Command-Line Options

- **Experimental input**:
  - `--exp_data` – Single file mode (default).
  - `--exp_config` – JSON configuration for multiple datasets.
- **Output control**:
  - `--output_dir` – Collects plots, logs, parameters (default `./fit_results`).
  - `--work_dir` – NLCE scratch directory; reuse to avoid regenerating clusters.
- **Hamiltonian parameters**:
  - `--initial_Jxx`, `--initial_Jyy`, `--initial_Jzz` – Starting point for optimization.
  - `--bound_min`, `--bound_max` – Shared bounds on `Jxx`, `Jyy`, `Jzz`.
  - Non-linear constraint: `0.125·Jzz ≥ Jxx`, `0.2·Jzz ≥ Jyy - 0.4·Jxx`, `Jyy + 0.4·Jxx + 0.2·Jzz ≥ 0` enforced via `NonlinearConstraint`.
- **Optional fit parameters**:
  - `--fit_broadening` – Optimize Gaussian width(s); default initial value from `--initial_sigma`.
  - `--fit_g_renorm` – Scale applied field (`h` → `g_renorm · h`).
  - `--fit_random_transverse_field` – Include random transverse field width. Uses multiple NLCE runs averaged over deterministic seeds when width > 0.
- **NLCE execution**:
  - `--max_order`, `--temp_min`, `--temp_max`, `--temp_bins` – Simulation grid.
  - `--skip_cluster_gen`, `--skip_ham_prep` – Reuse pre-generated assets.
  - `--measure_spin` – Switch NLCE to spin observable (instead of specific heat).
  - `--ED_method` – `FULL`, `OSS`, or `mTPQ` to match the executable capabilities.
  - `--h`, `--field_dir` – Default field strength/direction for single dataset mode.
- **Optimization strategy**:
  - `--method` – `auto` (default) cycles through multi-start, evolutionary, basin hopping, dual annealing, and Bayesian options (when available).
  - Alternative SciPy solvers: `SLSQP`, `L-BFGS-B`, `COBYLA`, `Nelder-Mead`, etc.
  - `--n_starts`, `--popsize`, `--n_calls`, `--n_initial_points`, `--acq_func` – Tunables for global/Bayesian optimizers.
  - `--plot_only` – Skip optimization and simply evaluate/plot the initial parameter guess.
- **Peak weighting**:
  - `--enable_peak_weighting` – Boost chi-squared contributions near the specific heat peak.
  - `--peak_width_factor`, `--peak_weight_factor` – Define weighting window and magnitude.
- **Random field averaging**:
  - `--random_field_n_runs` – Number of deterministic seeds to average when fitting a random transverse field.
- **Reproducibility**:
  - `--disorder_seed_base` – Base seed used when generating deterministic random fields for each run.

### 5.2 Gaussian Broadening

When `--fit_broadening` is set, the script applies `scipy.ndimage.gaussian_filter1d` to the calculated curve. For the first dataset the width is taken directly, while subsequent datasets use halved widths internally to maintain stability (matching the code’s `sigmas[1:-1] = sigmas[1:-1] / 2`).

### 5.3 Random Transverse Field Fits

Enable `--fit_random_transverse_field` to introduce an additional width parameter that toggles multiple NLCE evaluations per dataset. Average results are computed over `random_field_n_runs` seeds. Expect longer run times; increase `--num_workers` to parallelize.

### 5.4 Symmetry and Field Directions

- The pipeline can symmetrize single-direction field calculations via `--symmetrized` when random transverse fields are not being fitted.
- For `mTPQ` runs, the `--method=mTPQ` flag is added automatically.

## 6. Outputs

Running the script populates `--output_dir` (default `fit_results`):

- `nlc_fit.log` – Detailed execution log, command echoes, chi-squared per dataset.
- `dataset_{i}_comparison.png` – Three-panel plot (data vs fit, residuals, peak detail or relative residuals).
- `specific_heat_fit.png` – Combined summary across all datasets.
- `bayesian_optimization_convergence.png` – Only when Bayesian methods are active.
- `best_parameters.txt` – Human-readable summary including final chi-squared.
- `optimization_results.json` – Machine-readable record with metadata, parameter history (for Bayesian runs), and optional nuisance parameters.

Scratch artifacts reside in `--work_dir`:

- `clusters_order_{N}` – Pre-generated cluster libraries.
- `run_*` subdirectories when random field averaging is enabled.
- `nlc_results_order_{N}` – NLCE thermodynamic outputs used for interpolation.

## 7. Example Workflows

### 7.1 Quick Look Without Optimization

```bash
python3 util/nlc_fit.py \
  --exp_data specific_heat_Pr2Zr2O7.txt \
  --plot_only --max_order 2 \
  --initial_Jxx 0.2 --initial_Jyy 0.2 --initial_Jzz 1.0
```

Generates plots using the initial guess and reports the baseline chi-squared.

### 7.2 Multi-Field Fit With Peak Weighting

```bash
python3 util/nlc_fit.py \
  --exp_config experimental_data_config.json \
  --fit_broadening --fit_g_renorm \
  --enable_peak_weighting --peak_width_factor 1.5 --peak_weight_factor 4.0 \
  --method auto --max_iter 2000
```

Uses multiple global optimizers, includes Gaussian widths, and prioritizes peak alignment.

### 7.3 Random Transverse Field Search

```bash
python3 util/nlc_fit.py \
  --exp_data specific_heat_Pr2Zr2O7.txt \
  --fit_random_transverse_field --random_field_n_runs 8 \
  --num_workers 8 --method differential_evolution
```

Explores transverse disorder strength using evolutionary optimization with averaged NLCE evaluations.

## 8. Troubleshooting & Tips

- **ED executable missing**: Build `ED` and update `--ed_executable`. Ensure it is executable.
- **Cluster regeneration every run**: Provide a persistent `--work_dir` and pass `--skip_cluster_gen` once clusters exist.
- **Chi-squared NaNs or zero-length arrays**: Inspect logs for NLCE failures; tweak `--temp_min`, `--temp_max`, or reduce `max_order`.
- **Bayesian optimizer unavailable**: Install `scikit-optimize` or choose a different `--method`.
- **Long runtimes**: Reduce `--temp_bins`, lower `max_order`, or start with `--plot_only` to gauge scale. Use `--num_workers` to parallelize random-field averaging.
- **Field direction symmetrization**: For high-symmetry fields ([111], [100]), the script can explore equivalent directions; keep an eye on log output (“Symmetry-equivalent field directions”).

## 9. Cleaning Up

Temporary directories created via `tempfile.mkdtemp` are removed automatically when using the default `--work_dir`. When supplying a custom working directory, delete it manually if space becomes an issue.

---

With the C++ executables compiled and inputs prepared, `nlc_fit.py` provides a reproducible bridge between experimental thermodynamics and NLCE-based microscopic models. Adjust the optimization options according to the complexity of your dataset and the time budget available.
