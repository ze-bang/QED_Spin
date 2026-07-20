"""Supplementary audit: cells the capability matrix does NOT sweep.

N=10 Heisenberg ring. Dense numpy eigh is ground truth for everything.
Each probe prints PASS/FAIL/MISSING with the measured deviation, and
never aborts the run -- exceptions are findings, not crashes.
"""
import sys, os, cmath, traceback
import numpy as np
import qed

N = 10
DIM = 1 << N
ETA = 0.1
Q = 0.5  # units of 2*pi
OMEGA = np.linspace(0.0, 4.0, 120)

def build_ring(n):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    return b.to_operator()

def build_probe(n):
    zb = qed.input.HamiltonianBuilder(n)
    q = 2.0 * np.pi * Q
    for i in range(n):
        zb.add_one_body(qed.input.Op.Sz, i, cmath.exp(1j * q * i) / np.sqrt(n))
    return zb.to_operator()

# dense truth ---------------------------------------------------------------
Hd = np.zeros((DIM, DIM))
def szv(s, i): return 0.5 if (s >> i) & 1 else -0.5
for s in range(DIM):
    for i in range(N):
        j = (i + 1) % N
        Hd[s, s] += szv(s, i) * szv(s, j)
        if ((s >> i) & 1) != ((s >> j) & 1):
            Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
w, V = np.linalg.eigh(Hd)
E0 = w[0]

Od = np.zeros((DIM, DIM), complex)
q = 2.0 * np.pi * Q
for s in range(DIM):
    Od[s, s] = sum(cmath.exp(1j * q * i) * szv(s, i) for i in range(N)) / np.sqrt(N)

def dense_dssf_T(T):
    """S(omega, T) = sum_nm p_n |<m|O|n>|^2 (eta/pi)/((w-(Em-En))^2+eta^2)."""
    if T is None or T == 0:
        p = np.zeros(DIM); p[0] = 1.0
    else:
        x = np.exp(-(w - w[0]) / T); p = x / x.sum()
    M = np.abs(V.conj().T @ Od @ V) ** 2      # |<m|O|n>|^2 at [n, m]
    S = np.zeros_like(OMEGA)
    keep = p > 1e-14
    for n_i in np.nonzero(keep)[0]:
        dE = w - w[n_i]
        for k_i, om in enumerate(OMEGA):
            S[k_i] += p[n_i] * np.sum(M[n_i] * (ETA / np.pi)
                                      / ((om - dE) ** 2 + ETA ** 2))
    return S

def thermo_E(T):
    x = np.exp(-(w - w[0]) / T)
    return float((w * x).sum() / x.sum())

H = build_ring(N)
probe = build_probe(N)
gen = qed.find_symmetries(H, verbose=False).full_set
results = []

def report(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name, detail))
    print(f"[{tag}] {name}: {detail}", flush=True)

def probe_cell(name, fn):
    try:
        fn()
    except Exception as e:
        results.append(("MISSING", name, f"{type(e).__name__}: {e}"))
        print(f"[MISSING] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

# ---------------------------------------------------------------- 1. little-group GPU
def cell_lg_gpu_gs():
    os.environ["ED_SYM_LG_GPU"] = "1"
    try:
        r = qed.solve(H, symmetry=gen, num_eigenvalues=1,
                      point_group="full", verbose=False)
        lane = getattr(getattr(r, "backend", None), "lane", "?")
        d = abs(float(r.eigenvalues[0]) - E0)
        report("GS little-group + ED_SYM_LG_GPU=1", d < 1e-8,
               f"dE0={d:.1e}, reported lane={lane!r}")
    finally:
        del os.environ["ED_SYM_LG_GPU"]
probe_cell("GS little-group GPU", cell_lg_gpu_gs)

def cell_lg_gpu_thermal():
    os.environ["ED_SYM_LG_GPU"] = "1"
    try:
        r = qed.thermal(H, method="FTLM", T_min=0.2, T_max=5.0, num_T=8,
                        symmetry=gen, point_group="full", verbose=False)
        temps = np.asarray(r.temperatures); E = np.asarray(r.energy)
        d = max(abs(E[i] - thermo_E(T)) for i, T in enumerate(temps))
        report("thermal little-group + ED_SYM_LG_GPU=1", d < 1e-8,
               f"max|dE(T)|={d:.1e}")
    finally:
        del os.environ["ED_SYM_LG_GPU"]
probe_cell("thermal little-group GPU", cell_lg_gpu_thermal)

def cell_full_spectrum_gpu():
    fs = qed.full_spectrum(H, symmetry=gen, device="gpu", verbose=False)
    a = np.sort(np.asarray(fs.eigenvalues))
    d = float(np.max(np.abs(a - w))) if len(a) == len(w) else float("inf")
    report("full_spectrum symmetry + device='gpu'", d < 1e-8,
           f"max|dE|={d:.1e} over {len(a)}/{DIM} values")
probe_cell("full_spectrum GPU", cell_full_spectrum_gpu)

def cell_full_spectrum_pg_gpu():
    fs = qed.full_spectrum(H, symmetry=gen, point_group="full",
                           device="gpu", verbose=False)
    a = np.sort(np.asarray(fs.eigenvalues))
    d = float(np.max(np.abs(a - w))) if len(a) == len(w) else float("inf")
    report("full_spectrum point_group='full' + device='gpu'", d < 1e-8,
           f"max|dE|={d:.1e} over {len(a)}/{DIM} values")
probe_cell("full_spectrum little-group GPU", cell_full_spectrum_pg_gpu)

# ---------------------------------------------------------------- 2. composed DSSF
S0 = dense_dssf_T(None)

def dssf_cell(name, **kw):
    r = qed.spectral(H, [probe], omega=OMEGA, eta=ETA, verbose=False, **kw)
    S = np.asarray(r.S_real).ravel()
    d = float(np.max(np.abs(S - S0)))
    report(name, d < 1e-8, f"max|dS|={d:.1e}")

for dev in ("cpu", "gpu"):
    probe_cell(f"DSSF flip require [{dev}]",
               lambda dev=dev: dssf_cell(
                   f"DSSF U(1)+spatial+flip='require' [{dev}]",
                   symmetry=gen, sz=N // 2, momentum_transfer=[Q],
                   spin_flip="require", device=dev))
    probe_cell(f"DSSF TR require [{dev}]",
               lambda dev=dev: dssf_cell(
                   f"DSSF U(1)+spatial+TR='require' [{dev}]",
                   symmetry=gen, sz=N // 2, momentum_transfer=[Q],
                   time_reversal="require", device=dev))
probe_cell("DSSF point_group full",
           lambda: dssf_cell("DSSF point_group='full' (little group)",
                             symmetry=gen, point_group="full"))

# ---------------------------------------------------------------- 3. finite-T DSSF
def cell_dssf_T(dev):
    T = 1.0
    ST = dense_dssf_T(T)
    r = qed.spectral(H, [probe], omega=OMEGA, eta=ETA, T=T,
                     symmetry=gen, momentum_transfer=[Q],
                     krylov_dim=200, num_random_vectors=40,
                     device=dev, verbose=False)
    S = np.asarray(r.S_real).ravel()
    if S.size != OMEGA.size:
        S = np.asarray(r.S_real)[..., :]
        S = S.reshape(-1, OMEGA.size)[0]
    num = float(np.trapezoid(S, OMEGA)); den = float(np.trapezoid(ST, OMEGA))
    peak_ratio = float(S.max() / ST.max())
    rel = float(np.max(np.abs(S - ST)) / ST.max())
    report(f"DSSF finite-T T=1.0 [{dev}]", rel < 0.10,
           f"max rel dev={rel:.2e}, integral ratio={num/den:.4f}, "
           f"peak ratio={peak_ratio:.4f}")
for dev in ("cpu", "gpu"):
    probe_cell(f"DSSF finite-T [{dev}]", lambda dev=dev: cell_dssf_T(dev))

# ---------------------------------------------------------------- 4. cTPQ
def cell_ctpq(dev):
    r = qed.thermal(H, method="cTPQ", T_min=0.5, T_max=5.0, num_T=8,
                    symmetry=gen, random_seed=7, device=dev, verbose=False)
    temps = np.asarray(getattr(r, "temperatures", None)
                       if getattr(r, "temperatures", None) is not None
                       else r.temperature)
    E = np.asarray(r.energy)
    rel = max(abs(E[i] - thermo_E(T)) / abs(thermo_E(T))
              for i, T in enumerate(temps))
    report(f"thermal cTPQ symmetry [{dev}]", rel < 0.05,
           f"max rel dev E(T)={rel:.2e}")
for dev in ("cpu", "gpu"):
    probe_cell(f"cTPQ [{dev}]", lambda dev=dev: cell_ctpq(dev))

# ---------------------------------------------------------------- summary
print("\n===== SUMMARY =====")
for tag, name, detail in results:
    print(f"{tag:8s} {name}: {detail}")
n_fail = sum(1 for t, _, _ in results if t != "PASS")
print(f"\n{len(results)} probes, {n_fail} not passing")
