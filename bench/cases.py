"""Benchmark cases: each is one qed call on a size that engages the real kernels. The
resource string is the sbatch request of that case (walltime about twice the last run;
cases without a recorded run carry an estimate). Keyword arguments of a case are size
overrides (``run.py <case> --param N=16``); the defaults are the benchmark."""
from __future__ import annotations

import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
from grid.models import Model, chain, fourier, triangular  # noqa: E402

from qed import Symmetry, dynamics, eigs, expect, spectrum, thermal  # noqa: E402

import qed  # noqa: E402
import bench_models as bm  # noqa: E402



def _szq(m, q):
    return Model("obs", m.N, fourier(m.N, m.coords, m.shape, q, "z"), [], (), []).operator()


def _eigs_metrics(r):
    """The per-block phase record of an eigs result (EigResult.block_stats), summed over blocks:
    the engine-side numbers run.py records next to wall time and RSS."""
    bs = list(r.block_stats)
    applies = sum(b["applies"] for b in bs)
    apply_s = sum(b["apply_s"] for b in bs)
    nnz = sum(b["nnz"] for b in bs)
    csr_bytes = sum(b["csr_bytes"] for b in bs)
    return {"E0": float(r.energies[0]) if len(r.energies) else None,
            "blocks": len(bs), "max_dim": max((b["dim"] for b in bs), default=0),
            "lane": "+".join(sorted({b["lane"] for b in bs})),
            "applies": applies, "apply_s": round(apply_s, 4),
            "s_per_apply": apply_s / applies if applies else None,
            "build_s": round(sum(b["build_s"] for b in bs), 4),
            # the walk's abelian orbit table is shared by its blocks: counted once
            "orbit_s": round(max((b["context_orbit_s"] for b in bs), default=0.0)
                             + sum(b["star_orbit_s"] for b in bs), 4),
            "star_build_s": round(sum(b["star_build_s"] for b in bs), 4),
            "other_s": round(sum(b["other_s"] for b in bs), 4),
            "solve_s": round(sum(b["solve_s"] for b in bs), 4),
            "nnz": nnz, "bytes_per_nnz": csr_bytes / nnz if nnz else None,
            "device_blocks": int(r.device_blocks)}


def tri30_lg_eigs_cpu():
    m = triangular(6, Ly=5)
    r = eigs(m.operator(), 1, sym=Symmetry(spatial="auto", sz=15))
    return {"E0": float(r.energies[0])}


def chain32_abelian_eigs_gpu():
    m = chain(32)
    r = eigs(m.operator(), 1, sym=Symmetry(spatial=m.generator_set(), point_group=False, sz=16),
             device="gpu")
    return {"E0": float(r.energies[0]), "device_blocks": r.device_blocks}


def _ftlm(device):
    m = chain(28)
    r = thermal(m.operator(), np.linspace(0.1, 4.0, 20), method="ftlm",
                sym=Symmetry(spatial=m.generator_set(), point_group=False), samples=4, krylov=60,
                seed=7, device=device)
    return {"E(Tmin)": float(r.E[0]), "C_max": float(np.max(r.C)), "device_blocks": r.device_blocks}


def chain28_ftlm_cpu():
    return _ftlm("cpu")


def chain28_ftlm_gpu():
    return _ftlm("gpu")


def chain30_mtpq_gpu():
    m = chain(30)
    r = thermal(m.operator(), np.linspace(0.2, 4.0, 20), method="mtpq",
                sym=Symmetry(spatial=m.generator_set(), point_group=False), samples=1, seed=7,
                device="gpu")
    return {"E(Tmin)": float(r.E[0]), "device_blocks": r.device_blocks}


def chain30_dyn0_cpu():
    m = chain(30)
    w = np.linspace(-1, 7, 401)
    r = dynamics(m.operator(), _szq(m, (10,)), w, eta=0.05, krylov=200,
                 sym=Symmetry(spatial=m.generator_set(), point_group=False, sz=15))
    return {"weight": float(np.trapezoid(r.S[0], w))}


def chain24_dynT_cpu():
    m = chain(24)
    w = np.linspace(-3, 7, 301)
    r = dynamics(m.operator(), _szq(m, (8,)), w, eta=0.1, krylov=80, T=[1.0], samples=5,
                 sym=Symmetry(spatial=m.generator_set(), point_group=False))
    return {"weight": float(np.trapezoid(r.S[0], w))}


def _tri20_exact(device):
    m = triangular(5, Ly=4)
    r = thermal(m.operator(), np.linspace(0.1, 4.0, 20), method="exact", sym=Symmetry.auto(), device=device)
    return {"E(Tmin)": float(r.E[0]), "C_max": float(np.max(r.C))}


def tri20_lg_exact_thermal_cpu():
    return _tri20_exact("cpu")


def tri20_lg_exact_thermal_gpu():
    return _tri20_exact("auto")   # dense blocks from kDeviceDenseMinDim on the device, the rest in the host pool


# ---- tri36: 6x6 triangular J1 Heisenberg, n_up = 18, flip and TR off (the XDiag twin) -------
def _tri36(device, momentum=None, table=None):
    H, lat, spatial = bm.tri36()
    sym = Symmetry(spatial=spatial, sz=lat.N // 2, spin_flip="off", time_reversal="off")
    chars = None
    if table is not None:
        A, _ = sym.groups(H)
        chars = bm.char_table(bm.residue_namer(A, lat.point_group()), table)
    mom = None if momentum is None else {tuple(T): th for T, th in zip(lat.momentum_generators(), momentum)}
    r = eigs(H, 1, sym=bm.select_block(sym, H, momentum=mom, characters=chars), prune=False, device=device)
    return _eigs_metrics(r)


def tri36_G_A1_char_cpu():
    """Gamma A1 selected by character (character 1 on every residue); XDiag E0 -20.173442240303."""
    return _tri36("cpu")


def tri36_G_A1_char_gpu():
    return _tri36("gpu")


def tri36_G_E1_cpu():
    return _tri36("cpu", table=bm.C6V_E1)


def tri36_K_E_cpu():
    return _tri36("cpu", momentum=(Fraction(2, 3), Fraction(1, 3)), table=bm.C3V_E)


# ---- 36-site kagome (the BFG campaign's 36d torus) -------------------------------------------
def _bfg36_k3(device, jpm=-0.5):
    """Gamma A1 in both spin-flip halves, 3 levels per block."""
    H, _ = bm.bfg36(jpm)
    sym = Symmetry(sz=18)
    r = eigs(H, 3, sym=bm.select_block(sym, H), prune=False, device=device)
    return _eigs_metrics(r) | {"E": [float(e) for e in r.energies]}


def bfg36_k3_cpu(jpm=-0.5):
    return _bfg36_k3("cpu", jpm)


def bfg36_k3_gpu(jpm=-0.5):
    return _bfg36_k3("gpu", jpm)


def kagome36_expect_cpu():
    """Ground state of the kagome Heisenberg torus (Gamma A1) and two observables on it."""
    H, k = bm.kagome36_heisenberg()
    bonds = k.bonds(bm.KAGOME_NN)
    ops = [H, qed.input.HamiltonianBuilder(k.N).heisenberg(bonds[:1], 1.0).to_operator()]
    sym = Symmetry(sz=k.N // 2)
    res = expect(H, ops, 1, sym=bm.select_block(sym, H))
    v = res.values[0]
    return _eigs_metrics(res.eigs) | {"<H>": float(v[0].real), "<S.S>nn": float(v[1].real)}


# ---- chain30 k = 0 (the XDiag twin): NN ring, translations only, n_up = N/2 -----------------
def chain30_k0_cpu(N=30):
    H, gens = bm.heisenberg_ring(N)
    sym = Symmetry(spatial=gens, sz=N // 2, spin_flip="off", time_reversal="off")
    return _eigs_metrics(eigs(H, 1, sym=bm.select_block(sym, H), prune=False))


# ---- NLCE: every triangle-based cluster up to 16 sites, full spectrum -------------------------
def _nlce(device, max_order=8, max_sites=16):
    t0 = time.perf_counter()
    clusters = bm.nlce_triangle_clusters(max_order, max_sites)
    t_gen = time.perf_counter() - t0
    e_sum, dims = 0.0, 0
    for n, bonds in clusters:
        r = spectrum(bm.cluster_operator(n, bonds), sym=Symmetry.auto(), device=device)
        e_sum += float(r.energies[0])
        dims += 1 << n
    wall = time.perf_counter() - t0
    sizes = {}
    for n, _ in clusters:
        sizes[n] = sizes.get(n, 0) + 1
    return {"E0": e_sum, "clusters": len(clusters), "sites": sizes, "hilbert_total": dims,
            "generate_s": round(t_gen, 2), "clusters_per_hour": 3600.0 * len(clusters) / (wall - t_gen)}


def nlce_le16_exact_cpu(max_order=8, max_sites=16):
    return _nlce("cpu", max_order, max_sites)


def nlce_le16_exact_gpu(max_order=8, max_sites=16):
    return _nlce("gpu", max_order, max_sites)


# ---- tri24: 6x4 triangular torus, translations ----------------------------------------------
def _tri24():
    m = triangular(6, Ly=4)
    return m, Symmetry(spatial=m.generator_set(), point_group=False)


def tri24_ftlm_cpu():
    m, sym = _tri24()
    r = thermal(m.operator(), np.linspace(0.1, 4.0, 20), method="ftlm", sym=sym, samples=4, krylov=60,
                seed=7)
    return {"E(Tmin)": float(r.E[0]), "C_max": float(np.max(r.C)), "device_blocks": r.device_blocks}


def tri24_mtpq_gpu():
    m, sym = _tri24()
    r = thermal(m.operator(), np.linspace(0.2, 4.0, 20), method="mtpq", sym=sym, samples=1, seed=7,
                device="gpu")
    return {"E(Tmin)": float(r.E[0]), "device_blocks": r.device_blocks}


def tri24_dyn0_cpu():
    m = triangular(6, Ly=4)
    w = np.linspace(-1, 9, 401)
    r = dynamics(m.operator(), _szq(m, (3, 2)), w, eta=0.05, krylov=200,
                 sym=Symmetry(spatial=m.generator_set(), point_group=False, sz=12))
    return {"weight": float(np.trapezoid(r.S[0], w))}


def tri24_dynT_gpu():
    m, sym = _tri24()
    w = np.linspace(-3, 9, 301)
    r = dynamics(m.operator(), _szq(m, (3, 2)), w, eta=0.1, krylov=80, T=[1.0], samples=5, sym=sym,
                 device="gpu")
    return {"weight": float(np.trapezoid(r.S[0], w)), "device_blocks": r.device_blocks}


# ---- eigs(k > 1) matvecs against ARPACK (audit repro P3-krylov-01) ---------------------------
def krylov_vs_arpack(N=18):
    """H applies of qed.eigs for k = 1, 3, 6 on one Sz block without spatial symmetry, against
    ARPACK implicit restart (scipy eigsh, ncv = 2k + 60) on an independent sparse matrix."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    H, bonds = bm.random_xxz_chain(N)
    Hs = bm.sector_matrix(N, N // 2, bonds)
    D = Hs.shape[0]
    out = {}
    for k in (1, 3, 6):
        r = eigs(H, k, sym=Symmetry(spatial=None, sz=N // 2, spin_flip="off", time_reversal="off"),
                 prune=False, allow_partial=True)
        q = _eigs_metrics(r)
        count = [0]

        def mv(x):
            count[0] += 1
            return Hs @ x
        w = eigsh(LinearOperator((D, D), matvec=mv, dtype=float), k=k, which="SA", ncv=min(D - 1, 2 * k + 60),
                  tol=1e-10, v0=np.random.default_rng(7).standard_normal(D), return_eigenvectors=False)
        w = np.sort(w)
        out[f"k{k}"] = {"qed_applies": q["applies"], "arpack_matvecs": count[0],
                        "ratio": q["applies"] / max(count[0], 1),
                        "max_dE": float(np.max(np.abs(np.asarray(r.energies[:k]) - w[:len(r.energies[:k])]))),
                        "complete": bool(r.complete)}
        if k == 1:
            out["E0"] = float(r.energies[0])
    return out


CASES = {
    "tri30_lg_eigs_cpu":          (tri30_lg_eigs_cpu,          "-c 32 --mem=96G -t 0:15:00"),
    "chain32_abelian_eigs_gpu":   (chain32_abelian_eigs_gpu,   "-c 16 --mem=96G -t 1:00:00 --gpus-per-node=h100:1"),
    "chain28_ftlm_cpu":           (chain28_ftlm_cpu,           "-c 32 --mem=64G -t 0:30:00"),
    "chain28_ftlm_gpu":           (chain28_ftlm_gpu,           "-c 16 --mem=64G -t 0:30:00 --gpus-per-node=h100:1"),
    "chain30_mtpq_gpu":           (chain30_mtpq_gpu,           "-c 16 --mem=64G -t 4:00:00 --gpus-per-node=h100:1"),
    "chain30_dyn0_cpu":           (chain30_dyn0_cpu,           "-c 32 --mem=64G -t 0:30:00"),
    "chain24_dynT_cpu":           (chain24_dynT_cpu,           "-c 32 --mem=32G -t 1:30:00"),
    "tri20_lg_exact_thermal_cpu": (tri20_lg_exact_thermal_cpu, "-c 32 --mem=64G -t 1:00:00"),
    "tri20_lg_exact_thermal_gpu": (tri20_lg_exact_thermal_gpu, "-c 16 --mem=64G -t 0:30:00 --gpus-per-node=h100:1"),
    "tri36_G_A1_char_cpu":        (tri36_G_A1_char_cpu,        "-c 32 --mem=180G -t 3:00:00"),
    "tri36_G_A1_char_gpu":        (tri36_G_A1_char_gpu,        "-c 16 --mem=180G -t 3:00:00 --gpus-per-node=h100:1"),
    "tri36_G_E1_cpu":             (tri36_G_E1_cpu,             "-c 32 --mem=240G -t 4:00:00"),
    "tri36_K_E_cpu":              (tri36_K_E_cpu,              "-c 32 --mem=240G -t 4:00:00"),
    "bfg36_k3_cpu":               (bfg36_k3_cpu,               "-c 32 --mem=180G -t 2:30:00"),
    "bfg36_k3_gpu":               (bfg36_k3_gpu,               "-c 16 --mem=180G -t 2:30:00 --gpus-per-node=h100:1"),
    "kagome36_expect_cpu":        (kagome36_expect_cpu,        "-c 32 --mem=180G -t 2:30:00"),
    "chain30_k0_cpu":             (chain30_k0_cpu,             "-c 32 --mem=16G -t 0:15:00"),
    "nlce_le16_exact_cpu":        (nlce_le16_exact_cpu,        "-c 32 --mem=32G -t 2:00:00"),
    "nlce_le16_exact_gpu":        (nlce_le16_exact_gpu,        "-c 16 --mem=32G -t 2:00:00 --gpus-per-node=h100:1"),
    "tri24_ftlm_cpu":             (tri24_ftlm_cpu,             "-c 32 --mem=32G -t 0:30:00"),
    "tri24_mtpq_gpu":             (tri24_mtpq_gpu,             "-c 16 --mem=32G -t 1:00:00 --gpus-per-node=h100:1"),
    "tri24_dyn0_cpu":             (tri24_dyn0_cpu,             "-c 32 --mem=32G -t 0:30:00"),
    "tri24_dynT_gpu":             (tri24_dynT_gpu,             "-c 16 --mem=32G -t 1:00:00 --gpus-per-node=h100:1"),
    "krylov_vs_arpack":           (krylov_vs_arpack,           "-c 8 --mem=8G -t 0:15:00"),
}
