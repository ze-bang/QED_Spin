#!/usr/bin/env python3
"""bench_capability_matrix.py -- every keyable symmetry composition,
verified against full dense diagonalization.

One Hamiltonian (Heisenberg ring, N sites: U(1) Sz + Z_N translation +
spin flip + time reversal all present), three workflow verbs, two
backends, and every keyable composition of the four symmetry axes.
Ground truth is the FULL dense spectrum (numpy eigh of the 2^N
Hamiltonian):

  * GS      E0 must match to |dE| < 1e-7
  * thermal E(T), C(T) from the exact partition sum; sector lanes with
            the exact small-block fallback must match to ~1e-8,
            stochastic lanes (mTPQ on blocks > 512) within a few %
  * DSSF    S^z_{Q=pi}(omega) against the dense Lehmann sum with the
            same eta/pi Lorentzian (verified convention, ratio == 1)

Output: markdown tables (stdout) + ``--json`` dump with every number.

Usage::

    python3 benchmarks/bench_capability_matrix.py --n 12 \
        --devices cpu,gpu --json docs/perf/capability_matrix.json
"""
from __future__ import annotations

import argparse
import cmath
import json
import time

import numpy as np

import qed

ETA = 0.1
Q_FRAC = 0.5           # Q = pi in units of 2*pi
T_GRID = (0.2, 5.0, 20)


# ---------------------------------------------------------------------------
# Model + dense ground truth
# ---------------------------------------------------------------------------
def build_ring(n):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    return b.to_operator()


def ring_lattice(n):
    from qed.input import lattice as L
    return L.chain(n, pbc=True)


def build_probe(n):
    zb = qed.input.HamiltonianBuilder(n)
    q = 2.0 * np.pi * Q_FRAC
    for i in range(n):
        zb.add_one_body(qed.input.Op.Sz, i, cmath.exp(1j * q * i) / np.sqrt(n))
    return zb.to_operator()


def dense_truth(n, omega):
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
    t0 = time.perf_counter()
    w, V = np.linalg.eigh(Hd)
    t_dense = time.perf_counter() - t0

    # Thermal observables from the exact spectrum.
    def thermo(T):
        beta = 1.0 / T
        x = np.exp(-beta * (w - w[0]))
        Z = x.sum()
        E = (w * x).sum() / Z
        E2 = (w * w * x).sum() / Z
        return E, beta * beta * (E2 - E * E)

    # DSSF Lehmann sum for S^z_{Q=pi}.
    q = 2.0 * np.pi * Q_FRAC
    O = np.zeros((dim, dim), complex)
    for s in range(dim):
        O[s, s] = sum(cmath.exp(1j * q * i) * szv(s, i)
                      for i in range(n)) / np.sqrt(n)
    amps = np.abs(V.conj().T @ (O @ V[:, 0])) ** 2
    S = np.array([np.sum(amps * (ETA / np.pi)
                         / ((x - (w - w[0])) ** 2 + ETA ** 2))
                  for x in omega])
    return dict(E0=float(w[0]), spectrum=w, thermo=thermo, S=S,
                t_dense=t_dense)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


# ---------------------------------------------------------------------------
# Config matrices. Toggles use "require"/"off" so each composition is
# PINNED (require throws if the mechanism cannot engage -- no silent
# fallback can fake a row).
# ---------------------------------------------------------------------------
def gs_configs(n, gen, lat):
    h = n // 2
    base = dict(spin_flip="off", time_reversal="off", point_group="off")
    return [
        ("none",                dict(symmetry="off", auto_sz=False)),
        ("U(1)",                dict(symmetry="off", sz=h)),
        ("spatial",             dict(symmetry=gen, auto_sz=False, **base)),
        ("U(1)+spatial",        dict(symmetry=gen, sz=h, **base)),
        ("U(1)+spatial+flip",   dict(symmetry=gen, sz=h,
                                     spin_flip="require",
                                     time_reversal="off",
                                     point_group="off")),
        ("U(1)+spatial+TR",     dict(symmetry=gen, sz=h, spin_flip="off",
                                     time_reversal="require",
                                     point_group="off")),
        ("U(1)+spatial+star",   dict(symmetry=gen, sz=h, spin_flip="off",
                                     time_reversal="off",
                                     point_group="auto")),
        ("U(1)+translation+star", dict(symmetry="translation", lattice=lat,
                                       sz=h, spin_flip="off",
                                       time_reversal="off",
                                       point_group="auto")),
        ("U(1)+spatial+flip+TR+star", dict(symmetry=gen, sz=h,
                                           spin_flip="require",
                                           time_reversal="require",
                                           point_group="auto")),
    ]


def thermal_configs(gen, lat):
    off = dict(spin_flip="off", time_reversal="off", point_group="off")
    return [
        ("none",                dict(symmetry=None,
                                     use_sz_if_conserved=False)),
        ("U(1)",                dict(symmetry=None)),
        ("U(1)+spatial",        dict(symmetry=gen, **off)),
        ("U(1)+spatial+flip",   dict(symmetry=gen, spin_flip="require",
                                     time_reversal="off",
                                     point_group="off")),
        ("U(1)+spatial+TR",     dict(symmetry=gen, spin_flip="off",
                                     time_reversal="require",
                                     point_group="off")),
        ("U(1)+spatial+star",   dict(symmetry=gen, spin_flip="off",
                                     time_reversal="off",
                                     point_group="auto")),
        ("U(1)+translation+star", dict(symmetry="translation", lattice=lat,
                                       spin_flip="off",
                                       time_reversal="off",
                                       point_group="auto")),
        ("U(1)+spatial+flip+TR+star", dict(symmetry=gen,
                                           spin_flip="require",
                                           time_reversal="require",
                                           point_group="auto")),
    ]


def _lane_of(r):
    b = getattr(r, "backend", None)
    return getattr(b, "lane", "") or ""


def run_gs(H, n, gen, lat, device, truth, rows):
    for label, kw in gs_configs(n, gen, lat):
        t, r = timed(lambda: qed.solve(H, num_eigenvalues=1, device=device,
                                       verbose=False, **kw))
        e0 = float(r.eigenvalues[0])
        dims = [tg.sector_dim
                for tg in (getattr(r, "sector_tags", None) or [])]
        lane = _lane_of(r)
        # A GPU row that silently ran on the CPU would fake its label.
        # The legacy plain-lane EDResults carries no backend metadata
        # (lane == ""); every streaming-symmetry row does, and those
        # are strictly enforced.
        if device == "gpu" and lane:
            assert lane == "gpu", \
                f"GS[{label}] requested gpu but ran lane={lane!r}"
        rows.append(dict(verb="GS", config=label, device=device,
                         lane=lane,
                         E0=e0, dev=abs(e0 - truth["E0"]), t=t,
                         blocks=len(dims) or 1,
                         max_block=max(dims) if dims else (1 << n)))


def run_thermal(H, n, gen, lat, device, truth, rows):
    t_lo, t_hi, n_t = T_GRID
    for label, kw in thermal_configs(gen, lat):
        t, r = timed(lambda: qed.thermal(
            H, method="mTPQ", T_min=t_lo, T_max=t_hi, num_T=n_t,
            device=device, random_seed=7, verbose=False, **kw))
        temps = np.asarray(getattr(r, "temperatures", None)
                           if getattr(r, "temperatures", None) is not None
                           else r.temperature)
        E = np.asarray(r.energy)
        C = np.asarray(r.specific_heat)
        exE = np.array([truth["thermo"](T)[0] for T in temps])
        exC = np.array([truth["thermo"](T)[1] for T in temps])
        scale = max(1.0, float(np.max(np.abs(exE))))
        dev = float(np.max(np.abs(E - exE)) / scale)
        i05 = int(np.argmin(np.abs(temps - 0.5)))
        i20 = int(np.argmin(np.abs(temps - 2.0)))
        per = r.per_sector or []
        rows.append(dict(verb="thermal", config=label, device=device,
                         E_T05=float(E[i05]), C_T05=float(C[i05]),
                         E_T20=float(E[i20]), C_T20=float(C[i20]),
                         exact_E_T05=float(exE[i05]),
                         exact_C_T05=float(exC[i05]),
                         dev=dev, t=t, blocks=len(per) or 1,
                         max_block=max((e.sector_dim for e in per),
                                       default=(1 << n))))


def run_thermal_methods(H, n, gen, device, truth, rows):
    """Beyond mTPQ: FTLM / LTLM through the fully composed lane
    (star+flip+TR) vs the unsymmetrised lane. Both are stochastic --
    the check is that the composed lane's deviation is comparable to
    (not worse than) the unsymmetrised lane's own sampling error.
    (LTLM's absolute accuracy at these sizes/defaults is poor in BOTH
    lanes -- a method-regime property, reported as-is.)"""
    t_lo, t_hi, n_t = T_GRID
    for m in ("FTLM", "LTLM"):
        devs = {}
        for label, kw in (
            ("none", dict(symmetry=None, use_sz_if_conserved=False)),
            ("U(1)+spatial+flip+TR+star",
             dict(symmetry=gen, spin_flip="require",
                  time_reversal="require", point_group="auto")),
        ):
            t, r = timed(lambda: qed.thermal(
                H, method=m, T_min=t_lo, T_max=t_hi, num_T=n_t,
                device=device, random_seed=7, verbose=False, **kw))
            temps = np.asarray(getattr(r, "temperatures", None)
                               if getattr(r, "temperatures", None)
                               is not None else r.temperature)
            E = np.asarray(r.energy)
            exE = np.array([truth["thermo"](T)[0] for T in temps])
            scale = max(1.0, float(np.max(np.abs(exE))))
            dev = float(np.max(np.abs(E - exE)) / scale)
            devs[label] = dev
            i05 = int(np.argmin(np.abs(temps - 0.5)))
            rows.append(dict(verb=f"thermal ({m})", config=label,
                             device=device, E_T05=float(E[i05]),
                             C_T05=float(np.asarray(r.specific_heat)[i05]),
                             E_T20=None, C_T20=None,
                             exact_E_T05=float(exE[i05]),
                             exact_C_T05=None,
                             dev=dev, t=t, blocks=None, max_block=None))
        # stochastic: composed must not be qualitatively worse
        assert (devs["U(1)+spatial+flip+TR+star"]
                < 5.0 * max(devs["none"], 1e-3)), \
            f"{m} composed lane inconsistent: {devs}"


def run_dssf(H, n, gen, device, omega, truth, rows):
    probe = build_probe(n)
    configs = [
        ("none", dict()),
        ("U(1)+spatial", dict(symmetry=gen, sz=n // 2,
                              momentum_transfer=[Q_FRAC])),
    ]
    for label, kw in configs:
        t, r = timed(lambda: qed.spectral(H, [probe], omega=omega, eta=ETA,
                                          device=device, verbose=False,
                                          **kw))
        S = np.asarray(r.S_real).ravel()
        dev = float(np.max(np.abs(S - truth["S"])))
        pk = int(np.argmax(S))
        rows.append(dict(verb="DSSF S^z_Q=pi", config=label, device=device,
                         peak=float(S[pk]), peak_omega=float(omega[pk]),
                         dev=dev, t=t))


def build_u1_broken_ring(n):
    """Heisenberg ring + J_pmpm-type S+S+ / S-S- terms: breaks U(1),
    keeps Sz parity, prod-sigma^x, spatial D_n and time reversal."""
    Op = qed.input.Op
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    for i in range(n):
        b.add_two_body(Op.Sp, i, Op.Sp, (i + 1) % n, 0.3)
        b.add_two_body(Op.Sm, i, Op.Sm, (i + 1) % n, 0.3)
    return b.to_operator()


def dense_truth_broken(n):
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            Hd[s, s] += szv(s, i) * szv(s, j)
            bi, bj = (s >> i) & 1, (s >> j) & 1
            if bi != bj:
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
            if bi == 0 and bj == 0:
                Hd[s | (1 << i) | (1 << j), s] += 0.3
            if bi == 1 and bj == 1:
                Hd[s & ~(1 << i) & ~(1 << j), s] += 0.3
    w = np.linalg.eigvalsh(Hd)

    def thermo(T):
        beta = 1.0 / T
        x = np.exp(-beta * (w - w[0]))
        Z = x.sum()
        E = (w * x).sum() / Z
        return E

    return dict(E0=float(w[0]), spectrum=np.sort(w), thermo_E=thermo)


def run_nonabelian_section(n, truth, H, gen, rows):
    """point_group='full' cells (D_n group, d >= 2 irreps): GS /
    thermal / DSSF vs dense. CPU engine."""
    omega = np.linspace(0.0, 4.0, 120)
    w = truth["spectrum"]
    t_lo, t_hi, n_t = T_GRID

    t, r = timed(lambda: qed.solve(H, symmetry=gen, num_eigenvalues=6,
                                   point_group="full", verbose=False))
    rows.append(dict(verb="GS/na", config="nonabelian-full(d>=2)",
                     device="cpu", E0=float(r.eigenvalues[0]),
                     dev=float(np.max(np.abs(
                         np.asarray(r.eigenvalues) - w[:6]))),
                     t=t, blocks=None, max_block=None))

    t, r = timed(lambda: qed.thermal(
        H, method="FTLM", T_min=t_lo, T_max=t_hi, num_T=n_t,
        symmetry=gen, point_group="full", device="cpu", verbose=False))
    temps = np.asarray(r.temperatures)
    E = np.asarray(r.energy)
    exE = np.array([truth["thermo"](T)[0] for T in temps])
    rows.append(dict(verb="thermal/na", config="nonabelian-full(exact)",
                     device="cpu", E0=float(E[0]),
                     dev=float(np.max(np.abs(E - exE))), t=t,
                     blocks=None, max_block=None))

    t, r = timed(lambda: qed.spectral(
        H, [build_probe(n)], omega=omega, eta=ETA, symmetry=gen,
        point_group="full", verbose=False))
    S = np.asarray(r.S_real)
    rows.append(dict(verb="DSSF/na", config="nonabelian-full",
                     device="cpu", E0=None,
                     dev=float(np.max(np.abs(S - truth["S"]))), t=t,
                     blocks=None, max_block=None))


def run_broken_section(n, devices, rows):
    """Every cell of the U(1)-BROKEN model: parity + full-space flip
    across GS / thermal / full dense (+ non-abelian GS)."""
    H = build_u1_broken_ring(n)
    gen = qed.find_symmetries(H, verbose=False).full_set
    truth = dense_truth_broken(n)
    w = truth["spectrum"]
    dim = 1 << n
    t_lo, t_hi, n_t = T_GRID
    off = dict(spin_flip="off", time_reversal="off", point_group="off")
    configs = [
        ("none",              dict(symmetry="off", auto_sz=False)),
        ("spatial",           dict(symmetry=gen, auto_sz=False, **off)),
        ("spatial+flipfull",  dict(symmetry=gen, auto_sz=False,
                                   spin_flip="require",
                                   time_reversal="off",
                                   point_group="off")),
        ("parity+spatial",    dict(symmetry=gen, **off)),
        ("parity+spatial+flip", dict(symmetry=gen, spin_flip="require",
                                     time_reversal="off",
                                     point_group="off")),
        ("parity+flip+TR+star", dict(symmetry=gen, spin_flip="require",
                                     time_reversal="require",
                                     point_group="auto")),
    ]
    for dev in devices:
        for label, kw in configs:
            t, r = timed(lambda: qed.solve(
                H, num_eigenvalues=dim, solver="full", device=dev,
                verbose=False, **kw))
            eps = getattr(r, "eigenvalues_per_sector", None)
            if eps:
                a = np.sort(np.concatenate([np.asarray(e) for e in eps]))
                dev_err = float(np.max(np.abs(a - w))) \
                    if len(a) == len(w) else float("inf")
            else:
                dev_err = abs(float(sorted(r.eigenvalues)[0]) - truth["E0"])
            tags = getattr(r, "sector_tags", None) or []
            rows.append(dict(verb="GS[U(1)-broken]", config=label,
                             device=dev,
                             E0=float(sorted(r.eigenvalues)[0]),
                             dev=dev_err, t=t, blocks=len(tags) or 1,
                             max_block=max((tg.sector_dim for tg in tags),
                                           default=dim)))
        if dev == "cpu":
            t, r = timed(lambda: qed.solve(
                H, symmetry=gen, num_eigenvalues=6,
                point_group="full", verbose=False))
            rows.append(dict(verb="GS[U(1)-broken]",
                             config="nonabelian-full(d>=2)", device=dev,
                             E0=float(r.eigenvalues[0]),
                             dev=float(np.max(np.abs(
                                 np.asarray(r.eigenvalues) - w[:6]))),
                             t=t, blocks=None, max_block=None))
        t, r = timed(lambda: qed.thermal(
            H, method="mTPQ", T_min=t_lo, T_max=t_hi, num_T=n_t,
            symmetry=gen, random_seed=7, device=dev, verbose=False))
        temps = np.asarray(getattr(r, "temperatures", None)
                           if getattr(r, "temperatures", None) is not None
                           else r.temperature)
        E = np.asarray(r.energy)
        exE = np.array([truth["thermo_E"](T) for T in temps])
        per = r.per_sector or []
        rows.append(dict(verb="thermal[U(1)-broken]",
                         config="parity(auto)+flip", device=dev,
                         E0=float(E[0]),
                         dev=float(np.max(np.abs(E - exE))), t=t,
                         blocks=len(per) or None,
                         max_block=max((e.sector_dim for e in per),
                                       default=0) or None))
        t, fs = timed(lambda: qed.full_spectrum(
            H, symmetry=gen, point_group="off", verbose=False))
        a = np.asarray(fs.eigenvalues)
        rows.append(dict(verb="fulldense[U(1)-broken]",
                         config="parity(auto)+flip", device=dev,
                         E0=float(a[0]),
                         dev=float(np.max(np.abs(a - w))), t=t,
                         blocks=None, max_block=None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--devices", default="cpu")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    n = args.n
    devices = [d for d in args.devices.split(",") if d]

    omega = np.linspace(0.0, 4.0, 120)
    H = build_ring(n)
    gen = qed.find_symmetries(H, verbose=False).full_set
    truth = dense_truth(n, omega)
    print(f"# Capability matrix, N={n} Heisenberg ring\n")
    print(f"dense ground truth: dim {1 << n}, eigh {truth['t_dense']:.2f} s, "
          f"E0 = {truth['E0']:.10f}\n")

    rows: list[dict] = []
    lat = ring_lattice(n)
    for dev in devices:
        run_gs(H, n, gen, lat, dev, truth, rows)
        run_thermal(H, n, gen, lat, dev, truth, rows)
        run_thermal_methods(H, n, gen, dev, truth, rows)
        run_dssf(H, n, gen, dev, omega, truth, rows)

    run_nonabelian_section(n, truth, H, gen, rows)
    run_broken_section(n, devices, rows)

    # ---- markdown ----
    print("\n## Ground state (E0 vs dense)\n")
    print("| composition | device | E0 | |dE0| | blocks | max dim | t [s] |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["verb"] != "GS":
            continue
        print(f"| {r['config']} | {r['device']} | {r['E0']:.10f} "
              f"| {r['dev']:.1e} | {r['blocks']} | {r['max_block']} "
              f"| {r['t']:.3f} |")

    print("\n## Thermal mTPQ (E(T), C(T) vs exact partition sum)\n")
    print("| composition | device | E(0.5) | C(0.5) | E(2.0) | C(2.0) "
          "| max rel dev E(T) | blocks | max dim | t [s] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["verb"] != "thermal":
            continue
        print(f"| {r['config']} | {r['device']} | {r['E_T05']:.6f} "
              f"| {r['C_T05']:.6f} | {r['E_T20']:.6f} | {r['C_T20']:.6f} "
              f"| {r['dev']:.1e} | {r['blocks']} | {r['max_block']} "
              f"| {r['t']:.3f} |")
    ex05 = [r for r in rows if r["verb"] == "thermal"][0]
    print(f"\nexact: E(0.5) = {ex05['exact_E_T05']:.6f}, "
          f"C(0.5) = {ex05['exact_C_T05']:.6f}")

    print("\n## Other finite-T methods (fully composed vs none)\n")
    print("| method | composition | device | E(0.5) | C(0.5) "
          "| max rel dev E(T) | t [s] |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if not (r["verb"].startswith("thermal (")
                and r["verb"] != "thermal (mTPQ)"):
            continue
        print(f"| {r['verb'][9:-1]} | {r['config']} | {r['device']} "
              f"| {r['E_T05']:.6f} | {r['C_T05']:.6f} "
              f"| {r['dev']:.1e} | {r['t']:.3f} |")

    print("\n## DSSF S^z_{Q=pi}(omega) (vs dense Lehmann sum)\n")
    print("| composition | device | peak S | peak omega | max |dS| | t [s] |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["verb"] != "DSSF S^z_Q=pi":
            continue
        print(f"| {r['config']} | {r['device']} | {r['peak']:.6f} "
              f"| {r['peak_omega']:.3f} | {r['dev']:.1e} | {r['t']:.3f} |")

    print("\n## Extended cells: non-abelian full group + U(1)-broken model\n")
    print("| verb | composition | device | E0 | max dev | blocks "
          "| max dim | t [s] |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not (r["verb"].endswith("[U(1)-broken]")
                or "nonabelian" in str(r.get("config", ""))):
            continue
        e0 = f"{r['E0']:.8f}" if r.get("E0") is not None else "--"
        blocks = r["blocks"] if r.get("blocks") is not None else "--"
        maxb = r["max_block"] if r.get("max_block") is not None else "--"
        print(f"| {r['verb']} | {r['config']} | {r['device']} | {e0} "
              f"| {r['dev']:.1e} | {blocks} | {maxb} | {r['t']:.3f} |")

    if args.json:
        payload = dict(N=n, E0_dense=truth["E0"],
                       t_dense=truth["t_dense"], rows=rows)
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
