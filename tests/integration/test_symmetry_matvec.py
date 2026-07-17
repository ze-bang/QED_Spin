"""Guard the symmetry-matvec sub-modes against each other.

The rep matvec has two sub-modes selected by env (read once per process,
hence subprocesses here):
  * default                 -> reduced-CSR  (RepReducedCsr)
  * ED_SYM_REDUCED_CSR=0    -> rep walk     (RepStream, CSR-free)
(The legacy orbit lane -- ED_SYM_REP=0 / OrbitMaterialized -- was retired in
Stage 11c-2b; the rep kernel is the ONE representation.)

They MUST give the same physics. A mis-wired default (the old orbit path was
silently ~5-48x slower) slipped through precisely because nothing compared
them. The equivalence test is the fast CI gate; the perf guard (default must
not be catastrophically slower than the fastest strategy) is `slow`-marked.

Worker mode: `python test_symmetry_matvec.py --worker` prints `RESULT eig=.. time=..`.
"""
import os, sys, subprocess, time, contextlib, io
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STRATEGIES = {
    "reduced_csr": {},                          # default
    "rep_walk":    {"ED_SYM_REDUCED_CSR": "0"},
}


def _run_worker(env_extra, lx=2, ly=2):
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("OMP_NUM_THREADS", "4")
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", str(lx), str(ly)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
    assert line, f"worker produced no RESULT (stderr tail:\n{out.stderr[-800:]})"
    kv = dict(tok.split("=") for tok in line[-1].split()[1:])
    return float(kv["eig"]), float(kv["time"])


def test_symmetry_matvec_equivalence():
    """Both sub-modes give the SAME ground state (12-site kagome, sz+trans)."""
    eigs = {name: _run_worker(env)[0] for name, env in STRATEGIES.items()}
    ref = eigs["reduced_csr"]
    for name, e in eigs.items():
        assert abs(e - ref) < 1e-9, f"{name} GS {e} != reduced_csr {ref} (diff {e-ref:.2e})"


@pytest.mark.slow
def test_default_not_catastrophically_slow():
    """Default (reduced-CSR) must not be much slower than the fastest strategy.
    Catches a mis-defaulted sym-matvec (the retired orbit path was 5-48x slower)."""
    times = {name: _run_worker(env, lx=3, ly=2)[1] for name, env in STRATEGIES.items()}
    fastest = min(times.values())
    assert times["reduced_csr"] <= 1.8 * fastest, \
        f"default reduced-CSR slow: {times}"


# --------------------------------------------------------------------------
def _worker(lx, ly):
    try:
        import qed
    except ImportError:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
        import qed
    from qed.input import lattice as L, HamiltonianBuilder as HB
    lat = L.kagome(lx, ly, pbc=True); N = lat.num_sites
    nn = [(b.i, b.j) for b in lat.nn_bonds]
    H = HB(N).xxz(nn, -1.0, 1.0).to_operator()
    with contextlib.redirect_stdout(io.StringIO()):
        trans = qed.find_symmetries(H, lattice=lat, translation_only=True,
                                    verbose=False).translation_set
        t = time.time()
        r = qed.solve(H, num_eigenvalues=1, solver="LANCZOS", device="cpu",
                      sz=N // 2, symmetry=trans, max_iterations=150,
                      verbose=False, auto_sz=False)
        dt = time.time() - t
    print(f"RESULT eig={r.eigenvalues[0]:.10f} time={dt:.3f}", flush=True)


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "--worker":
    _worker(int(sys.argv[2]) if len(sys.argv) > 2 else 2,
            int(sys.argv[3]) if len(sys.argv) > 3 else 2)
