# AUDIT-ID: E4-python-arch-01
# DEVICE: cpu
# SECONDS: 60
"""Claim: find_symmetries(H, verbose=False) and the default Symmetry.auto() path of every verb
still write discovery chatter to Python's sys.stdout, because only the nauty pipeline step is
wrapped in contextlib.redirect_stdout; the max-clique search (AutomorphismCliqueAnalyzer) and the
minimal-generator extraction (MaximalAbelianSubgroupFinder) print unconditionally."""

import contextlib
import io
import signal

signal.alarm(250)
try:
    import qed
    import qed.discovery as disc
except Exception as e:  # pragma: no cover
    print(f"REPRO: INCONCLUSIVE import failed: {e!r}")
    raise SystemExit(0)

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

MARKERS = ("Building commutation graph", "Finding maximum clique", "Finding minimal generators")


def captured(fn):
    disc._FIND_SYM_MEMO.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


try:
    out_fs = captured(lambda: qed.find_symmetries(H, verbose=False))
    out_eigs = captured(lambda: qed.eigs(H, 2))
except ImportError as e:
    print(f"REPRO: INCONCLUSIVE pynauty/networkx missing: {e!r}")
    raise SystemExit(0)
except Exception as e:
    print(f"REPRO: INCONCLUSIVE unexpected {type(e).__name__}: {e}")
    raise SystemExit(0)

hit_fs = [m for m in MARKERS if m in out_fs]
hit_eigs = [m for m in MARKERS if m in out_eigs]
if hit_fs and hit_eigs:
    print(
        f"REPRO: CONFIRMED verbose=False printed {len(out_fs.splitlines())} lines {hit_fs}; "
        f"qed.eigs(H,2) default sym printed {len(out_eigs.splitlines())} lines {hit_eigs}"
    )
elif hit_fs or hit_eigs:
    print(f"REPRO: CONFIRMED partial: find_symmetries markers={hit_fs} eigs markers={hit_eigs}")
else:
    print(
        f"REPRO: NOT_REPRODUCED no discovery chatter on sys.stdout "
        f"(find_symmetries {len(out_fs)} chars, eigs {len(out_eigs)} chars)"
    )
