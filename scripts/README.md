# scripts/

Build, test and gate tooling. Nothing here is imported by the library.

```
scripts/
├── build.sh                 the one build entry point (see its header); clusters/*.env per site
├── check_env_registry.sh    the ED_* environment contract (registry <-> sources, no raw getenv)
├── check_no_print.sh        the library writes nothing to stdout/stderr (log channel only)
├── gate/                    the gate: build job + CPU and GPU task arrays (tasks.sh lists the stages)
└── golden/                  golden suite: CPU/GPU compare, record, bless
```

## Gate (Alliance clusters; never on a login node)

`gate/submit.sh <account>` submits the build and, depending on it, one CPU and one GPU job
array; every task appends `<stage> <exit code>` to `logs/gate/<build id>/rc`. Summary:

```
awk '{c[$1]=$2} END {for (k in c) print k, c[k]}' logs/gate/<build id>/rc
```

A change is ready when every stage reports 0. Run the gate from a snapshot of the tree
(a worktree synced with `rsync --checksum`), not the tree being edited: the tasks import
`python/` while they run.

The same script runs part of the table or the other build:

```
gate/submit.sh <account> ctest                    # the C++ unit tests (on a GPU slice)
gate/submit.sh <account> pytest 'grid_cpu_*'      # stages by name or glob
gate/submit.sh <account> --variant cpu ctest      # the CPU-only build and its unit tests
gate/submit.sh <account> --build-only             # build the extension and nothing else
```
