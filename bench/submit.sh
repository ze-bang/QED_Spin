#!/bin/bash
# Submit every benchmark case (or the named ones) as its own job.
#   bench/submit.sh <account> [case ...]
#   REPEATS=3 (default): runs per job, each a fresh process; the walltime scales with it.
# Each job runs bench/run.py on the cuda build of this checkout (scripts/golden/env.sh),
# threads pinned to cores. XDiag twins run through bench/xdiag/twin.sbatch instead.
set -euo pipefail
cd "$(dirname "$0")/.."
acct=$1; shift
repeats=${REPEATS:-3}
if [ $# -gt 0 ]; then
    cases=("$@")
else
    mapfile -t cases < <(sed -n 's/^    "\([A-Za-z0-9_]*\)": *(.*/\1/p' bench/cases.py)
fi
# H:MM:SS times n
scale() { IFS=: read -r h m s <<< "$1"; local t=$(( (10#$h * 3600 + 10#$m * 60 + 10#$s) * $2 ));
          printf '%d:%02d:%02d' $((t / 3600)) $((t % 3600 / 60)) $((t % 60)); }
for c in "${cases[@]}"; do
    res=$(sed -n "s/^    \"$c\": *([^,]*, *\"\(.*\)\"),/\1/p" bench/cases.py)
    [ -n "$res" ] || { echo "unknown case $c"; exit 1; }
    t=$(sed -n 's/.*-t \([0-9:]*\).*/\1/p' <<< "$res")
    res=${res/-t $t/-t $(scale "$t" "$repeats")}
    # shellcheck disable=SC2086
    sbatch --account="$acct" --job-name="bench_$c" --output=logs/%x-%j.out $res \
        --export=ALL,QED_VARIANT=cuda,CASE="$c",REPEATS="$repeats",OMP_PROC_BIND=close,OMP_PLACES=cores \
        --wrap='source scripts/golden/env.sh && cd bench && python -u run.py "$CASE" --repeats "$REPEATS"'
done
