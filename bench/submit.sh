#!/bin/bash
# Submit every benchmark case (or the named ones) as its own job.
#   bench/submit.sh <account> [case ...]
# Each job runs bench/run.py on the cuda build of this checkout (scripts/golden/env.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
acct=$1; shift
if [ $# -gt 0 ]; then
    cases=("$@")
else
    mapfile -t cases < <(sed -n 's/^    "\([A-Za-z0-9_]*\)": *(.*/\1/p' bench/cases.py)
fi
for c in "${cases[@]}"; do
    res=$(sed -n "s/^    \"$c\": *([^,]*, *\"\(.*\)\"),/\1/p" bench/cases.py)
    [ -n "$res" ] || { echo "unknown case $c"; exit 1; }
    # shellcheck disable=SC2086
    sbatch --account="$acct" --job-name="bench_$c" --output=logs/%x-%j.out $res \
        --export=ALL,QED_VARIANT=cuda,CASE="$c" --wrap='source scripts/golden/env.sh && cd bench && python -u run.py "$CASE"'
done
