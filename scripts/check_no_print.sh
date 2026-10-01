#!/bin/bash
# The library writes nothing to the console on its own (pure grep):
#   1. no C++ source in include/, src/ or python/qed/_bindings writes to stdout / stderr
#      (std::cout / cerr / clog, printf / fprintf / puts / fputs / putchar / perror);
#      everything goes through ED_LOG (include/ed/core/log.h, the one sink);
#   2. no module of python/qed prints or writes to sys.stdout / sys.stderr, or hides
#      output with redirect_stdout / redirect_stderr; it logs through qed._log.
# ALLOW lists the files exempt from the check (path relative to the repository root).
# Exit 1 on any hit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ALLOW=(
  include/ed/core/log.h        # the stream sink itself
)
excl=()
for f in "${ALLOW[@]}"; do excl+=(-e "^${f}:"); done
filter() { if [ ${#excl[@]} -gt 0 ]; then grep -v "${excl[@]}" || true; else cat; fi; }

cpp=$(grep -rnE 'std::(cout|cerr|clog)\b|(^|[^A-Za-z0-9_])(v?f?printf|puts|fputs|putchar|perror)[[:space:]]*\(' \
        include src python/qed/_bindings \
        --include=*.h --include=*.hpp --include=*.cpp --include=*.cu --include=*.cuh \
      | filter || true)
py=$(grep -rnE '(^|[^A-Za-z0-9_.])print[[:space:]]*\(|sys\.(stdout|stderr)\.write|redirect_std(out|err)' \
        python/qed --include=*.py \
      | filter || true)
status=0
if [ -n "${cpp}" ]; then echo "C++ CONSOLE WRITES (use ED_LOG, include/ed/core/log.h):"; echo "${cpp}" | sed 's/^/  /'; status=1; fi
if [ -n "${py}" ];  then echo "PYTHON CONSOLE WRITES (use qed._log):"; echo "${py}" | sed 's/^/  /'; status=1; fi
echo "no_print: status=${status}"
exit ${status}
