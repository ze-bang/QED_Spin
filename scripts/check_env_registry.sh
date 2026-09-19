#!/bin/bash
# Both directions of the environment-variable contract (pure grep):
#   1. every ED_* / QED_* name the sources mention next to an environment read is a row of
#      include/ed/config/env_registry.h;
#   2. every row is mentioned somewhere outside the registry (no dead rows);
#   3. no row is declared twice;
#   4. no C++ source reads an ED_* / QED_* variable with a raw getenv (go through ed::env).
# Exit 1 on any discrepancy.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REG=include/ed/config/env_registry.h
INC="--include=*.h --include=*.cpp --include=*.cu --include=*.cuh --include=*.py"
src_dirs="include src python/qed python/edlib"
reg=$(grep -oE 'X\("(ED|QED)_[A-Z0-9_]+"' "${REG}" | grep -oE '(ED|QED)_[A-Z0-9_]+' | sort -u)
dup=$(grep -oE 'X\("(ED|QED)_[A-Z0-9_]+"' "${REG}" | sort | uniq -d || true)
# names that appear as a string literal on a line that reads the environment or calls one
# of the name-parameterised helpers
used=$(grep -rhE 'getenv|environ|env::(flag|tristate|integer|real|text|raw)|read_cutoff|env_flag|env_int|env_size|env_double' \
         ${src_dirs} ${INC} --exclude=env_registry.h \
       | grep -oE '"(ED|QED)_[A-Z0-9_]+"' | tr -d '"' | sort -u)
all=$(grep -rhoE '(ED|QED)_[A-Z0-9_]+' ${src_dirs} ${INC} --exclude=env_registry.h | sort -u)
status=0
missing=$(comm -13 <(echo "${reg}") <(echo "${used}") || true)
dead=$(comm -23 <(echo "${reg}") <(echo "${all}") || true)
if [ -n "${missing}" ]; then echo "READ BUT NOT REGISTERED:"; echo "${missing}" | sed 's/^/  /'; status=1; fi
if [ -n "${dead}" ];    then echo "REGISTERED BUT NEVER MENTIONED IN THE SOURCES:"; echo "${dead}" | sed 's/^/  /'; status=1; fi
if [ -n "${dup}" ];     then echo "DUPLICATE ROWS:"; echo "${dup}" | sed 's/^/  /'; status=1; fi
raw=$(grep -rnE 'getenv\("(ED|QED)_' include src python/qed/_bindings --include=*.h --include=*.cpp --include=*.cu --include=*.cuh \
       | grep -v "^${REG}:" || true)
if [ -n "${raw}" ];     then echo "RAW getenv OF A REGISTERED-NAMESPACE VARIABLE (use ed::env):"; echo "${raw}" | sed 's/^/  /'; status=1; fi
echo "registry rows: $(echo "${reg}" | wc -l); names read in sources: $(echo "${used}" | wc -l); status=${status}"
exit ${status}
