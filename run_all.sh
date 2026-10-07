#!/usr/bin/env bash
# CABEF full pipeline. Fails fast: a broken stage must not silently produce
# a partial results table.
set -euo pipefail
cd "$(dirname "$0")"
for s in 01_sample 02_demographic 03_toxicity 04_geographic 05_bbq_ccf 06_cbi; do
  echo ""; echo "########## $s ##########"
  python scripts/${s}.py "$@"
done
echo ""; echo "Done. See results/paper_tables.md"
