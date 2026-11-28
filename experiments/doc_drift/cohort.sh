#!/bin/bash
# Run a set of pages through record.py, two runs each with a repair pass, into a results folder.
#
#     experiments/doc_drift/cohort.sh <results-dir> <url> [<url> ...]
#     ATOMIZER=strict STORE=drift-store-strict experiments/doc_drift/cohort.sh experiments/doc_drift/results/strict/cohort <url> ...
#
# This produced results/cohort/ and results/holdout/ (sentence extractor) and the same folders under
# results/strict/ (strict extractor), a few pages per batch in parallel. Needs the same model access as run.py.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?results directory}
shift
mkdir -p "$OUT"
cd "$HERE/../.."
for url in "$@"; do
  slug=$(basename "$url" .html)
  echo "=== $slug"
  python -I "$HERE/record.py" "$url" 2 "$OUT/$slug-runs.json" repair 2>&1 | grep -v "^Warning"
done
echo "batch done"
