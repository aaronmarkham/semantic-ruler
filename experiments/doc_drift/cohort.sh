#!/bin/bash
# Run a set of pages through record.py, two runs each with a repair pass, into a results folder.
#
#     experiments/doc_drift/cohort.sh <results-dir> <url> [<url> ...]
#     ATOMIZER=strict STORE=drift-store-strict experiments/doc_drift/cohort.sh experiments/doc_drift/results/strict/cohort <url> ...
#
# This produced results/cohort/ (sentence extractor) and results/strict/cohort/ (strict extractor),
# three batches of three or four pages in parallel. The OpenAI key and Claude access are as for run.py.
set -u
HERE=/bin
OUT=; shift
mkdir -p ""
cd "/../.."
for url in ""; do
  slug=
  echo "=== "
  python -I "/record.py" "" 2 "/-runs.json" repair 2>&1 | grep -v "^Warning"
done
echo "batch done"
