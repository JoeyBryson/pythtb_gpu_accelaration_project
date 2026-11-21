#!/usr/bin/env bash
# Run our code on a Kaggle GPU (T4) from your laptop, and fetch the results.
#
#   cd hackathon && bash kaggle/gpu_run.sh
#
# Needs a Kaggle API token in ~/.kaggle/access_token (Kaggle > Settings > API), never in
# the repo. What runs on the GPU is set by JOBS in kaggle/run.py.
# The ids in kaggle/*-metadata.json use Oscar's Kaggle account; change "oscarklemenz" to
# your username to run on yours.
#
# Steps: pack the code -> upload as a new version of the PRIVATE dataset -> run
# kaggle/run.py with a GPU -> wait -> download log + CSVs to results/kaggle/<time>/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # hackathon/
ROOT="$(cd "$HERE/.." && pwd)"                            # repo root (PythTB fork)
K() { uvx -q --from 'kaggle>=1.8' --with six kaggle "$@"; }
OWNER=$(python3 -c "import json;print(json.load(open('$HERE/kaggle/kernel-metadata.json'))['id'])")
STAMP=$(date +%Y%m%d-%H%M%S)
STAGE=$(mktemp -d)

echo "== packing code"
mkdir -p "$STAGE/dataset" "$STAGE/kernel"
COPYFILE_DISABLE=1 tar -C "$ROOT" --exclude='.venv'   # COPYFILE_DISABLE: no macOS ._ files --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='hackathon/joeys_og_folder' \
    --exclude='hackathon/results' \
    -czf "$STAGE/dataset/code.tar.gz" pyproject.toml README.md LICENSE pythtb hackathon
cp "$HERE/kaggle/dataset-metadata.json" "$STAGE/dataset/"
cp "$HERE/kaggle/run.py" "$HERE/kaggle/kernel-metadata.json" "$STAGE/kernel/"

echo "== uploading new dataset version (private)"
K datasets version -p "$STAGE/dataset" -m "run $STAMP"
DATASET=$(python3 -c "import json;print(json.load(open('$HERE/kaggle/dataset-metadata.json'))['id'])")
until [ "$(K datasets status "$DATASET" | tail -1)" = "ready" ]; do sleep 10; done

echo "== starting GPU run ($OWNER)"
K kernels push -p "$STAGE/kernel" --accelerator NvidiaTeslaT4
while true; do
    STATUS=$(K kernels status "$OWNER" | tail -1)
    echo "$(date +%H:%M:%S) $STATUS"
    case "$STATUS" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
    sleep 20
done

OUT="$HERE/results/kaggle/$STAMP"
mkdir -p "$OUT"
K kernels output "$OWNER" -p "$OUT"
echo "== results in $OUT"
ls "$OUT"
