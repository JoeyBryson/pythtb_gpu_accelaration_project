#!/usr/bin/env bash
# Run our code on a Kaggle GPU (T4) from your laptop, and fetch the results.
#
#   cd hackathon && KAGGLE_USER=<your-kaggle-username> bash kaggle/gpu_run.sh
#   PRESET=micro bash kaggle/gpu_run.sh    # another job preset from kaggle/run.py
#
# Full instructions: kaggle/README.md. Needs a Kaggle API token in ~/.kaggle/access_token
# (Kaggle > Settings > API), never in the repo. What runs on the GPU is set by JOBS in
# kaggle/run.py. KAGGLE_USER defaults to oscarklemenz.
#
# Steps: pack the code -> upload as a new version of a PRIVATE dataset on your account
# (created the first time) -> run kaggle/run.py with a GPU -> wait -> download the log and
# CSVs to results/kaggle/<time>/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # hackathon/
ROOT="$(cd "$HERE/.." && pwd)"                            # repo root (PythTB fork)
K() { uvx -q --from 'kaggle>=1.8' --with six kaggle "$@"; }
USER_NAME="${KAGGLE_USER:-oscarklemenz}"
PRESET="${PRESET:-speed}"
DATASET="$USER_NAME/pythtb-gpu-hackathon-code"
# each preset is its own Kaggle job, so different presets don't replace each other's runs
SLUG="pythtb-gpu-run"; [ "$PRESET" != "speed" ] && SLUG="pythtb-gpu-$PRESET"
KERNEL="$USER_NAME/$SLUG"
grep -q "^    \"$PRESET\": \[" "$HERE/kaggle/run.py" || {
    echo "unknown PRESET '$PRESET': see PRESETS in kaggle/run.py"; exit 1; }
STAMP=$(date +%Y%m%d-%H%M%S)
STAGE=$(mktemp -d)

echo "== packing code (Kaggle account: $USER_NAME, preset: $PRESET, job: $KERNEL)"
mkdir -p "$STAGE/dataset" "$STAGE/kernel"
# COPYFILE_DISABLE=1: don't pack macOS "._" metadata files
COPYFILE_DISABLE=1 tar -C "$ROOT" --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='hackathon/joeys_og_folder' \
    --exclude='hackathon/results' \
    -czf "$STAGE/dataset/code.tar.gz" pyproject.toml README.md LICENSE pythtb hackathon
# metadata files carry the account name in their ids: rewrite them for this user
sed "s#oscarklemenz/#$USER_NAME/#" "$HERE/kaggle/dataset-metadata.json" > "$STAGE/dataset/dataset-metadata.json"
sed -e "s#oscarklemenz/pythtb-gpu-run#$KERNEL#" -e "s#\"pythtb-gpu-run\"#\"$SLUG\"#" \
    "$HERE/kaggle/kernel-metadata.json" > "$STAGE/kernel/kernel-metadata.json"
sed "s#^PRESET = \"speed\"#PRESET = \"$PRESET\"#" "$HERE/kaggle/run.py" > "$STAGE/kernel/run.py"
grep -q "^PRESET = \"$PRESET\"" "$STAGE/kernel/run.py" || { echo "unknown preset handling"; exit 1; }

if K datasets status "$DATASET" >/dev/null 2>&1; then
    echo "== uploading new version of private dataset $DATASET"
    K datasets version -p "$STAGE/dataset" -m "run $STAMP"
else
    echo "== first run on this account: creating private dataset $DATASET"
    K datasets create -p "$STAGE/dataset"
fi
until [ "$(K datasets status "$DATASET" | tail -1)" = "ready" ]; do sleep 10; done

echo "== starting GPU run $KERNEL (watch live at https://www.kaggle.com/code/$KERNEL)"
K kernels push -p "$STAGE/kernel" --accelerator NvidiaTeslaT4
while true; do
    STATUS=$(K kernels status "$KERNEL" | tail -1)
    echo "$(date +%H:%M:%S) $STATUS"
    case "$STATUS" in *COMPLETE*|*ERROR*|*CANCEL*) break ;; esac
    sleep 20
done

OUT="$HERE/results/kaggle/$STAMP"
mkdir -p "$OUT"
K kernels output "$KERNEL" -p "$OUT"
echo "== results in $OUT"
ls "$OUT"
