"""Run our code on a Modal cloud GPU from your laptop, and fetch the results.

Written at the hackathon (21 Nov 2026). The Modal counterpart of kaggle/gpu_run.sh: same
presets (from kaggle/run.py), but any GPU Modal offers (T4, L4, A10G, A100, H100, ...), no
queue, and logs stream straight to your terminal.

From hackathon/ (full instructions: modal_gpu/README.md):
  uvx modal run modal_gpu/app.py                              # preset "speed" on a T4
  uvx modal run modal_gpu/app.py --preset check --gpu H100    # versions + GPU + eigh timing
  uvx modal run modal_gpu/app.py --cmd "scripts/ensemble_sweep.py --quick" --gpu A100
  uvx modal run modal_gpu/app.py --list-presets               # show presets, run nothing

Results land in results/modal/<time>/ (plus modal-run.log). They are also kept on the Modal
volume "pythtb-gpu-results" under runs/<time>/, written while the run goes, so a run you stop
or that times out keeps its data: `uvx modal volume get pythtb-gpu-results runs/<time> results/modal/`.

Only stdlib + modal are imported at the top: this file is also imported on your laptop.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[1]  # hackathon/
ROOT = HERE.parent  # repo root (PythTB fork)
REMOTE = "/proj"  # code lives at /proj/pythtb and /proj/hackathon in the container
VOLUME_NAME = "pythtb-gpu-results"
VOL = "/vol"

# Same dependencies as hackathon/pyproject.toml. torch from PyPI on Linux ships with CUDA
# (CUDA 13.2 build on 21 Nov). If the GPU driver is too old for it, rebuild with an older CUDA
# build, e.g. TORCH_INDEX=https://download.pytorch.org/whl/cu128 uvx modal run ...
# PythTB and gpu_engine are not pip-installed: they're found via PYTHONPATH.
TORCH_INDEX = os.environ.get("TORCH_INDEX")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.4", **({"index_url": TORCH_INDEX} if TORCH_INDEX else {}))
    .uv_pip_install("numpy>=2.0", "matplotlib>=3.9", "tqdm", "threadpoolctl>=3.8", "pytest>=8")
    .env({"PYTHONPATH": f"{REMOTE}:{REMOTE}/hackathon", "PYTHONUNBUFFERED": "1",
          "MPLBACKEND": "Agg"})
    .add_local_dir(ROOT / "pythtb", f"{REMOTE}/pythtb", ignore=["**/__pycache__"])
    .add_local_dir(
        HERE, f"{REMOTE}/hackathon",
        ignore=[".venv", "**/__pycache__", "*.egg-info", ".pytest_cache", ".ruff_cache",
                "results", "joeys_og_folder", "presentation", "notebooks"],
    )
)

app = modal.App("pythtb-gpu", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ----------------------------------------------------------------------------- presets

# Modal-only presets, on top of the Kaggle ones.
MODAL_PRESETS = {
    # "check": versions, GPU name, float32 vs float64 eigh timing (~1-2 min incl. start-up).
    # Run this first on a new GPU type to decide the precision strategy.
    "check": [["scripts/check_env.py"]],
    "tests": [["-m", "pytest", "-q", "-p", "no:cacheprovider"]],
}


def load_presets() -> dict[str, list[list[str]]]:
    """Kaggle presets (kaggle/run.py, single source of truth) + MODAL_PRESETS.

    Kaggle jobs write to /kaggle/working/out/; here that becomes results/ (which the container
    links to the volume), so the outputs come back with everything else.
    """
    spec = importlib.util.spec_from_file_location("kaggle_run", HERE / "kaggle" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # stdlib imports + constants only; main() is guarded
    fix = lambda a: a.replace("/kaggle/working/out/", "results/")
    presets = {name: [[fix(a) for a in job] for job in jobs] for name, jobs in mod.PRESETS.items()}
    return {**presets, **MODAL_PRESETS}


# ----------------------------------------------------------------------------- remote side

def _run_logged(cmd: list[str], cwd: str, log) -> int:
    """Run cmd, streaming its output to the terminal and to the log file on the volume."""
    import subprocess

    start = time.time()
    head = f"\n[{time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}"
    print(head, flush=True)
    log.write(head + "\n")
    proc = subprocess.Popen(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    code = proc.wait()
    tail = f"[{time.strftime('%H:%M:%S')}] exit {code} after {time.time() - start:.0f}s"
    print(tail, flush=True)
    log.write(tail + "\n")
    return code


def _gpu_status() -> str:
    """GPU name/memory/driver and whether torch can use it, as one line ("" if no GPU).

    A driver too old for torch's CUDA build would otherwise make every script quietly fall
    back to CPU, so this is printed at the top of every run.
    """
    import shutil
    import subprocess
    import sys

    if not shutil.which("nvidia-smi"):
        return ""
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], check=False, capture_output=True,
                         text=True).stdout.strip()
    code = "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    out = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True,
                         text=True).stdout.split()
    if out and out[-1] == "True":
        return f"{gpu}; torch {out[0]} (CUDA {out[1]}) sees it"
    return (f"{gpu}\nWARNING: torch can't use this GPU ({' '.join(out)}), jobs will run on CPU. "
            "See modal_gpu/README.md, 'torch can't see the GPU'.")


# No GPU in the decorator on purpose: main() adds the chosen one with .with_options(gpu=...),
# and `--gpu cpu` runs the same jobs on CPU only (cheap plumbing test, or a CPU baseline).
@app.function(cpu=4.0, timeout=4 * 3600, volumes={VOL: volume})
def run_jobs(jobs: list[list[str]], stamp: str) -> dict[str, bytes]:
    """Run each job as `python <job...>` from hackathon/; return every file the jobs wrote.

    jobs: list of argument lists, e.g. [["scripts/check_env.py"], ["-m", "pytest", "-q"]].
    A failing job is reported but doesn't stop the later jobs (as on Kaggle).
    """
    import sys
    import threading

    hack = Path(REMOTE) / "hackathon"
    out = Path(VOL) / "runs" / stamp
    out.mkdir(parents=True, exist_ok=True)
    # scripts write to hackathon/results/: point that at the volume so outputs persist live
    results = hack / "results"
    if results.is_symlink() or results.exists():
        results.unlink()
    results.symlink_to(out, target_is_directory=True)

    # commit the volume every minute, so a stopped / timed-out run keeps what it has
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(60):
            try:
                volume.commit()
            except Exception as e:  # never kill the run over a failed background commit
                print(f"(volume commit failed: {e})", flush=True)

    threading.Thread(target=committer, daemon=True).start()

    gpu = _gpu_status() or "none (CPU only)"
    with open(out / "modal-run.log", "a") as log:
        log.write(f"Modal run {stamp}, {os.cpu_count()} CPUs visible, GPU: {gpu}\n")
        print(f"GPU: {gpu}", flush=True)
        codes = [_run_logged([sys.executable, *job], str(hack), log) for job in jobs]
        log.write(f"\nexit codes: {codes}\n")

    stop.set()
    volume.commit()
    files = {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    print(f"\nexit codes: {codes}; returning {len(files)} files", flush=True)
    return files


# ----------------------------------------------------------------------------- laptop side

@app.local_entrypoint()
def main(preset: str = "speed", gpu: str = "T4", cmd: str = "", timeout_h: float = 4.0,
         cpu: float = 4.0, list_presets: bool = False) -> None:
    """--preset NAME | --cmd "script.py args; other.py args"
    --gpu T4|L4|A10G|L40S|A100|A100-80GB|H100|H200|B200|cpu   --cpu CORES   --timeout-h HOURS"""
    presets = load_presets()
    if list_presets:
        for name, jobs in presets.items():
            print(f"{name}:")
            for job in jobs:
                print("    python", " ".join(job))
        return
    if cmd:
        jobs = [shlex.split(c) for c in cmd.split(";") if c.strip()]
        label = "custom"
    else:
        if preset not in presets:
            raise SystemExit(f"unknown preset '{preset}'; choose from {sorted(presets)}")
        jobs, label = presets[preset], preset

    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{label}-{gpu.lower()}"
    print(f"== Modal run {stamp}: {len(jobs)} job(s) on {gpu}, timeout {timeout_h:g} h")
    for job in jobs:
        print("   python", " ".join(job))

    opts = {"timeout": int(timeout_h * 3600), "cpu": cpu}
    if gpu.lower() not in ("cpu", "none"):
        opts["gpu"] = gpu
    fn = run_jobs.with_options(**opts)
    files = fn.remote(jobs, stamp)

    dest = HERE / "results" / "modal" / stamp
    for rel, data in files.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(data)
    print(f"== {len(files)} file(s) in {dest}")
    for rel in sorted(files):
        print("  ", rel)
    print(f"(also on the Modal volume: {VOLUME_NAME}:runs/{stamp})")
