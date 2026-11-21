"""Run our disorder sweeps on many Modal GPUs at once.

Kept separate from kaggle/ (which works and is left unchanged). Same science code: every
(lambda_nu, W, L) point goes through analysis/sweep.py::run_point, i.e. the spin-up Chern number
C_up (gpu_engine.spin_chern) with refinement at nk = 10, so rows are identical to Kaggle/laptop
runs. Points are independent, so Modal fans them out over up to --max-gpus containers.

Local side needs only `modal` (no torch). From hackathon/:
    modal run modal_runs/app.py --preset test                 # tests + timing on one GPU (~5 min)
    modal run modal_runs/app.py --preset timing               # seconds per sample at each L
    modal run modal_runs/app.py --preset wc1 --dry-run        # list jobs + cost estimate, no GPU
    modal run modal_runs/app.py --preset wc1 --max-gpus 10    # the real run

Results: results/modal/<preset>.csv, one row per point, appended as each point finishes.
Re-running the same command skips points already in the file (resume after a crash/cancel).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # hackathon/modal_runs
HACKATHON = HERE.parent
ROOT = HACKATHON.parent  # repo root: PythTB fork

# ---------------------------------------------------------------- image (built once, cached)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy>=2.0", "matplotlib>=3.9", "tqdm", "threadpoolctl>=3.8.0", "pytest")
    .add_local_dir(
        ROOT, "/root/proj", copy=True,
        ignore=["**/.venv/**", "**/__pycache__/**", "**/*.egg-info/**", "**/.pytest_cache/**",
                "**/.ruff_cache/**", ".git/**", "hackathon/results/**", "hackathon/joeys_og_folder/**",
                "docs/**", "**/*.ipynb"],
    )
    .run_commands("pip install -e /root/proj", "pip install --no-deps -e /root/proj/hackathon")
)

app = modal.App("pythtb-tai-sweeps", image=image)

# GPU hourly rates (USD, approximate, third-party sources Nov 2026; check modal.com/pricing).
RATE = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A100": 2.10, "A100-80GB": 2.50, "H100": 3.95}

# ---------------------------------------------------------------- remote side

_warm = False


@app.function(gpu="A100", timeout=3 * 3600, max_containers=10)
def run_point(size: int, lambda_nu: float, W: float, samples: int, nk: int, refine_nk: int,
              precision: str, seed: int) -> dict:
    """One sweep point on the GPU: exactly analysis/sweep.py::run_point (invariant c_up)."""
    global _warm
    import argparse
    import importlib.util
    import sys

    sys.path.insert(0, "/root/proj/hackathon")
    spec = importlib.util.spec_from_file_location("sweep", "/root/proj/hackathon/analysis/sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    from gpu_engine import get_backend
    from gpu_engine.spin_chern import run_spin_chern

    be = get_backend(precision)
    if not _warm:  # first call in this container: CUDA init and kernel loading, not timed
        run_spin_chern(size, 1.0, nk, [0, 1], be, delta=lambda_nu, refine_nk=None)
        _warm = True
    args = argparse.Namespace(size=size, nk=nk, refine_nk=refine_nk, samples=samples,
                              precision=precision, seed=seed, invariant="c_up")
    return sweep.run_point(args, lambda_nu, W, be, spin_blocks=False)


@app.function(gpu="A100", timeout=1800)
def run_tests() -> str:
    """C_up tests vs the validated CPU version, then the timing benchmark at L = 8/12/16."""
    import subprocess

    out = []
    for cmd in (["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_spin_chern.py"],
                ["python", "benchmarks/gpu_spin_chern.py", "--sizes", "8", "12", "16",
                 "--configs", "20", "--W", "2.0"]):
        r = subprocess.run(cmd, cwd="/root/proj/hackathon", capture_output=True, text=True)
        out.append(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}[exit {r.returncode}]\n")
    return "\n".join(out)


# ---------------------------------------------------------------- job presets (local side)

# Orth et al. Fig. 5 tight-binding Wc, doubled to our W convention (reference_data/orth2016_fig5.csv)
ORTH_WC = {1.60: 1.57, 1.65: 2.14, 1.70: 2.57, 1.75: 2.86, 1.80: 3.29, 1.85: 3.57, 1.90: 3.86, 2.00: 4.43}


def grid(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def preset_jobs(preset: str, small_gpu: str, big_gpu: str) -> list[dict]:
    """Each job: one (lambda_nu, W, L) point. Small sizes on the cheap GPU, big ones on the fast one."""
    def job(L, ld, W, samples):
        return {"size": L, "lambda_nu": ld, "W": W, "samples": samples, "nk": 6, "refine_nk": 10,
                "precision": "single", "seed": 0, "gpu": small_gpu if L <= 12 else big_gpu}

    if preset == "timing":  # 50 samples at the transition, every size we might use
        return [job(L, 1.65, 2.0, 50) for L in (8, 12, 16, 24, 32)]
    if preset == "pilot":  # whole W range, two sizes: where are both transitions?
        return [job(L, 1.65, W, 100) for L in (8, 12) for W in grid(0.0, 9.0, 0.5)]
    if preset == "wc1":  # FSS of the lower transition (core goal)
        return [job(L, 1.65, W, 1000) for L in (8, 12, 16, 24) for W in grid(1.0, 3.0, 0.2)]
    if preset == "wc2":  # FSS of the upper transition (TAI -> Anderson insulator)
        return [job(L, 1.65, W, 1000) for L in (8, 12, 16, 24) for W in grid(6.0, 8.0, 0.2)]
    if preset == "fig5":  # Wc1 at more lambda_nu: an FSS version of Orth Fig. 5
        return [job(L, ld, W, 500) for ld, wc in ORTH_WC.items() if ld != 1.65 for L in (8, 12, 16)
                for W in grid(max(0.0, wc - 1.0), wc + 1.0, 0.2)]
    if preset == "big":  # largest size at both transitions
        return [job(32, 1.65, W, 300) for W in grid(1.4, 2.6, 0.2) + grid(6.4, 7.6, 0.2)]
    raise SystemExit(f"unknown preset {preset!r}: test, timing, pilot, wc1, wc2, fig5, big")


# rough seconds per sample on an A100, single precision, incl. ~20% refinement (estimates until
# the `timing` preset measures them; L4 assumed ~2x slower). L = 8 from R11: 0.074 s per sample
# on a T4 incl. refinement, so ~0.02 s on an A100 (small matrices: overheads dominate).
EST_S = {8: 0.02, 12: 0.05, 16: 0.2, 24: 1.7, 32: 10.0}


def estimate(jobs: list[dict]) -> tuple[float, float]:
    gpu_h, usd = 0.0, 0.0
    for j in jobs:
        s = EST_S.get(j["size"], 1.0) * j["samples"] * (2.0 if j["gpu"] in ("L4", "T4", "A10") else 1.0)
        s += 20  # per-point overhead (container start-up amortised, Python, copies)
        gpu_h += s / 3600
        usd += s / 3600 * RATE.get(j["gpu"], 2.5)
    return gpu_h, usd


# ---------------------------------------------------------------- CSV (same columns as sweep.py)

FIELDS = [
    "lambda_nu", "W", "supercell", "nk", "refine_nk", "samples", "precision", "spin_blocks",
    "n_reliable", "n_topological", "frac_topological", "stderr", "n_unreliable", "n_gap_closed",
    "n_refined", "t_gpu", "t_refine", "invariant", "device", "gpu", "timestamp",
]


def done_points(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(float(r["lambda_nu"]), float(r["W"]), int(r["supercell"]), int(r["samples"]))
                for r in csv.DictReader(f)}


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


@app.local_entrypoint()
def main(preset: str = "test", max_gpus: int = 10, small_gpu: str = "L4", big_gpu: str = "A100",
         dry_run: bool = False, out: str = ""):
    if preset == "test":
        print(run_tests.with_options(gpu=big_gpu).remote())
        return

    jobs = preset_jobs(preset, small_gpu, big_gpu)
    path = Path(out) if out else HACKATHON / "results" / "modal" / f"{preset}.csv"
    done = done_points(path)
    todo = [j for j in jobs if (j["lambda_nu"], j["W"], j["size"], j["samples"]) not in done]
    gpu_h, usd = estimate(todo)
    print(f"preset {preset}: {len(jobs)} points, {len(jobs) - len(todo)} already in {path}, "
          f"{len(todo)} to run on up to {max_gpus} GPUs ({small_gpu} for L <= 12, {big_gpu} above)")
    print(f"rough estimate: {gpu_h:.1f} GPU-hours, ~${usd:.0f}, "
          f"~{gpu_h / max(1, min(max_gpus, len(todo))) * 60:.0f} min wall time if spread evenly")
    if dry_run or not todo:
        return

    t0, gpu_s, n_ok, n_err = time.time(), 0.0, 0, 0
    for gpu in sorted({j["gpu"] for j in todo}):
        part = [j for j in todo if j["gpu"] == gpu]
        fn = run_point.with_options(gpu=gpu, max_containers=max_gpus)
        args = [(j["size"], j["lambda_nu"], j["W"], j["samples"], j["nk"], j["refine_nk"],
                 j["precision"], j["seed"]) for j in part]
        print(f"\n== {len(part)} points on {gpu}")
        for res in fn.starmap(args, order_outputs=False, return_exceptions=True):
            if isinstance(res, BaseException):
                n_err += 1
                print(f"  FAILED: {res!r}")
                continue
            append_row(path, res)
            n_ok += 1
            gpu_s += float(res["t_gpu"]) + float(res["t_refine"])
            print(f"  [{time.time() - t0:6.0f}s] L={res['supercell']:>2} lambda_nu={res['lambda_nu']} "
                  f"W={float(res['W']):5.2f}: f = {res['frac_topological']} +- {res['stderr']} "
                  f"(unreliable {res['n_unreliable']}, refined {res['n_refined']}) "
                  f"{float(res['t_gpu']) + float(res['t_refine']):.1f}s on {res['gpu']}")
    print(f"\ndone: {n_ok} points saved to {path}, {n_err} failed; "
          f"compute time inside the jobs {gpu_s / 3600:.2f} GPU-h; wall {(time.time() - t0) / 60:.1f} min")
