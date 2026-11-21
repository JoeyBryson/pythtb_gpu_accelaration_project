"""What would a whole disorder sweep cost on CPU vs GPU? Measured per sample, extrapolated.

Times three ways of answering "topological or not?" for a few disorder samples, on the
SAME machine, then multiplies by the number of samples in a full sweep:

  pythtb_cpu  PythTB 2.0 as-is (Wilson loops, gap-tracking Z2), all CPU cores
              -> what a PythTB user would do today
  c_up_cpu    our spin-up Chern number algorithm in NumPy, all CPU cores
              -> how much of the gain is the better algorithm alone
  c_up_gpu    the same algorithm, batched on the GPU (incl. refinement of uncertain samples)
              -> how much the GPU adds on top

Cost per sample barely depends on W (same matrix sizes); measured at W = 2.0 for
lambda_nu = 1.65, i.e. at the transition, where the GPU path refines the most samples
(the CPU baselines don't refine at all), so the GPU's advantage is if anything understated.
The result is an ESTIMATE from measured rates: label it as such.

Run from hackathon/:  uv run python benchmarks/sweep_cost.py --sizes 8 12 16
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from common import RESULTS_DIR, hardware_info, log, save_rows, scaled_nk
from cpu_baseline import bench_parallel

from gpu_engine import get_backend, sync
from gpu_engine.spin_chern import run_spin_chern

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from sweep import _cpu_c_up_task  # (Joey's validated NumPy C_up, one sample per task)

# samples to time per method, per size: enough for a stable rate, small enough to be quick
CPU_SAMPLES = {8: 24, 12: 12, 16: 8}


def gpu_rate(n: int, nk: int, lam: float, W: float, samples: int) -> float:
    be = get_backend("single")
    run_spin_chern(n, W, nk, [0], be, delta=lam, refine_nk=None)  # warm-up
    sync(be)
    t0 = time.perf_counter()
    run_spin_chern(n, W, nk, list(range(samples)), be, delta=lam, refine_nk=10)
    sync(be)
    return (time.perf_counter() - t0) / samples


def c_up_cpu_rate(n: int, nk: int, lam: float, W: float, samples: int) -> float:
    tasks = [(n, W, nk, lam, s) for s in range(samples)]
    with ProcessPoolExecutor(os.cpu_count(), mp_context=mp.get_context("fork")) as pool:
        list(pool.map(_cpu_c_up_task, tasks[: os.cpu_count()]))  # warm-up every worker
        t0 = time.perf_counter()
        list(pool.map(_cpu_c_up_task, tasks))
        return (time.perf_counter() - t0) / samples


def pythtb_rate(n: int, nk: int, lam: float, W: float, samples: int) -> float:
    bench_parallel(n, nk, W, samples, 0, workers=os.cpu_count(), reliability=False,
                   params={"delta": lam, "rashba": 0.0})
    with (RESULTS_DIR / "cpu_parallel.csv").open() as f:  # bench_parallel saves its wall time
        return float(list(csv.DictReader(f))[-1]["wall_s"]) / samples


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--lambda-nu", type=float, default=1.65)
    p.add_argument("--W", type=float, default=2.0)
    p.add_argument("--gpu-samples", type=int, default=100)
    p.add_argument("--sweep-samples", type=int, default=41 * 2 * 200,
                   help="samples in the sweep to extrapolate to (default 41 W x 2 lambda_nu x 200)")
    p.add_argument("--skip-pythtb", action="store_true")
    args = p.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    hw = hardware_info()
    log(f"sweep cost estimate on CPU {hw['cpu']} ({hw['n_cpus']} cores) and GPU {gpu}; "
        f"extrapolating to {args.sweep_samples} samples")
    rows = []
    for n in args.sizes:
        nk, cpu_n = scaled_nk(n), CPU_SAMPLES.get(n, 8)
        log(f"{n}x{n}, mesh {nk}x{nk}: timing GPU C_up on {args.gpu_samples} samples ...")
        r_gpu = gpu_rate(n, nk, args.lambda_nu, args.W, args.gpu_samples)
        log(f"{n}x{n}: timing C_up in NumPy on all cores, {cpu_n} samples ...")
        r_cup = c_up_cpu_rate(n, nk, args.lambda_nu, args.W, cpu_n)
        r_pyt = float("nan")
        if not args.skip_pythtb:
            log(f"{n}x{n}: timing PythTB on all cores, {cpu_n} samples ...")
            r_pyt = pythtb_rate(n, nk, args.lambda_nu, args.W, cpu_n)

        N = args.sweep_samples
        log(f"{n}x{n} per sample: PythTB {r_pyt:.3f} s | C_up CPU {r_cup:.3f} s | C_up GPU {r_gpu:.4f} s")
        log(f"{n}x{n} whole sweep ({N} samples): PythTB ~{hours(r_pyt * N)} | C_up CPU ~{hours(r_cup * N)}"
            f" | C_up GPU ~{hours(r_gpu * N)}  ->  algorithm {r_pyt / r_cup:.1f}x, "
            f"GPU on top {r_cup / r_gpu:.1f}x, total {r_pyt / r_gpu:.1f}x")
        rows.append({
            "supercell": n, "nk": nk, "lambda_nu": args.lambda_nu, "W": args.W,
            "s_per_sample_pythtb_cpu": f"{r_pyt:.4f}", "s_per_sample_c_up_cpu": f"{r_cup:.4f}",
            "s_per_sample_c_up_gpu": f"{r_gpu:.5f}", "sweep_samples": N,
            "sweep_h_pythtb_cpu": f"{r_pyt * N / 3600:.3f}", "sweep_h_c_up_cpu": f"{r_cup * N / 3600:.3f}",
            "sweep_h_c_up_gpu": f"{r_gpu * N / 3600:.4f}",
            "x_algorithm": f"{r_pyt / r_cup:.2f}", "x_gpu_on_top": f"{r_cup / r_gpu:.2f}",
            "x_total": f"{r_pyt / r_gpu:.2f}", "gpu": gpu, "cpu_samples": cpu_n,
            "gpu_samples": args.gpu_samples,
        })
    print(f"saved to {save_rows('sweep_cost', rows)}")


if __name__ == "__main__":
    main()
