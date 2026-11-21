"""Disorder sweep: fraction of samples that are topological vs disorder strength W.

For each staggered potential lambda_nu (`--deltas`) and each W on a grid, run `--samples`
disorder configurations through the batched GPU pipeline and record:
  - fraction topological (among reliable answers) with its binomial standard error
  - how many answers were unreliable, and how many had the band gap closed
Unreliable answers are re-run once on a finer k-mesh (`--refine-nk`); the rest stay unreliable
and are reported, never silently dropped.

Invariant (`--invariant`, default c_up): the spin-up Chern number C_up from the winding of
arg det(Wilson loop), Z2 = C_up mod 2 (gpu_engine.spin_chern; Rashba = 0 only). It converges
at nk = 6 where gap-tracking Z2 does not (analysis/stage0_chern_up.py, 21 Nov). `--invariant z2`
uses the gap-tracking pipeline (gpu_engine.pipeline.run_batch) instead.

Model: Kane-Mele, t = 1, lambda_SO = 0.3, lambda_R = 0 (team choice), on-site disorder
uniform in [-W/2, W/2] (our convention). The spin-block split is used where it is faster on a
GPU (supercells >= 12x12, RESULTS.md R10).

Saves one CSV row per (lambda_nu, W) as soon as it's done, and skips points already in the
output file, so a stopped run can be resumed by running the same command again.

Optional `--cpu-baseline`: time PythTB on all CPU cores for a few samples at a few W values,
on the same machine, to estimate what the whole sweep would cost on the CPU
(cost per sample barely depends on W: same matrix sizes).

Run from hackathon/:
  uv run python analysis/sweep.py --size 8 --samples 100 --W-step 0.5 --out results/sweep/pilot_8x8.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from common import hardware_info, log, scaled_nk

from gpu_engine import get_backend
from gpu_engine.pipeline import run_batch
from gpu_engine.spin_chern import run_spin_chern

FIELDS = [
    "lambda_nu", "W", "supercell", "nk", "refine_nk", "samples", "precision", "spin_blocks",
    "n_reliable", "n_topological", "frac_topological", "stderr", "n_unreliable", "n_gap_closed",
    "n_refined", "t_gpu", "t_refine", "invariant", "device", "gpu", "timestamp",
]


def point_seeds(base: int, lambda_nu: float, W: float, n: int) -> list[int]:
    """Reproducible, independent seeds for one (lambda_nu, W) point."""
    ss = np.random.SeedSequence([base, round(lambda_nu * 1000), round(W * 1000)])
    return [int(c.generate_state(1)[0]) for c in ss.spawn(n)]


def done_points(path: Path) -> set[tuple]:
    """(lambda_nu, W, supercell, nk, samples) already in the output file: skipped on resume."""
    if not path.exists():
        return set()
    with path.open() as f:
        return {(float(r["lambda_nu"]), float(r["W"]), int(r["supercell"]), int(r["nk"]),
                 int(r["samples"])) for r in csv.DictReader(f)}


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_point(args, lambda_nu: float, W: float, be, spin_blocks: bool) -> dict:
    params = {"delta": lambda_nu, "rashba": 0.0}
    seeds = point_seeds(args.seed, lambda_nu, W, args.samples)

    if args.invariant == "c_up":  # refinement of flagged samples happens inside run_spin_chern
        res = run_spin_chern(args.size, W, args.nk, seeds, be, delta=lambda_nu,
                             refine_nk=args.refine_nk or None)
        z2, reliable, gap = res.z2, res.reliable, res.gap_up
        t_gpu, t_refine = res.timings["first_pass"], res.timings["refine"]
        redo = range(res.timings["n_refined"])
    else:
        t0 = time.perf_counter()
        res = run_batch(args.size, W, args.nk, seeds, be, check=True, spin_blocks=spin_blocks, **params)
        t_gpu = time.perf_counter() - t0
        z2, reliable, gap = res.z2.copy(), res.reliable.copy(), res.band_gap.copy()

        # second pass: unreliable answers (not gap-closed ones: a finer mesh won't open a gap)
        redo = np.where(~reliable & (gap > 0))[0]
        t_refine = 0.0
        if len(redo) and args.refine_nk:
            t0 = time.perf_counter()
            fine = run_batch(args.size, W, args.refine_nk, [seeds[i] for i in redo], be, check=True,
                             spin_blocks=spin_blocks, **params)
            t_refine = time.perf_counter() - t0
            z2[redo], reliable[redo] = fine.z2, fine.reliable

    n_rel = int(reliable.sum())
    n_top = int(z2[reliable].sum())
    p = n_top / n_rel if n_rel else float("nan")
    return {
        "lambda_nu": lambda_nu, "W": W, "supercell": args.size, "nk": args.nk,
        "refine_nk": args.refine_nk, "samples": args.samples, "precision": args.precision,
        "spin_blocks": spin_blocks, "n_reliable": n_rel, "n_topological": n_top,
        "frac_topological": f"{p:.4f}",
        "stderr": f"{np.sqrt(p * (1 - p) / n_rel):.4f}" if n_rel else "nan",
        "n_unreliable": int((~reliable).sum()), "n_gap_closed": int((gap <= 0).sum()),
        "n_refined": len(redo), "t_gpu": f"{t_gpu:.3f}", "t_refine": f"{t_refine:.3f}",
        "invariant": args.invariant,
        "device": be.device.type,
        "gpu": torch.cuda.get_device_name(0) if be.is_gpu else "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _cpu_c_up_task(task: tuple) -> int:
    """One sample of the same C_up algorithm in plain NumPy (analysis/stage0_chern_up.py)."""
    import importlib.util

    from threadpoolctl import threadpool_limits

    from gpu_engine.reference import disordered_kane_mele

    size, W, nk, lambda_nu, seed = task
    spec = importlib.util.spec_from_file_location(
        "stage0_chern_up", Path(__file__).resolve().parent / "stage0_chern_up.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with threadpool_limits(1):  # one BLAS thread per worker process
        model = disordered_kane_mele(size, W, np.random.default_rng(seed), delta=lambda_nu,
                                     t=1.0, soc=0.3, rashba=0.0)
        return mod.c_up(model, nk)[0]


def cpu_c_up_baseline(args) -> float:
    """Seconds per sample for OUR C_up algorithm in NumPy on all CPU cores (no refinement):
    the fair 'same algorithm, best CPU' comparison next to PythTB as-is."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(args.size, W, args.nk, ld, s) for ld in args.deltas for W in args.cpu_baseline_W
             for s in range(args.cpu_baseline_samples)]
    with ProcessPoolExecutor(os.cpu_count(), mp_context=mp.get_context("fork")) as pool:
        list(pool.map(_cpu_c_up_task, tasks[: os.cpu_count()]))  # warm-up every worker
        t0 = time.perf_counter()
        list(pool.map(_cpu_c_up_task, tasks))
        return (time.perf_counter() - t0) / len(tasks)


def cpu_baseline(args, n_points: int, total_samples: int, gpu_seconds: float) -> None:
    """Time PythTB (all cores) on a few samples, same machine; extrapolate to the sweep."""
    from cpu_baseline import bench_parallel  # benchmarks/, on the path above

    per_sample = []
    for lambda_nu in args.deltas:
        for W in args.cpu_baseline_W:
            t0 = time.perf_counter()
            bench_parallel(args.size, args.nk, W, args.cpu_baseline_samples, args.seed,
                           workers=os.cpu_count(), reliability=False,
                           params={"delta": lambda_nu, "rashba": 0.0})
            per_sample.append((time.perf_counter() - t0) / args.cpu_baseline_samples)
    # wall time incl. process start-up, so slightly pessimistic for the GPU's advantage
    cpu_total = float(np.mean(per_sample)) * total_samples
    log(f"CPU baseline: PythTB on {os.cpu_count()} cores, {np.mean(per_sample):.3f} s per sample "
        f"(from {len(per_sample)} points x {args.cpu_baseline_samples} samples)")
    log(f"whole sweep ({n_points} points, {total_samples} samples): GPU {gpu_seconds / 60:.1f} min "
        f"measured vs PythTB ~{cpu_total / 3600:.1f} h estimated -> ~{cpu_total / gpu_seconds:.0f}x")
    if args.invariant == "c_up":
        same = cpu_c_up_baseline(args)
        same_total = same * total_samples
        log(f"CPU, same C_up algorithm in NumPy on {os.cpu_count()} cores: {same:.4f} s per sample "
            f"-> whole sweep ~{same_total / 3600:.2f} h vs GPU {gpu_seconds / 60:.1f} min "
            f"-> ~{same_total / gpu_seconds:.1f}x (excludes refinement on both sides)")
        append_row(args.out.with_name(args.out.stem + "_cpu_estimate.csv"), {
            "lambda_nu": "all", "W": "all", "supercell": args.size, "nk": args.nk,
            "samples": total_samples, "t_gpu": f"{gpu_seconds:.1f}", "t_refine": f"{same_total:.1f}",
            "precision": f"cpu C_up numpy s/sample {same:.4f}", "invariant": "c_up_cpu",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    append_row(args.out.with_name(args.out.stem + "_cpu_estimate.csv"), {
        "lambda_nu": "all", "W": "all", "supercell": args.size, "nk": args.nk, "samples": total_samples,
        "t_gpu": f"{gpu_seconds:.1f}", "t_refine": f"{cpu_total:.1f}",
        "precision": f"cpu s/sample {np.mean(per_sample):.4f}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--size", type=int, default=8, help="supercell n (n x n)")
    p.add_argument("--deltas", type=float, nargs="+", default=[1.0, 1.65], help="lambda_nu values")
    p.add_argument("--W-min", type=float, default=0.0)
    p.add_argument("--W-max", type=float, default=10.0)
    p.add_argument("--W-step", type=float, default=0.25)
    p.add_argument("--samples", type=int, default=100, help="disorder configurations per point")
    p.add_argument("--nk", type=int, default=None, help="default: scaled_nk(size)")
    p.add_argument("--refine-nk", type=int, default=None, help="finer mesh for the 2nd pass (default 2*nk; 0 = off)")
    p.add_argument("--precision", choices=["single", "double"], default="single")
    p.add_argument("--spin-blocks", choices=["auto", "on", "off"], default="auto",
                   help="only for --invariant z2 (c_up always uses the spin-up block)")
    p.add_argument("--invariant", choices=["c_up", "z2"], default="c_up",
                   help="c_up: spin-up Chern number (default); z2: gap-tracking pipeline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cpu-baseline", action="store_true", help="also time PythTB on a subset")
    p.add_argument("--cpu-baseline-W", type=float, nargs="+", default=[2.0, 6.0])
    p.add_argument("--cpu-baseline-samples", type=int, default=12)
    args = p.parse_args()
    args.nk = args.nk or scaled_nk(args.size)
    if args.refine_nk is None:  # c_up: nk = 10 was enough in stage 0b; z2: twice the mesh
        args.refine_nk = 10 if args.invariant == "c_up" else 2 * args.nk

    be = get_backend(args.precision)
    spin_blocks = {"on": True, "off": False, "auto": args.size >= 12}[args.spin_blocks]
    Ws = np.round(np.arange(args.W_min, args.W_max + 1e-9, args.W_step), 4)
    done = done_points(args.out)
    hw = hardware_info()
    log(f"sweep {args.size}x{args.size}, mesh {args.nk} (refine {args.refine_nk}), lambda_nu {args.deltas}, "
        f"{len(Ws)} W values x {args.samples} samples, {args.precision}, invariant {args.invariant}, "
        f"spin blocks {spin_blocks}, "
        f"on {be.describe()} / CPU {hw['cpu']} ({hw['n_cpus']})")
    if done:
        log(f"resuming: {len(done)} points already in {args.out}")

    if args.invariant == "c_up":  # warm-up (kernel loading, allocation)
        run_spin_chern(args.size, 1.0, args.nk, [0], be, delta=args.deltas[0], refine_nk=None)
    else:
        run_batch(args.size, 1.0, args.nk, [0], be, check=False, spin_blocks=spin_blocks, rashba=0.0)
    gpu_seconds, n_points = 0.0, 0
    for lambda_nu in args.deltas:
        for W in Ws:
            key = (lambda_nu, float(W), args.size, args.nk, args.samples)
            if key in done:
                continue
            row = run_point(args, lambda_nu, float(W), be, spin_blocks)
            append_row(args.out, row)
            gpu_seconds += float(row["t_gpu"]) + float(row["t_refine"])
            n_points += 1
            log(f"lambda_nu={lambda_nu:.2f} W={W:5.2f}: topological {row['n_topological']}/"
                f"{row['n_reliable']} = {row['frac_topological']} +- {row['stderr']}  "
                f"(unreliable {row['n_unreliable']}, gap closed {row['n_gap_closed']}, "
                f"refined {row['n_refined']})  {float(row['t_gpu']) + float(row['t_refine']):.1f}s")
    log(f"done: {n_points} new points in {gpu_seconds / 60:.1f} min of GPU time -> {args.out}")

    if args.cpu_baseline and n_points:
        cpu_baseline(args, n_points, n_points * args.samples, gpu_seconds)


if __name__ == "__main__":
    main()
