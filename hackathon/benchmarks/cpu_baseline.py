"""CPU baseline: how long plain PythTB 2.0 takes, in four clearly defined scenarios.

Every GPU speed-up we quote is measured against one of these. See benchmarks/README.md
for what each scenario means and why it is there.

  clean     one clean supercell per size, fixed 31x31 mesh  (compare with prep runs)
  scaled    one clean supercell per size, realistic mesh      (what we actually run)
  serial    many disorder configs in a plain Python loop      (the naive baseline)
  parallel  many disorder configs spread over all CPU cores   (the strongest CPU baseline)

Timing rule: the reliability diagnostics (run_config(check=True): an SVD over every link,
~15-20% of the pipeline's cost) never run inside a timed region. Timed runs use check=False;
the z2_reliable flags come from one separate, untimed pass (see reliability_pass), shared by
serial and parallel runs of the same configurations. --no-reliability skips that pass.

Examples (from the hackathon/ folder):
  uv run python benchmarks/cpu_baseline.py clean --sizes 2 4 6 8
  uv run python benchmarks/cpu_baseline.py scaled --sizes 4 8 12 16
  uv run python benchmarks/cpu_baseline.py serial --size 6 --configs 20 --W 2.0
  uv run python benchmarks/cpu_baseline.py parallel --size 6 --configs 20 --W 2.0
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import Progress, log, print_header, save_rows, scaled_nk

from gpu_engine.reference import ConfigResult, run_config


def matrix_dim(supercell: int) -> int:
    """Size of H(k): 2 orbitals x 2 spins per unit cell, n x n cells."""
    return 4 * supercell * supercell


def warm_up() -> None:
    """First call pays one-off import and memory-allocation costs; don't time it."""
    run_config(1, W=0, nk=4)


def config_seeds(base_seed: int, configs: int) -> list[int]:
    """One independent, reproducible random seed per configuration.

    Serial and parallel runs use the same seeds, so they simulate exactly the same
    materials and must give exactly the same Z2 answers (a free correctness check).
    """
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(base_seed).spawn(configs)]


def summarise(
    results: list[ConfigResult], checked: list[ConfigResult] | None
) -> dict[str, float]:
    """Total seconds per pipeline stage (from the timed runs), plus the physics answers.

    checked: the same configurations from the untimed reliability pass (or None if skipped),
    which supplies z2_reliable. Its Z2 must equal the timed runs' Z2, config by config.
    """
    if checked is None:
        trusted, n_unreliable = [], float("nan")
    else:
        if [r.z2 for r in results] != [c.z2 for c in checked]:
            raise RuntimeError("timed and reliability passes gave different Z2: not deterministic?")
        trusted = [c.z2 for c in checked if c.z2_reliable]  # only count answers we can trust
        n_unreliable = len(results) - len(trusted)
    return {
        "t_build": sum(r.t_build for r in results),
        "t_solve": sum(r.t_solve for r in results),
        "t_wilson": sum(r.t_wilson for r in results),
        "t_z2": sum(r.t_z2 for r in results),
        "frac_topological": sum(trusted) / len(trusted) if trusted else float("nan"),
        "n_unreliable": n_unreliable,
    }


# ---------------------------------------------------------------- scenario: clean / scaled


def bench_clean_sizes(scenario: str, sizes: list[int], nk_for, repeats: int) -> None:
    """One clean supercell per size. Best of `repeats` runs (least affected by noise)."""
    rows = []
    for n in sizes:
        nk = nk_for(n)
        runs = [run_config(n, W=0, nk=nk) for _ in range(repeats)]
        best = min(runs, key=lambda r: r.t_total)
        print(
            f"{n:2d}x{n:<2d}  H {matrix_dim(n):4d}x{matrix_dim(n):<4d}  mesh {nk:2d}x{nk:<2d} | "
            f"solve {best.t_solve:8.3f} s  wilson {best.t_wilson:8.3f} s  "
            f"total {best.t_total:8.3f} s  (median {statistics.median(r.t_total for r in runs):.3f})"
            f"  Z2={best.z2}{'' if best.z2 is None or best.z2_reliable else ' (UNRELIABLE)'}"
        )
        rows.append(
            {
                "supercell": n,
                "matrix_dim": matrix_dim(n),
                "nk": nk,
                "repeats": repeats,
                "t_solve": f"{best.t_solve:.4f}",
                "t_wilson": f"{best.t_wilson:.4f}",
                "t_total": f"{best.t_total:.4f}",
                "t_total_median": f"{statistics.median(r.t_total for r in runs):.4f}",
                "z2": best.z2,
                "z2_margin": "" if best.z2_margin is None else f"{best.z2_margin:.3f}",
                "z2_reliable": best.z2_reliable,
            }
        )
    print(f"\nsaved to {save_rows(scenario, rows)}")


# ---------------------------------------------------------------- scenario: serial


def bench_serial(
    n: int, nk: int, W: float, configs: int, seed: int,
    workers: int | None = None, reliability: bool = True, params: dict | None = None,
) -> list[ConfigResult]:
    """The naive baseline: one configuration after another in a Python loop.

    Timed with check=False; `workers` processes run the untimed reliability pass.
    params: model parameters passed to run_config (e.g. {"rashba": 0.0}); default tutorial.
    """
    params = params or {}
    seeds = config_seeds(seed, configs)
    log(f"PythTB serial: {configs} configs of {n}x{n}, mesh {nk}x{nk}, W={W}")
    progress = Progress(f"PythTB serial {n}x{n}", configs)
    t0 = time.perf_counter()
    results = []
    for s in seeds:
        results.append(run_config(n, W, nk, seed=s, check=False, **params))
        progress.step()
    wall = time.perf_counter() - t0
    checked = (
        reliability_pass(n, nk, W, seeds, workers or os.cpu_count(), params) if reliability else None
    )
    report("serial", n, nk, W, configs, wall, results, checked, workers=1)
    return results


# ---------------------------------------------------------------- scenario: parallel


def _one_blas_thread() -> None:
    """Run in each worker: use 1 BLAS thread, so the workers don't fight over cores."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def _run_config_task(args: tuple) -> ConfigResult:
    """Timed worker task: the pipeline only, no reliability diagnostics."""
    n, W, nk, s, params = args
    return run_config(n, W, nk, seed=s, check=False, **params)


def _run_config_checked_task(args: tuple) -> ConfigResult:
    """Untimed worker task: the pipeline plus the reliability diagnostics."""
    n, W, nk, s, params = args
    return run_config(n, W, nk, seed=s, check=True, **params)


def _worker_pool(workers: int) -> ProcessPoolExecutor:
    """One process per core, 1 BLAS thread each, same behaviour on macOS and Linux."""
    # Apple Accelerate only reads its thread count when a process starts, so set it in
    # the environment before the worker processes are created.
    for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    return ProcessPoolExecutor(
        workers, mp_context=mp.get_context("spawn"), initializer=_one_blas_thread
    )


_RELIABILITY_CACHE: dict[tuple, list[ConfigResult]] = {}


def reliability_pass(
    n: int, nk: int, W: float, seeds: list[int], workers: int, params: dict | None = None
) -> list[ConfigResult]:
    """The z2_reliable flags for these configurations, from an UNTIMED pass with check=True.

    Kept out of every timed region because the diagnostics cost ~15-20% of the pipeline
    (21 Nov: 15% of serial wall time on KAG, ~20% on MBP). Runs on all cores and is cached,
    so serial and parallel runs of the same configurations share one pass.
    """
    params = params or {}
    key = (n, nk, W, tuple(seeds), tuple(sorted(params.items())))
    if key not in _RELIABILITY_CACHE:
        log(f"reliability pass (untimed): {len(seeds)} configs of {n}x{n} on {workers} processes")
        with _worker_pool(workers) as pool:
            tasks = [(n, W, nk, s, params) for s in seeds]
            _RELIABILITY_CACHE[key] = list(pool.map(_run_config_checked_task, tasks))
    return _RELIABILITY_CACHE[key]


def bench_parallel(
    n: int, nk: int, W: float, configs: int, seed: int, workers: int, reliability: bool = True,
    params: dict | None = None,
) -> list[ConfigResult]:
    """The strongest simple CPU baseline: configurations are independent, so give each
    CPU core its own configuration (one process per core, 1 BLAS thread each).

    This is the comparison a sceptical judge would ask for, so we report it alongside
    the serial loop. Timed with check=False (see reliability_pass).
    """
    params = params or {}
    seeds = config_seeds(seed, configs)
    tasks = [(n, W, nk, s, params) for s in seeds]
    with _worker_pool(workers) as pool:
        # warm up every worker (imports, first allocation) before starting the clock
        list(pool.map(_run_config_task, [(1, 0.0, 4, 0, {})] * workers))
        log(f"PythTB parallel: {configs} configs of {n}x{n} on {workers} processes")
        progress = Progress(f"PythTB parallel {n}x{n}", configs)
        t0 = time.perf_counter()
        results = []
        for r in pool.map(_run_config_task, tasks):  # results arrive in order as they finish
            results.append(r)
            progress.step()
        wall = time.perf_counter() - t0
    checked = reliability_pass(n, nk, W, seeds, workers, params) if reliability else None
    report("parallel", n, nk, W, configs, wall, results, checked, workers=workers)
    return results


# ---------------------------------------------------------------- reporting


def report(
    scenario: str, n: int, nk: int, W: float, configs: int, wall: float,
    results: list[ConfigResult], checked: list[ConfigResult] | None, workers: int,
) -> None:
    s = summarise(results, checked)
    print(
        f"{n}x{n} supercell (H {matrix_dim(n)}x{matrix_dim(n)}), mesh {nk}x{nk}, "
        f"{configs} configs at W={W}, {workers} worker(s)\n"
        f"  wall time   {wall:8.2f} s   ->  {configs / wall:7.2f} configs/s  "
        f"({wall / configs:.3f} s per config)\n"
        f"  stage time  build {s['t_build']:.2f}  solve {s['t_solve']:.2f}  "
        f"wilson {s['t_wilson']:.2f}  z2 {s['t_z2']:.3f}  (summed over configs)\n"
        + (
            f"  physics     fraction topological = {s['frac_topological']:.2f}  "
            f"({s['n_unreliable']} of {configs} Z2 answers flagged unreliable and left out)"
            if checked is not None
            else "  physics     reliability pass skipped (--no-reliability)"
        )
    )
    row = {
        "supercell": n, "matrix_dim": matrix_dim(n), "nk": nk, "W": W, "configs": configs,
        "workers": workers, "wall_s": f"{wall:.4f}", "configs_per_s": f"{configs / wall:.4f}",
        **{k: f"{v:.4f}" for k, v in s.items()},
    }
    print(f"  saved to {save_rows(f'cpu_{scenario}', [row])}\n")


# ---------------------------------------------------------------- command line


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="scenario", required=True)

    clean = sub.add_parser("clean", help="fixed 31x31 mesh, clean model")
    clean.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6, 8])
    clean.add_argument("--nk", type=int, default=31, help="31 matches the prep runs in CLAUDE.md")
    clean.add_argument("--repeats", type=int, default=3)

    scaled = sub.add_parser("scaled", help="realistic mesh per size, clean model")
    scaled.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 16])
    scaled.add_argument("--nk-unit-cell", type=int, default=48)
    scaled.add_argument("--repeats", type=int, default=3)

    for name in ("serial", "parallel", "both"):
        d = sub.add_parser(name, help=f"{name}: many disorder configurations")
        d.add_argument("--size", type=int, default=6, help="supercell n (n x n)")
        d.add_argument("--nk", type=int, default=None, help="default: scaled_nk(size)")
        d.add_argument("--W", type=float, default=2.0, help="disorder strength")
        d.add_argument("--configs", type=int, default=20)
        d.add_argument("--seed", type=int, default=0)
        d.add_argument("--workers", type=int, default=os.cpu_count())
        d.add_argument("--no-reliability", action="store_true",
                       help="skip the untimed reliability pass (no z2_reliable flags)")

    args = p.parse_args()
    warm_up()

    if args.scenario == "clean":
        print_header("clean", f"clean model, fixed {args.nk}x{args.nk} mesh")
        bench_clean_sizes("cpu_clean", args.sizes, lambda _n: args.nk, args.repeats)
    elif args.scenario == "scaled":
        print_header("scaled", f"clean model, mesh ~ {args.nk_unit_cell}/n per direction")
        bench_clean_sizes(
            "cpu_scaled", args.sizes, lambda n: scaled_nk(n, args.nk_unit_cell), args.repeats
        )
    else:
        nk = args.nk or scaled_nk(args.size)
        common_args = (args.size, nk, args.W, args.configs, args.seed)
        if args.scenario in ("serial", "both"):
            print_header("serial", "disorder configs in a Python loop")
            serial = bench_serial(
                *common_args, workers=args.workers, reliability=not args.no_reliability
            )
        if args.scenario in ("parallel", "both"):
            print_header("parallel", f"disorder configs over {args.workers} processes")
            parallel = bench_parallel(
                *common_args, workers=args.workers, reliability=not args.no_reliability
            )
        if args.scenario == "both":
            same = [r.z2 for r in serial] == [r.z2 for r in parallel]
            print(f"serial and parallel give identical Z2 for every config: {same}")


if __name__ == "__main__":
    main()
