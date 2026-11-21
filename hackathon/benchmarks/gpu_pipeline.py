"""Full-pipeline benchmark: many disorder configurations -> Z2 each, CPU vs GPU.

Everything on the SAME machine, the SAME configurations (same seeds):
  pythtb serial    PythTB 2.0 in a Python loop (what users do today)
  pythtb parallel  PythTB 2.0, one configuration per CPU core (the strongest simple CPU baseline)
  ours             batched pipeline (gpu_engine.pipeline.run_batch) on CPU and GPU,
                   double and single precision, incl. all host<->device copies

A speed-up is only reported alongside whether every configuration's Z2 agrees with PythTB.
Reliability checks are excluded from all timings (both sides).

Example (from hackathon/):
  uv run python benchmarks/gpu_pipeline.py --sizes 6 8 --configs 40 --W 2.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from common import hardware_info, save_rows, scaled_nk
from cpu_baseline import bench_parallel, bench_serial, config_seeds

from gpu_engine import get_backend
from gpu_engine.pipeline import run_batch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[6, 8])
    p.add_argument("--configs", type=int, default=40)
    p.add_argument("--W", type=float, default=2.0)
    p.add_argument("--nk", type=int, default=None, help="default: scaled_nk(size)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--skip-serial", action="store_true", help="skip the slow PythTB serial loop")
    args = p.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    hw = hardware_info()
    print(f"full pipeline on CPU {hw['cpu']} ({hw['n_cpus']} cpus), GPU {gpu}\n")

    rows = []
    for n in args.sizes:
        nk = args.nk or scaled_nk(n)
        seeds = config_seeds(0, args.configs)
        common_args = (n, nk, args.W, args.configs, 0)

        baselines = {}
        if not args.skip_serial:
            res = bench_serial(*common_args)
            baselines["pythtb_serial"] = (sum(r.t_total for r in res), [r.z2 for r in res])
        res = bench_parallel(*common_args, workers=args.workers)
        # wall time of the parallel run is printed by bench_parallel; recompute from rows
        wall = float(open_last_row("cpu_parallel")["wall_s"])
        baselines["pythtb_parallel"] = (wall, [r.z2 for r in res])
        ref_z2 = np.array(baselines["pythtb_parallel"][1])

        for device in devices:
            for precision in ("double", "single"):
                be = get_backend(precision, device)
                run_batch(n, args.W, nk, seeds[:1], be, check=False)  # warm-up
                out = run_batch(n, args.W, nk, seeds, be, check=True)
                t = out.timings["total"]
                agree = int((out.z2 == ref_z2).sum())
                line = (
                    f"  ours {device:4s} {precision:6s} {t:8.3f} s  "
                    f"(solve {out.timings['solve']:.3f}, wilson {out.timings['wilson']:.3f}, "
                    f"z2 {out.timings['z2']:.3f})  Z2 agrees {agree}/{args.configs}  "
                    f"unreliable {int((~out.reliable).sum())}  NS fallbacks {out.ns_fallbacks}"
                )
                for name, (t_ref, _) in baselines.items():
                    line += f"\n      vs {name:15s} {t_ref:8.3f} s -> {t_ref / t:6.1f}x"
                print(line)
                row = {
                    "supercell": n, "nk": nk, "configs": args.configs, "W": args.W,
                    "device": device, "gpu": gpu if device == "cuda" else "",
                    "precision": precision, "t_ours": f"{t:.4f}",
                    "t_solve": f"{out.timings['solve']:.4f}",
                    "t_wilson": f"{out.timings['wilson']:.4f}",
                    "t_z2": f"{out.timings['z2']:.4f}",
                    "z2_agree": agree, "n_unreliable": int((~out.reliable).sum()),
                    "ns_fallbacks": out.ns_fallbacks, "workers": args.workers,
                }
                for name, (t_ref, _) in baselines.items():
                    row[f"t_{name}"] = f"{t_ref:.4f}"
                    row[f"speedup_vs_{name}"] = f"{t_ref / t:.2f}"
                rows.append(row)
        print()
    print(f"saved to {save_rows('full_pipeline', rows)}")


def open_last_row(scenario: str) -> dict:
    import csv

    from common import RESULTS_DIR

    with (RESULTS_DIR / f"{scenario}.csv").open() as f:
        return list(csv.DictReader(f))[-1]


if __name__ == "__main__":
    main()
