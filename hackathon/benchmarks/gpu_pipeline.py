"""Full-pipeline benchmark: many disorder configurations -> Z2 each, CPU vs GPU.

Everything on the SAME machine, the SAME configurations (same seeds):
  pythtb serial    PythTB 2.0 in a Python loop (what users do today)
  pythtb parallel  PythTB 2.0, one configuration per CPU core (the strongest simple CPU baseline)
  ours             batched pipeline (gpu_engine.pipeline.run_batch) on CPU and GPU,
                   double and single precision, incl. all host<->device copies

A speed-up is only reported alongside whether every configuration's Z2 agrees with PythTB.
Reliability checks are excluded from all timings (both sides): PythTB serial and parallel are
timed with run_config(check=False) and get their flags from one untimed pass
(cpu_baseline.reliability_pass); ours is timed by run_batch, which times its check separately.
Before 21 Nov ~20:00 the PythTB parallel wall time included the checks (~15% on KAG).

Example (from hackathon/):
  uv run python benchmarks/gpu_pipeline.py --sizes 6 8 --configs 40 --W 2.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from common import Progress, hardware_info, log, save_rows, scaled_nk
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
    p.add_argument("--no-reliability", action="store_true",
                   help="skip PythTB's untimed reliability pass (saves one parallel run)")
    p.add_argument("--precisions", nargs="+", default=["double", "single"],
                   choices=["double", "single"], help="which precisions to run ours in")
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"],
                   help="where to run ours: cpu, cuda or both (cuda skipped if unavailable)")
    args = p.parse_args()

    devices = [d for d in args.devices if d == "cpu" or torch.cuda.is_available()]
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
            res = bench_serial(*common_args, workers=args.workers,
                               reliability=not args.no_reliability)
            baselines["pythtb_serial"] = (sum(r.t_total for r in res), [r.z2 for r in res])
        res = bench_parallel(*common_args, workers=args.workers,
                             reliability=not args.no_reliability)
        # wall time of the parallel run is printed by bench_parallel; recompute from rows
        wall = float(open_last_row("cpu_parallel")["wall_s"])
        baselines["pythtb_parallel"] = (wall, [r.z2 for r in res])
        ref_z2 = np.array(baselines["pythtb_parallel"][1])

        for device in devices:
            for precision in args.precisions:
                be = get_backend(precision, device)
                log(f"ours {device} {precision}: {args.configs} configs of {n}x{n}, mesh {nk}x{nk}")
                run_batch(n, args.W, nk, seeds[:1], be, check=False)  # warm-up
                progress = Progress(f"ours {device} {precision} {n}x{n}", args.configs, every=1)
                # The reliability check (an SVD of every overlap) is untimed but slow in double
                # precision on a T4, and its answer doesn't depend on precision: run it once,
                # with the single-precision pass only.
                out = run_batch(n, args.W, nk, seeds, be, check=(precision == "single"),
                                progress=progress.step)
                t = out.timings["total"]
                checked = not np.isnan(out.min_overlap).all()
                n_unrel = int((~out.reliable).sum()) if checked else "not checked"
                agree = int((out.z2 == ref_z2).sum())
                line = (
                    f"  ours {device:4s} {precision:6s} {t:8.3f} s  "
                    f"(solve {out.timings['solve']:.3f}, wilson {out.timings['wilson']:.3f}, "
                    f"z2 {out.timings['z2']:.3f})  Z2 agrees {agree}/{args.configs}  "
                    f"unreliable {n_unrel}  NS fallbacks {out.ns_fallbacks}"
                    f"\n      wilson breakdown: "
                    + "  ".join(
                        f"{k[2:]} {out.timings[k]:.3f}"
                        for k in ("t_overlaps", "t_unitary", "t_product", "t_eigvals")
                    )
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
                    **{k: f"{out.timings[k]:.4f}"
                       for k in ("t_overlaps", "t_unitary", "t_product", "t_eigvals")},
                    "z2_agree": agree, "n_unreliable": n_unrel,
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
