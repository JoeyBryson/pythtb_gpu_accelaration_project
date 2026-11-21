"""Benchmark: spin-up Chern number pipeline (gpu_engine.spin_chern), seconds per config.

Production parameters (lambda_nu = 1.65, lambda_SO = 0.3, lambda_R = 0), W near the TAI
transition. For each size and precision: time per config, fraction refined, and agreement
of C_up with double precision. Timings include host<->device copies and refinement.

Example (from hackathon/):
  uv run python benchmarks/gpu_spin_chern.py --sizes 8 12 16 --configs 40 --W 2.0
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from common import hardware_info, save_rows

from gpu_engine import get_backend
from gpu_engine.spin_chern import run_spin_chern


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--configs", type=int, default=40)
    p.add_argument("--W", type=float, default=2.0)
    p.add_argument("--nk", type=int, default=6)
    p.add_argument("--refine-nk", type=int, default=10)
    p.add_argument("--devices", nargs="+", default=None, help="default: cuda if available, else cpu")
    args = p.parse_args()

    devices = args.devices or (["cuda"] if torch.cuda.is_available() else ["cpu"])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    hw = hardware_info()
    print(f"spin-up Chern pipeline on CPU {hw['cpu']} ({hw['n_cpus']} cpus), GPU {gpu}\n")

    rows = []
    seeds = list(range(args.configs))
    for n in args.sizes:
        for device in devices:
            ref = None
            for precision in ("double", "single"):
                be = get_backend(precision, device)
                run_spin_chern(n, args.W, args.nk, seeds[:2], be, delta=1.65, refine_nk=None)  # warm-up
                res = run_spin_chern(n, args.W, args.nk, seeds, be, delta=1.65, refine_nk=args.refine_nk)
                t = res.timings
                if precision == "double":
                    ref = res.c_up
                agree = int((res.c_up == ref).sum())
                print(
                    f"L={n:2d} {device:4s} {precision:6s} {t['total']:8.2f} s "
                    f"({t['total'] / len(seeds):.3f} s/config; refined {t['n_refined']}/{len(seeds)}) "
                    f"f_top={res.z2[res.reliable].mean() if res.reliable.any() else float('nan'):.2f} "
                    f"unreliable={1 - res.reliable.mean():.2f} C_up agrees with double {agree}/{len(seeds)}"
                )
                rows.append({
                    "supercell": n, "nk": args.nk, "refine_nk": args.refine_nk, "configs": len(seeds),
                    "W": args.W, "device": device, "gpu": gpu if device == "cuda" else "",
                    "precision": precision, "t_total": f"{t['total']:.4f}",
                    "t_first_pass": f"{t['first_pass']:.4f}", "t_refine": f"{t['refine']:.4f}",
                    "s_per_config": f"{t['total'] / len(seeds):.5f}", "n_refined": t["n_refined"],
                    "frac_unreliable": f"{1 - res.reliable.mean():.3f}",
                    "c_up_agree_double": agree,
                })
        print()
    print(f"saved to {save_rows('spin_chern_pipeline', rows)}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
