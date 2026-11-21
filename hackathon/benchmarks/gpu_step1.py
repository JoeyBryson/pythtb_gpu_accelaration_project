"""Step-1 benchmark: diagonalising H(k) for many disorder configurations.

Compares, on the SAME machine and the SAME configurations (same seeds):
  pythtb   PythTB 2.0, one configuration after another: build model + solve_model
  ours     batched: build H_clean once, add each config's random diagonal, one eigh call
           (CPU and/or GPU, double and single precision). Includes host->device copies.

Also checks every batched result against PythTB (eigenvalues), so a speed-up is only
reported if the answers agree.

Example (from hackathon/):
  uv run python benchmarks/gpu_step1.py --sizes 4 8 --configs 40
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from common import hardware_info, save_rows, scaled_nk

from gpu_engine import get_backend, sync
from gpu_engine.batched import solve_configs
from gpu_engine.reference import clean_kane_mele, disordered_kane_mele, solve


def time_pythtb(n: int, nk: int, W: float, seeds: list[int]) -> tuple[float, list[np.ndarray]]:
    """PythTB, serial: what a user does today. Returns (seconds, energies per config)."""
    t0 = time.perf_counter()
    energies = []
    for s in seeds:
        model = disordered_kane_mele(n, W, np.random.default_rng(s))
        energies.append(np.asarray(solve(model, nk).energies))
    return time.perf_counter() - t0, energies


def time_ours(n: int, nk: int, W: float, seeds: list[int], device: str, precision: str):
    """Batched solve, timed end to end (incl. building H and copying to the device)."""
    be = get_backend(precision, device)
    model = clean_kane_mele(n)
    solve_configs(model, nk, W, seeds[:1], be)  # warm-up (kernel loading, allocation)
    sync(be)
    t0 = time.perf_counter()
    energies, _ = solve_configs(model, nk, W, seeds, be)
    energies = energies.cpu()  # copy back: part of the honest end-to-end cost
    sync(be)
    return time.perf_counter() - t0, energies.to(torch.float64).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    p.add_argument("--configs", type=int, default=40)
    p.add_argument("--W", type=float, default=2.0)
    p.add_argument("--nk", type=int, default=None, help="default: scaled_nk(size)")
    args = p.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    hw = hardware_info()
    print(f"step-1 benchmark on CPU {hw['cpu']} ({hw['n_cpus']} cpus), GPU {gpu}\n")

    rows = []
    for n in args.sizes:
        nk = args.nk or scaled_nk(n)
        seeds = list(range(args.configs))
        t_ref, e_ref = time_pythtb(n, nk, args.W, seeds)
        e_ref = np.stack(e_ref)
        print(f"{n}x{n}, mesh {nk}x{nk}, {args.configs} configs: PythTB serial {t_ref:8.3f} s")

        for device in devices:
            for precision in ("double", "single"):
                t, e = time_ours(n, nk, args.W, seeds, device, precision)
                err = float(np.abs(e - e_ref).max())
                tol = 1e-8 if precision == "double" else 1e-3
                ok = err < tol
                print(
                    f"    ours {device:4s} {precision:6s} {t:8.3f} s   "
                    f"speed-up {t_ref / t:7.1f}x   max eigenvalue error {err:.1e} "
                    f"{'OK' if ok else 'MISMATCH'}"
                )
                rows.append({
                    "supercell": n, "nk": nk, "configs": args.configs, "W": args.W,
                    "device": device, "gpu": gpu if device == "cuda" else "",
                    "precision": precision, "t_pythtb_serial": f"{t_ref:.4f}",
                    "t_ours": f"{t:.4f}", "speedup_vs_pythtb_serial": f"{t_ref / t:.2f}",
                    "max_eig_error": f"{err:.2e}", "answers_match": ok,
                })
        print()
    print(f"saved to {save_rows('step1_batched_solve', rows)}")


if __name__ == "__main__":
    main()
