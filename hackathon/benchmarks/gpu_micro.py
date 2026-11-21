"""GPU micro-benchmarks for two design choices (no PythTB baseline, so it's quick):

  solve   full H(k) vs the spin-block split (Rashba = 0), per supercell size:
          where does splitting start to pay off on this GPU?
  eig     ways to get the eigenvalues of the Wilson matrices, timed and checked against
          exact answers: our Hermitian method in double (the current default), the general
          solver on the GPU (double and single), and the general solver on the CPU.

Run from hackathon/:  uv run python benchmarks/gpu_micro.py
Results: results/micro_solve.csv, results/micro_eig.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from common import hardware_info, log, save_rows

from gpu_engine import get_backend, sync
from gpu_engine.batched import solve_configs
from gpu_engine.reference import clean_kane_mele
from gpu_engine.wilson import (
    boundary_phase,
    ordered_product,
    overlaps_along_ky,
    unitary_eigenvalues,
    unitary_part,
)

NO_RASHBA = {"rashba": 0.0}


def best_time(fn, repeats: int = 3) -> float:
    """Best of `repeats` runs after one warm-up, with the GPU synchronised."""
    fn()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return min(times)


def bench_solve(sizes: list[int], configs: int, nk: int, device: str) -> None:
    rows = []
    for n in sizes:
        model = clean_kane_mele(n, **NO_RASHBA)
        seeds = list(range(configs))
        for precision in ("single", "double"):
            be = get_backend(precision, device)
            t_full = best_time(lambda m=model, s=seeds, b=be: solve_configs(m, nk, 2.0, s, b))
            t_blk = best_time(
                lambda m=model, s=seeds, b=be: solve_configs(m, nk, 2.0, s, b, spin_blocks=True)
            )
            log(f"solve {n:2d}x{n:<2d} {precision:6s}: full {t_full:7.3f} s   spin blocks "
                f"{t_blk:7.3f} s   -> split is {t_full / t_blk:5.2f}x {'faster' if t_blk < t_full else 'SLOWER'}")
            rows.append({"supercell": n, "matrix_dim": 4 * n * n, "nk": nk, "configs": configs,
                         "device": device, "precision": precision, "t_full": f"{t_full:.4f}",
                         "t_spin_blocks": f"{t_blk:.4f}", "split_speedup": f"{t_full / t_blk:.3f}"})
    print(f"saved to {save_rows('micro_solve', rows)}")


def wilson_matrices(n: int, configs: int, nk: int, be) -> torch.Tensor:
    """Real Wilson matrices from our pipeline (spin blocks, Rashba = 0): (C, 2, nk, m, m)."""
    model = clean_kane_mele(n, **NO_RASHBA)
    _, u = solve_configs(model, nk, 2.0, list(range(configs)), be, spin_blocks=True)
    phase = torch.as_tensor(boundary_phase(model, spin_blocks=True), dtype=u.dtype, device=u.device)
    U, _ = unitary_part(overlaps_along_ky(u, phase))
    return ordered_product(U)


def phases(lam: torch.Tensor) -> np.ndarray:
    return np.sort(np.remainder(np.angle(lam.cpu().numpy()), 2 * np.pi), axis=-1)


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    d = np.abs(a - b)
    return float(np.minimum(d, 2 * np.pi - d).max() / (2 * np.pi))  # in Wannier-centre units


def bench_eig(sizes: list[int], configs: int, nk: int, device: str) -> None:
    rows = []
    for n in sizes:
        W = wilson_matrices(n, configs, nk, get_backend("single", device))  # single-precision run
        exact = phases(torch.linalg.eigvals(W.cpu().to(torch.complex128)))
        methods = {
            "hermitian double (default)": lambda W=W: unitary_eigenvalues(W)[0],
            f"general {device} complex128": lambda W=W: torch.linalg.eigvals(W.to(torch.complex128)),
            f"general {device} complex64": lambda W=W: torch.linalg.eigvals(W),
            "general cpu complex128": lambda W=W: torch.linalg.eigvals(W.cpu().to(torch.complex128)),
        }
        for name, fn in methods.items():
            t = best_time(fn)
            err = max_error(phases(fn()), exact)
            log(f"eig {n:2d}x{n:<2d} ({W.shape[:-2].numel()} matrices of {W.shape[-1]}x{W.shape[-1]}): "
                f"{name:28s} {t:7.3f} s   max error {err:.1e}")
            rows.append({"supercell": n, "n_matrices": W.shape[:-2].numel(), "m": W.shape[-1],
                         "method": name, "seconds": f"{t:.4f}", "max_wcc_error": f"{err:.2e}"})
    print(f"saved to {save_rows('micro_eig', rows)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--solve-sizes", type=int, nargs="+", default=[6, 8, 10, 12, 16])
    p.add_argument("--eig-sizes", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--configs", type=int, default=20)
    p.add_argument("--nk", type=int, default=6)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    hw = hardware_info()
    print(f"micro-benchmarks on {device} (GPU {gpu}; CPU {hw['cpu']}, {hw['n_cpus']} cpus)\n", flush=True)
    bench_solve(args.solve_sizes, args.configs, args.nk, device)
    bench_eig(args.eig_sizes, max(2, args.configs // 4), args.nk, device)


if __name__ == "__main__":
    main()
