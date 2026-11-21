"""Where WFArray.solve_model spends its time, and why (PythTB 2.0, CPU).

Written at the hackathon (21 Nov 2026). Three parts, clean Kane-Mele supercells, 24x24 mesh:

  1. line profile of solve_model and the PythTB functions it calls (needs line_profiler)
  2. variants of the diagonalisation step on the same H(k) stack: eigh vs eigvalsh (cost of
     eigenvectors), 1 BLAS thread vs default, SciPy occupied-only eigh, the Hermiticity check
  3. does torch's CPU eigh use more than one thread? (the batched engine relies on it)

Run (from hackathon/):
  uv run --with line_profiler --with scipy python benchmarks/profile_solve_model.py
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python benchmarks/profile_solve_model.py --part 2
      (the second command gives the "1 BLAS thread" eigh number: Accelerate reads the thread
       count only at process start)
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from common import hardware_info
from pythtb import Mesh, TBModel, WFArray

from gpu_engine.batched import clean_hamiltonians
from gpu_engine.reference import clean_kane_mele

NK = 24
SIZES = (1, 3, 6, 8)


def best_of(f, repeats: int = 3) -> float:
    f()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def k_grid(nk: int) -> np.ndarray:
    k1 = np.arange(nk) / nk
    return np.stack(np.meshgrid(k1, k1, indexing="ij"), -1).reshape(-1, 2)  # (nk*nk, 2)


def part1_line_profile() -> None:
    if importlib.util.find_spec("line_profiler") is None:
        print("1. skipped: run with `uv run --with line_profiler ...`\n")
        return
    from line_profiler import LineProfiler

    print(f"1. Line profile of solve_model, {NK}x{NK} mesh (ms; lines >= 1% of the total shown)")
    for n in SIZES:
        model = clean_kane_mele(n)
        mesh = Mesh(["k", "k"])
        mesh.build_grid(shape=(NK, NK))
        wfa = WFArray(model.lattice, mesh, spinful=True)
        wfa.solve_model(model=model)  # warm-up
        lp = LineProfiler()
        funcs = (WFArray.solve_model, TBModel.solve_ham, TBModel._sol_ham,
                 TBModel._hamiltonian_periodic, WFArray.set_states)
        for f in funcs:
            lp.add_function(f)
        lp.runcall(wfa.solve_model, model=model)
        stats = lp.get_stats()
        timings = {}
        total = 0.0
        for (_fname, _lineno, func), lines in stats.timings.items():
            func = func.split(".")[-1]  # newer line_profiler reports "Class.method"
            t = {ln: tt * stats.unit for ln, _, tt in lines}
            timings[func] = t
            if func == "solve_model":
                total = sum(t.values())
        src = {f.__name__: Path(f.__code__.co_filename).read_text().splitlines() for f in funcs}
        print(f"   --- {n}x{n} (M={4 * n * n}): total {1e3 * total:.1f} ms")
        for func in ("_sol_ham", "_hamiltonian_periodic", "set_states"):
            for ln, t in sorted(timings.get(func, {}).items(), key=lambda kv: -kv[1]):
                if t >= 0.01 * total:
                    code_line = src[func][ln - 1].strip()[:70]
                    print(f"   {func:22s} {1e3 * t:9.1f} ms {100 * t / total:5.1f}%  {code_line}")
    print()


def part2_eigh_variants() -> None:
    print(f"2. Diagonalisation variants on the same H(k) stack, {NK}x{NK} mesh (s, best of 3)")
    have_scipy = importlib.util.find_spec("scipy") is not None
    for n in (3, 8):
        H = clean_kane_mele(n).hamiltonian(k_grid(NK), flatten_spin_axis=True)
        M = H.shape[-1]
        row = {
            "eigh": best_of(lambda H=H: np.linalg.eigh(H)),
            "eigvalsh": best_of(lambda H=H: np.linalg.eigvalsh(H)),
            "hermiticity check": best_of(lambda H=H: np.allclose(H, H.swapaxes(-1, -2).conj())),
        }
        if have_scipy:
            import scipy.linalg as sl

            row["scipy occupied-only (loop)"] = best_of(
                lambda H=H, M=M: [sl.eigh(h, subset_by_index=[0, M // 2 - 1], driver="evr") for h in H], 1
            )
        print(f"   {n}x{n} (M={M}, {H.shape[0]} matrices, H = {H.nbytes / 1e6:.0f} MB): "
              + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    print()


def part3_torch_threads() -> None:
    print("3. torch CPU eigh vs thread count: 6x6, 6 configs x 8x8 mesh = 384 matrices of 144x144 (s)")
    H = np.concatenate([clean_hamiltonians(clean_kane_mele(6), 8)] * 6)
    Ht = torch.as_tensor(H)
    default = torch.get_num_threads()
    print(f"   numpy eigh c128: {best_of(lambda: np.linalg.eigh(H), 1):.3f}")
    for nt in (default, 1):
        torch.set_num_threads(nt)
        t128 = best_of(lambda: torch.linalg.eigh(Ht), 1)
        t64 = best_of(lambda: torch.linalg.eigh(Ht.to(torch.complex64)), 1)
        print(f"   torch {nt:2d} thread(s): eigh c128 {t128:.3f}  c64 {t64:.3f}")
    torch.set_num_threads(default)
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--part", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    hw = hardware_info()
    print(f"CPU {hw['cpu']} ({hw['n_cpus']} cpus, BLAS {hw['blas']}, numpy {hw['numpy']}), "
          f"torch {torch.__version__}\n")
    parts = {1: part1_line_profile, 2: part2_eigh_variants, 3: part3_torch_threads}
    for k in args.part:
        parts[k]()


if __name__ == "__main__":
    main()
