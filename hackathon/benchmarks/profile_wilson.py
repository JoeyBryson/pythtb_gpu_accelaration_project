"""Where the time goes in the batched Wilson loop (gpu_engine/wilson.py), vs PythTB's berry_phase.

Written at the hackathon (21 Nov 2026). For a few supercell sizes, takes C disorder configs
(same seeds as the other benchmarks), solves them once with batched.solve_configs, then times:

  pythtb      reference.wannier_centres (WFArray.berry_phase), one config after another
  overlaps    wilson.overlaps_along_ky       M = <u(k)|u(k+dk)>, all lines and configs
  NS          Newton-Schulz, NS_MAX_STEPS fixed steps (as wilson.unitary_part does)
  check       the convergence check ||X^H X - I|| after NS
  SVD         the alternative unitary part, U = V @ Wh
  product     wilson.ordered_product (tree)
  eigvals     torch.linalg.eigvals of the Wilson matrices

plus how many Newton-Schulz steps the real overlaps actually need, what stopping early at
3 or 4 steps would save (time and max |U - U_svd|), and the effect of torch's CPU thread count. Example (from hackathon/):

  uv run python benchmarks/profile_wilson.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from common import hardware_info
from cpu_baseline import config_seeds
from pythtb import Mesh, WFArray

from gpu_engine import get_backend, sync
from gpu_engine.batched import solve_configs
from gpu_engine.reference import clean_kane_mele
from gpu_engine.reference import wannier_centres as pythtb_wannier_centres
from gpu_engine.wilson import (
    NS_MAX_STEPS,
    NS_TOL,
    boundary_phase,
    ordered_product,
    overlaps_along_ky,
    unitary_part_svd,
)

CASES = [(3, 16), (6, 8), (8, 6)]  # (supercell, nk) = scaled_nk meshes used in the sweeps
CONFIGS, W, SEED = 6, 4.0, 0


def best_of(f, be, repeats: int = 3) -> float:
    f()
    sync(be)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        sync(be)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def newton_schulz(M: torch.Tensor, steps: int) -> torch.Tensor:
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    X = M
    for _ in range(steps):
        X = 0.5 * X @ (3 * eye - X.mH @ X)
    return X


def residual(X: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
    return torch.linalg.matrix_norm(X.mH @ X - eye, ord="fro")


def pythtb_time(model, nk: int, u: np.ndarray) -> float:
    """berry_phase for each config, on the same states (u: (C, nk, nk, n_occ, N))."""
    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk), gamma_centered=True)
    n_occ = u.shape[-2]
    total = 0.0
    for uc in u:
        wfa = WFArray(model.lattice, mesh, nstates=n_occ, spinful=True)
        wfa.set_states(uc.reshape(nk, nk, n_occ, model.norb, 2), is_cell_periodic=True)
        pythtb_wannier_centres(wfa, n_occ)  # warm caches
        t0 = time.perf_counter()
        pythtb_wannier_centres(wfa, n_occ)
        total += time.perf_counter() - t0
    return total


def main() -> None:
    hw = hardware_info()
    print(f"CPU {hw['cpu']} ({hw['n_cpus']} cpus, BLAS {hw['blas']}), torch {torch.__version__}, "
          f"torch threads {torch.get_num_threads()}, GPU "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"{CONFIGS} configs at W={W} (team convention), Wilson loop along k_y, "
          f"NS_MAX_STEPS={NS_MAX_STEPS}\n")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    for n, nk in CASES:
        model = clean_kane_mele(n)
        seeds = config_seeds(SEED, CONFIGS)
        n_occ = model.norb
        print(f"=== {n}x{n} supercell, mesh {nk}x{nk}: {CONFIGS * nk * nk} links of "
              f"{n_occ}x{n_occ}, {CONFIGS * nk} Wilson matrices")
        be64 = get_backend("double", "cpu")
        _, u64 = solve_configs(model, nk, W, seeds, be64)
        t_py = pythtb_time(model, nk, u64.numpy())
        print(f"  PythTB berry_phase (per-line loop, SVD):  {t_py:.3f} s")

        for device in devices:
            for precision in ("double", "single"):
                be = get_backend(precision, device)
                u = u64.to(device=be.device, dtype=be.cdtype)
                phase = torch.as_tensor(boundary_phase(model), dtype=be.cdtype, device=be.device)
                M = overlaps_along_ky(u, phase)
                X = newton_schulz(M, NS_MAX_STEPS)
                Wm = ordered_product(X)
                t = {
                    "overlaps": best_of(lambda u=u, p=phase: overlaps_along_ky(u, p), be),
                    "NS": best_of(lambda M=M: newton_schulz(M, NS_MAX_STEPS), be),
                    "check": best_of(lambda X=X: residual(X), be),
                    "SVD": best_of(lambda M=M: unitary_part_svd(M), be),
                    "product": best_of(lambda X=X: ordered_product(X), be),
                    "eigvals": best_of(lambda Wm=Wm: torch.linalg.eigvals(Wm), be),
                }
                ns_total = t["overlaps"] + t["NS"] + t["check"] + t["product"] + t["eigvals"]
                svd_total = t["overlaps"] + t["SVD"] + t["product"] + t["eigvals"]
                print(f"  ours {device} {precision:6s}: " + "  ".join(
                    f"{k} {1e3 * v:7.1f}" for k, v in t.items()) + " ms")
                print(f"      Wilson total: NS path {ns_total:.3f} s ({t_py / ns_total:4.1f}x vs "
                      f"PythTB), SVD path {svd_total:.3f} s ({t_py / svd_total:4.1f}x); "
                      f"NS is {t['NS'] / t['SVD']:.1f}x the SVD time")

                # how many NS steps do these overlaps actually need?
                tol = NS_TOL[be.cdtype] * n_occ
                steps_needed, Xk = None, M
                eye = torch.eye(n_occ, dtype=be.cdtype, device=be.device)
                res = []
                for k in range(1, NS_MAX_STEPS + 1):
                    Xk = 0.5 * Xk @ (3 * eye - Xk.mH @ Xk)
                    r = float(residual(Xk).max())
                    res.append(r)
                    if steps_needed is None and r <= tol:
                        steps_needed = k
                sv = torch.linalg.svdvals(M)
                print(f"      overlap singular values: min {float(sv.min()):.3f}; NS residual "
                      f"(worst link) by step: " + " ".join(f"{r:.0e}" for r in res[:8])
                      + f" ...; steps to reach tol {tol:.0e}: {steps_needed}")
                U_svd = unitary_part_svd(M)
                early = []
                for k in (NS_MAX_STEPS, 4, 3):
                    tk = best_of(lambda M=M, k=k: residual(newton_schulz(M, k)), be)
                    err = float((newton_schulz(M, k) - U_svd).abs().max())
                    early.append(f"{k} steps {1e3 * tk:.1f} ms (max|U-U_svd| {err:.0e})")
                print("      NS + check, stopping early: " + " | ".join(early))
                if device == "cpu":
                    nt = torch.get_num_threads()
                    torch.set_num_threads(1)
                    t1_ns = best_of(lambda M=M: newton_schulz(M, NS_MAX_STEPS), be)
                    t1_svd = best_of(lambda M=M: unitary_part_svd(M), be)
                    torch.set_num_threads(nt)
                    print(f"      1 torch thread: NS {1e3 * t1_ns:.1f} ms (vs {1e3 * t['NS']:.1f} "
                          f"with {nt}), SVD {1e3 * t1_svd:.1f} ms (vs {1e3 * t['SVD']:.1f})")
        print()


if __name__ == "__main__":
    main()
