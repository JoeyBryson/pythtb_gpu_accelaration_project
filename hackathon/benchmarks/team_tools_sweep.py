"""Disorder sweep (W x configs -> fraction topological) with the team's tools, every engine
on the same materials.

Written at the hackathon (21 Nov 2026). Same test as scripts/sample_cpu_sim.py, but built
from gpu_engine.reference / gpu_engine.batched / benchmarks/cpu_baseline.py:

  pythtb_serial     reference.run_config in a Python loop          (what users do today)
  pythtb_parallel   same, one process per CPU core, 1 BLAS thread   (strongest simple CPU baseline)
  pyW-<d|s>         batched.solve_configs (H_clean once + random diagonals, one torch eigh for
                    all configs of a W), then PythTB's per-line Wilson loop (berry_phase)
  NS-<d|s>          pipeline.run_batch: batched solve + wilson.py (Newton-Schulz, tree product,
                    all lines and configs in one batch) + topology Z2. The team's pipeline as is.
  SVD-<d|s>         same as NS but wilson.wannier_centres(method="svd"): isolates what
                    Newton-Schulz itself costs or saves vs batching alone
  (d = complex128, s = complex64; cuda variants are added automatically when a GPU is present)

Every engine sees the same seeds, so Z2 and band gaps are compared config by config with
pythtb_serial. Timing includes building H and host<->device copies.

PythTB runs are timed with check=False: run_config's default check=True adds an SVD diagnostic
(~20% of wall time) that the batched path doesn't do. Reliability flags come from a separate,
untimed check=True pass.

W uses the team convention, on-site energies uniform in [-W/2, W/2] (W_team = 2 * W_Orth).

Example (from hackathon/):
  uv run python benchmarks/team_tools_sweep.py --sizes 3 6 --configs 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from common import hardware_info, save_rows, scaled_nk
from cpu_baseline import _one_blas_thread, config_seeds
from pythtb import Mesh, WFArray

from gpu_engine import get_backend, sync
from gpu_engine.batched import solve_configs
from gpu_engine.pipeline import half_bz_rows, run_batch
from gpu_engine.reference import clean_kane_mele, half_bz_flow, run_config, wannier_centres
from gpu_engine.topology import z2_from_wannier_flow
from gpu_engine.wilson import wannier_centres as batched_wannier_centres


def _run_config_nocheck(args: tuple):
    """Worker task: the timed pipeline only (no reliability diagnostic)."""
    n, W, nk, s = args
    return run_config(n, W, nk, seed=s, check=False)


def wilson_z2_from_states(model, nk: int, u_occ: np.ndarray) -> tuple[int, float]:
    """PythTB Wilson loop + team Z2 on states from the batched solve, for ONE config.

    u_occ: (nk, nk, n_occ, N) complex, spin flattened as (orbital, spin).
    Returns (z2, seconds spent in the Wilson loop).
    """
    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk), gamma_centered=True)  # same mesh as batched.k_mesh
    n_occ = u_occ.shape[-2]
    wfa = WFArray(model.lattice, mesh, nstates=n_occ, spinful=True)
    wfa.set_states(u_occ.reshape(nk, nk, n_occ, model.norb, 2), is_cell_periodic=True)
    t0 = time.perf_counter()
    centres = wannier_centres(wfa, n_occ)
    t_wilson = time.perf_counter() - t0
    return z2_from_wannier_flow(half_bz_flow(wfa, centres)), t_wilson


def run_batched(n: int, nk: int, W: float, seeds: list[int], device: str, precision: str):
    """Batched solve for all configs at once, then per-config Wilson loop on the CPU."""
    be = get_backend(precision, device)
    model = clean_kane_mele(n)
    t0 = time.perf_counter()
    energies, u_occ = solve_configs(model, nk, W, seeds, be)
    energies = energies.cpu().to(torch.float64).numpy()  # (C, nk, nk, N)
    u_occ = u_occ.cpu().to(torch.complex128).numpy()  # (C, nk, nk, n_occ, N)
    sync(be)
    t_solve = time.perf_counter() - t0
    n_occ = model.norb
    gaps = energies[..., n_occ].min(axis=(1, 2)) - energies[..., n_occ - 1].max(axis=(1, 2))
    z2s, t_wilson = [], 0.0
    for u in u_occ:
        z2, tw = wilson_z2_from_states(model, nk, u)
        z2s.append(z2)
        t_wilson += tw
    wall = time.perf_counter() - t0
    return {"wall": wall, "t_solve": t_solve, "t_wilson": t_wilson, "z2": z2s, "gap": list(gaps)}


def run_wilson_batched(n: int, nk: int, W: float, seeds: list[int], device: str, precision: str,
                       method: str):
    """Batched solve + batched Wilson loop (wilson.py) + Z2. method: "newton-schulz" calls the
    team's pipeline.run_batch unchanged; "svd" is the same chain with the SVD unitary part."""
    be = get_backend(precision, device)
    if method == "newton-schulz":
        sync(be)
        t0 = time.perf_counter()
        out = run_batch(n, W, nk, seeds, be, check=False)
        sync(be)
        wall = time.perf_counter() - t0
        return {"wall": wall, "t_solve": out.timings["solve"], "t_wilson": out.timings["wilson"],
                "z2": [int(z) for z in out.z2], "gap": list(out.band_gap),
                "fallbacks": out.ns_fallbacks}
    model = clean_kane_mele(n)
    n_occ = model.norb
    rows = half_bz_rows(nk)
    sync(be)
    t0 = time.perf_counter()
    energies, u = solve_configs(model, nk, W, seeds, be)
    sync(be)
    t1 = time.perf_counter()
    wcc, _ = batched_wannier_centres(u, model, method="svd")
    sync(be)
    t2 = time.perf_counter()
    flows = wcc[:, rows].double().cpu().numpy()
    z2 = [z2_from_wannier_flow(f) for f in flows]
    gap = (energies[..., n_occ].amin(dim=(1, 2)) - energies[..., n_occ - 1].amax(dim=(1, 2)))
    gap = gap.double().cpu().tolist()
    wall = time.perf_counter() - t0
    return {"wall": wall, "t_solve": t1 - t0, "t_wilson": t2 - t1, "z2": z2, "gap": gap,
            "fallbacks": 0}


def run_engine(name: str, n: int, nk: int, W: float, seeds: list[int]):
    """name = '<kind>-<d|s>[-cuda]', kind in pyW / NS / SVD."""
    kind, prec_code, *dev = name.split("-")
    device = dev[0] if dev else "cpu"
    precision = {"d": "double", "s": "single"}[prec_code]
    if kind == "pyW":
        return {**run_batched(n, nk, W, seeds, device, precision), "fallbacks": 0}
    method = {"NS": "newton-schulz", "SVD": "svd"}[kind]
    return run_wilson_batched(n, nk, W, seeds, device, precision, method)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[3, 6])
    p.add_argument("--W", type=float, nargs="+", default=[0, 1, 2, 4, 6, 8, 12, 16, 24],
                   help="team convention [-W/2, W/2]; default = 2x the sample sim's W values")
    p.add_argument("--configs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    engines = [f"{kind}-{prec}{'' if d == 'cpu' else '-' + d}"
               for d in devices for kind in ("pyW", "NS", "SVD") for prec in ("d", "s")]
    hw = hardware_info()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"CPU {hw['cpu']} ({hw['n_cpus']} cpus, BLAS {hw['blas']}, numpy {hw['numpy']}), "
          f"torch {torch.__version__}, GPU {gpu}\n")

    for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[var] = "1"  # read by the worker processes at start-up only
    pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                               initializer=_one_blas_thread)
    list(pool.map(_run_config_nocheck, [(1, 0.0, 4, 0)] * args.workers))  # warm up workers
    for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        del os.environ[var]
    run_config(1, W=0, nk=4)  # warm up this process
    for e in engines:  # warm up torch
        run_engine(e, 2, 6, 1.0, [0])

    rows = []
    for n in args.sizes:
        nk = scaled_nk(n)
        print(f"=== {n}x{n} supercell (H {4 * n * n}x{4 * n * n}), mesh {nk}x{nk}, "
              f"{args.configs} configs per W")
        header = f"{'W':>5} {'topo':>5} {'unrel':>5} | {'serial':>7} {'parallel':>8} " + " ".join(
            f"{e:>7}" for e in engines) + " | Z2 & gap agree"
        print(header)
        totals = {k: 0.0 for k in ["pythtb_serial", "pythtb_parallel", *engines]}
        stage = {k: [0.0, 0.0] for k in engines}
        fallbacks = dict.fromkeys(engines, 0)
        for W in args.W:
            seeds = config_seeds(args.seed + int(1000 * W), args.configs)

            ref = [run_config(n, W, nk, seed=s, check=True) for s in seeds]  # untimed: flags
            t0 = time.perf_counter()
            ser = [run_config(n, W, nk, seed=s, check=False) for s in seeds]
            t_serial = time.perf_counter() - t0

            t0 = time.perf_counter()
            par = list(pool.map(_run_config_nocheck, [(n, W, nk, s) for s in seeds]))
            t_par = time.perf_counter() - t0

            ref_z2 = [r.z2 for r in ref]
            ref_gap = np.array([r.band_gap for r in ref])
            agree = [r.z2 for r in par] == ref_z2 and [r.z2 for r in ser] == ref_z2
            out = {}
            for e in engines:
                out[e] = run_engine(e, n, nk, W, seeds)
                tol = 1e-8 if "-d" in e else 1e-3
                fallbacks[e] += out[e]["fallbacks"]
                agree &= out[e]["z2"] == ref_z2
                agree &= bool(np.abs(np.array(out[e]["gap"]) - ref_gap).max() < tol)
                stage[e][0] += out[e]["t_solve"]
                stage[e][1] += out[e]["t_wilson"]

            trusted = [r.z2 for r in ref if r.z2_reliable]
            frac = np.mean(trusted) if trusted else float("nan")
            totals["pythtb_serial"] += t_serial
            totals["pythtb_parallel"] += t_par
            for e in engines:
                totals[e] += out[e]["wall"]
            print(f"{W:5.1f} {frac:5.0%} {args.configs - len(trusted):5d} | {t_serial:7.2f} "
                  f"{t_par:8.2f} " + " ".join(f"{out[e]['wall']:7.2f}" for e in engines)
                  + f" | {'yes' if agree else 'NO'}")
            rows += [{
                "supercell": n, "nk": nk, "W": W, "configs": args.configs, "engine": k,
                "wall_s": f"{v:.4f}", "frac_topological": f"{frac:.3f}",
                "n_unreliable": args.configs - len(trusted), "answers_match_serial": agree,
                "torch": torch.__version__, "gpu": gpu,
            } for k, v in [("pythtb_serial", t_serial), ("pythtb_parallel", t_par)]
              + [(e, out[e]["wall"]) for e in engines]]

        print("total " + " ".join(f"{k}={v:.2f}s" for k, v in totals.items()))
        base = totals["pythtb_serial"]
        print("speed-up vs serial: " + "  ".join(
            f"{k}={base / v:.1f}x" for k, v in totals.items() if k != "pythtb_serial"))
        print("speed-up vs parallel: " + "  ".join(
            f"{e}={totals['pythtb_parallel'] / totals[e]:.1f}x" for e in engines))
        print("stage split, solve / Wilson (s): " + "  ".join(
            f"{e}={s:.2f}/{w:.2f}" for e, (s, w) in stage.items()))
        print("Newton-Schulz SVD fallbacks: " + "  ".join(
            f"{e}={v}" for e, v in fallbacks.items() if e.startswith("NS")))
        print()
    pool.shutdown()
    print(f"saved to {save_rows('team_tools_sweep', rows)}")


if __name__ == "__main__":
    main()
