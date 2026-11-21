"""Stage 0 (goal.md): does our method see the TAI at lambda_nu = 1.65t at all?

CPU only, plain PythTB via gpu_engine.reference. L = 8, nk = 6, lambda_SO = 0.3, lambda_R = 0,
W (ours, eps in [-W/2, W/2]) in {0, 1, 2, 3, 5, 8}, 20 seeds per W.

Expected if it works (Favata Fig. 7 / Orth Fig. 5, our units):
    W = 0, 1 -> trivial (f ~ 0);  W ~ 2 -> partial;  W = 3, 5 -> topological (f ~ 1);  W = 8 -> trivial

Mirrors reference.run_config step by step, and additionally records the DIRECT gap
(min over k of E_{n_occ}(k) - E_{n_occ-1}(k)) next to the indirect gap that `reliable` uses,
so we can decide the gap policy from data.

Run from hackathon/:  uv run python analysis/stage0_go_no_go.py   (about 3 min on 8 cores)\nMesh check:          uv run python analysis/stage0_go_no_go.py --nk 10 --W 2 3
Writes results/stage0_go_no_go.csv (one row per configuration).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gpu_engine.reference import (
    band_gap,
    clean_kane_mele,
    disordered_kane_mele,
    half_bz_flow,
    min_overlap_singular_value,
    solve,
    wannier_centres,
)
from gpu_engine.topology import RELIABLE_MARGIN, RELIABLE_OVERLAP, z2_from_wannier_flow, z2_margin

L, NK = 8, 6
PARAMS = {"delta": 1.65, "t": 1.0, "soc": 0.3, "rashba": 0.0}
W_VALUES = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
SEEDS = list(range(20))  # same seeds at every W: same disorder pattern, scaled by W
OUT = Path(__file__).resolve().parent.parent / "results" / "stage0_go_no_go.csv"
_p = argparse.ArgumentParser()
_p.add_argument("--nk", type=int, default=NK, help="k-mesh per direction (even)")
_p.add_argument("--W", type=float, nargs="+", default=W_VALUES, help="disorder strengths (ours)")
_args, _ = _p.parse_known_args()
NK, W_VALUES = _args.nk, _args.W
if NK != 6 or W_VALUES != [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]:
    OUT = OUT.with_name(f"stage0_go_no_go_nk{NK}.csv")


def one_config(args: tuple[float, int]) -> dict:
    W, seed = args
    t0 = time.perf_counter()
    if W == 0:
        model = clean_kane_mele(L, **PARAMS)
    else:
        model = disordered_kane_mele(L, W, np.random.default_rng(seed), **PARAMS)
    n_occ = model.norb
    wfa = solve(model, NK)
    flow = half_bz_flow(wfa, wannier_centres(wfa, n_occ))
    e = np.asarray(wfa.energies)  # (nk, nk, n_states)
    row = {
        "L": L, "nk": NK, "W": W, "seed": seed,
        "z2": z2_from_wannier_flow(flow),
        "z2_margin": z2_margin(flow),
        "min_overlap": min_overlap_singular_value(wfa, n_occ),
        "gap_indirect": band_gap(wfa, n_occ),
        "gap_direct": float((e[..., n_occ] - e[..., n_occ - 1]).min()),
        "seconds": time.perf_counter() - t0,
    }
    tracking_ok = row["z2_margin"] >= RELIABLE_MARGIN and row["min_overlap"] >= RELIABLE_OVERLAP
    row["reliable_indirect"] = tracking_ok and row["gap_indirect"] > 0  # current policy
    row["reliable_direct"] = tracking_ok and row["gap_direct"] > 0  # proposed policy
    return row


def summarise(rows: list[dict]) -> None:
    print(f"\nL = {L}, nk = {NK}, {PARAMS}\n")
    print(" W    n   f(indirect)  f(direct)  f(all)   u_ind u_dir  low_margin low_overlap"
          "  gap_ind<0 gap_dir<0  median gap_dir")
    for W in W_VALUES:
        r = [x for x in rows if x["W"] == W]
        z = np.array([x["z2"] for x in r])
        ri = np.array([x["reliable_indirect"] for x in r])
        rd = np.array([x["reliable_direct"] for x in r])

        def frac(mask):
            return f"{z[mask].mean():.2f} ({mask.sum():2d})" if mask.any() else "  -- ( 0)"

        print(
            f"{W:3.0f}  {len(r):3d}   {frac(ri):11s}  {frac(rd):9s}  {z.mean():.2f}    "
            f"{1 - ri.mean():.2f}  {1 - rd.mean():.2f}   "
            f"{sum(x['z2_margin'] < RELIABLE_MARGIN for x in r):6d}   "
            f"{sum(x['min_overlap'] < RELIABLE_OVERLAP for x in r):8d}     "
            f"{sum(x['gap_indirect'] <= 0 for x in r):6d}   {sum(x['gap_direct'] <= 0 for x in r):6d}"
            f"      {np.median([x['gap_direct'] for x in r]):.4f}"
        )
    print("\nf = fraction with Z2 = 1 among reliable configs (count in brackets); u = fraction unreliable")


def main() -> None:
    tasks = ([(0.0, 0)] if 0.0 in W_VALUES else []) + [(W, s) for W in W_VALUES if W > 0 for s in SEEDS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        rows = list(pool.map(one_config, tasks))
    print(f"{len(rows)} configs in {time.perf_counter() - t0:.0f} s on {os.cpu_count()} processes")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker
    main()
