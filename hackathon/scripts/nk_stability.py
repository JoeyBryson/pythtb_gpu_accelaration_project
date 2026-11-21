"""Is a Z2 call converged in the k-mesh? Same seeds, increasing nk.

Written at the hackathon (21 Nov 2026). The ensemble runs flag many samples "unresolved",
mostly because the smallest overlap singular value is below topology.RELIABLE_OVERLAP (0.8,
tuned on the clean model). This script asks the direct question: does Z2 change when the
twist mesh is refined? If unresolved calls at nk agree with the finest mesh, 0.8 is too
strict; if they flip, it is doing its job.

Run from hackathon/:
  uv run python scripts/nk_stability.py --L 3 --nks 10 16 22 --W 1.5 2.5 3 --samples 40
Output: results/nk_stability_L<L>.csv (one row per sample, W and nk)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_engine import get_backend
from gpu_engine.pipeline import run_batch

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--nks", type=int, nargs="+", default=[10, 16, 22], help="even, ascending")
    ap.add_argument("--W", type=float, nargs="+", default=[1.5, 2.5, 3.0])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--delta", type=float, default=1.65)
    ap.add_argument("--rashba", type=float, default=0.0)
    ap.add_argument("--soc", type=float, default=0.3)
    args = ap.parse_args()

    be = get_backend("double")
    seeds = list(range(args.samples))  # same seeds as the ensemble runs (seed0 = 0, shared)
    params = {"delta": args.delta, "soc": args.soc, "rashba": args.rashba}
    rows = []
    print(f"Hardware: {be.describe()}  L={args.L}  nks={args.nks}")
    print(f"{'W':>5} {'nk':>4} {'P(Z2=1)':>8} {'reliable':>9} {'agrees w/ finest':>17} "
          f"{'unreliable & agrees':>20}")
    for W in args.W:
        res = {nk: run_batch(args.L, W, nk, seeds, be, check=True, **params) for nk in args.nks}
        z_ref = res[args.nks[-1]].z2
        for nk, r in res.items():
            agree = r.z2 == z_ref
            unrel = ~r.reliable
            print(f"{W:5.2f} {nk:4d} {r.z2.mean():8.2f} {r.reliable.mean():9.0%} "
                  f"{agree.mean():17.0%} "
                  f"{(agree[unrel].mean() if unrel.any() else np.nan):20.0%}")
            for i, s in enumerate(seeds):
                rows.append({"L": args.L, "W": W, "nk": nk, "seed": s, "z2": int(r.z2[i]),
                             "reliable": int(r.reliable[i]), "min_overlap": r.min_overlap[i],
                             "z2_margin": r.z2_margin[i], "band_gap": r.band_gap[i],
                             "z2_finest": int(z_ref[i])})
    out = RESULTS / f"nk_stability_L{args.L}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
