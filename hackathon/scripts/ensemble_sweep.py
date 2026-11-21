"""Run a disorder ensemble sweep and save one row per sample (plus Wannier flows).

Written at the hackathon (21 Nov 2026). Produces the input for analysis/ensemble_plots.py.

Examples (from hackathon/):
  # quick laptop check (~1 min on CPU)
  uv run python scripts/ensemble_sweep.py --quick

  # TAI side of the phase diagram: clean model trivial (delta > 3 sqrt(3) soc), rashba = 0
  uv run python scripts/ensemble_sweep.py --L 4 6 8 --delta 1.7 --rashba 0 \\
      --W 0 0.5 1 1.5 2 2.5 3 3.5 4 5 6 8 --samples 1000 --tag tai

  # control: same, but each sublattice's mean disorder pinned to zero per sample
  uv run python scripts/ensemble_sweep.py ... --disorder pinned --tag tai_pinned

Output: results/ensemble_<tag>.csv (+ .json settings/hardware, + _flows.npz)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `analysis` imports from hackathon/
from analysis.ensemble import SweepConfig, run_sweep, save_rows
from gpu_engine import get_backend

RESULTS = Path(__file__).resolve().parents[1] / "results"

# k-mesh per supercell (even nk needed for Z2). Chosen 21 Nov from scripts/nk_stability.py
# (delta=1.65, rashba=0): 4x4 at nk=8 failed the overlap check even for the clean model, and
# Z2 near onset kept changing up to nk=16.
# The Dirac point K = (1/3, 2/3) of the primitive cell sits at (n/3, 2n/3) mod 1 in the n x n
# supercell zone: at Gamma when 3 | n, otherwise at 1/3, 2/3, which an nk-mesh contains only if
# 3 | nk. Missing K skips the gap minimum and lowers the overlap check non-monotonically
# (clean 2x2: min overlap 0.85, 0.72, 0.76, 0.87, 0.80, 0.90 at nk = 12, 14, 16, 18, 20, 24).
# So for 3 not dividing n we use nk divisible by 6. TO CHECK: convergence for n >= 5.
DEFAULT_NK = {2: 18, 3: 16, 4: 18, 5: 12, 6: 10, 8: 12, 10: 12, 12: 8, 16: 6}
# 16x16: nk=6 as in the team's runs (RESULTS.md R2/R10); clean Z2 correct there. TO CHECK.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--L", type=int, nargs="+", default=[3, 4], help="supercell sizes n (n x n)")
    ap.add_argument("--W", type=float, nargs="+",
                    default=[0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--delta", type=float, nargs="+", default=[1.0],
                    help="staggered potential(s); clean boundary 3*sqrt(3)*soc")
    ap.add_argument("--soc", type=float, default=0.3)
    ap.add_argument("--rashba", type=float, default=0.25)
    ap.add_argument("--samples", type=int, default=200, help="disorder configurations per W")
    ap.add_argument("--nk", type=int, default=None, help="override the k-mesh for every L")
    ap.add_argument("--disorder", choices=["uniform", "pinned"], default="uniform")
    ap.add_argument("--seeds", choices=["shared", "independent"], default="shared")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--keep-flows", type=int, default=200,
                    help="save Wannier flows for this many samples per W (0 = none)")
    ap.add_argument("--precision", choices=["double", "single"], default="double")
    ap.add_argument("--invariant", choices=["auto", "c_up", "z2"], default="auto",
                    help="c_up: spin-up Chern number (fastest; rashba = 0, uniform disorder); "
                         "z2: gap-tracking Wilson loop. auto = c_up when possible")
    ap.add_argument("--refine-nk", type=int, default=10,
                    help="c_up: re-run unreliable samples on this mesh (0 = don't)")
    ap.add_argument("--spin-blocks", action="store_true",
                    help="solve spin up/down as separate half-size problems (needs --rashba 0)")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--quick", action="store_true", help="tiny sweep for a laptop check")
    args = ap.parse_args()
    if args.quick:
        args.L, args.samples, args.W = [2, 3], 40, [0, 1, 2, 3, 4, 6, 8, 12]

    invariant = args.invariant
    if invariant == "auto":
        invariant = "c_up" if args.rashba == 0 and args.disorder == "uniform" else "z2"
    # c_up converges at nk = 6 (team's analysis/stage0_chern_up.py, RESULTS.md R11/R12)
    default_nk = {n: 6 for n in args.L} if invariant == "c_up" else DEFAULT_NK
    nk = {n: args.nk or default_nk.get(n, 4) for n in args.L}
    cfg = SweepConfig(
        supercells=args.L, Ws=args.W, n_samples=args.samples, nk=nk, deltas=args.delta,
        soc=args.soc, rashba=args.rashba, disorder=args.disorder, seeds=args.seeds,
        seed0=args.seed0, keep_flows=args.keep_flows, spin_blocks=args.spin_blocks,
        invariant=invariant, refine_nk=args.refine_nk or None,
    )
    be = get_backend(args.precision, args.device)
    print(f"Hardware: {be.describe()}  |  invariant: {invariant}")
    print(f"{len(args.L) * len(args.delta) * len(args.W) * args.samples} samples, nk={nk}\n")

    rows, flows = run_sweep(cfg, be)
    out = RESULTS / f"ensemble_{args.tag}.csv"
    save_rows(rows, out, cfg, be)
    if flows:
        np.savez_compressed(out.with_name(out.stem + "_flows.npz"), **flows)
    print(f"\nSaved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
