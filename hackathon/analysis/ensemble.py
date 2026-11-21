"""Population ("ensemble") runs: many disorder realisations per W, one tidy row per sample.

Written at the hackathon (21 Nov 2026). Uses the batched GPU pipeline (gpu_engine.pipeline
.run_batch) and adds, for every sample, statistics of its own random on-site energies so we
can ask *why* a particular sample is topological or not (see ensemble_plots.py).

Seeds
-----
seeds="shared" (default): sample i uses the same seed at every W. numpy's
uniform(-W/2, W/2) is -W/2 + W*r for the same r, so sample i is ONE disorder pattern scaled
by W. Following a sample across W then gives its own critical disorder (where its Z2 flips).
Caveat: points at different W are then correlated, so don't treat neighbouring W's error bars
as independent.
seeds="independent": fresh seeds at every W (plain independent ensembles).

Disorder ensembles
------------------
"uniform": eps_i ~ U[-W/2, W/2] per orbital (= per site), same for both spins. Matches
           gpu_engine.batched.disorder_shifts and gpu_engine.reference.
"pinned":  the same numbers with each sublattice's mean subtracted, so sublattice A and B
           both average exactly 0 in every sample. Paired control for the hypothesis below.
           Caveat (review, 21 Nov): subtracting the mean adds within-sublattice
           anticorrelations and lowers the variance by a factor (1 - 2/N_sites), so it is
           not a pure "remove the imbalance" operation; small effect for N_sites >= 32.

Hypothesis (to check): a finite supercell's random energies don't average to 0. A uniform
shift of +a on sublattice A and +b on B is exactly a staggered potential (a - b)/2 plus a
global shift (a + b)/2. So each sample's uniform staggered component is
delta_eff = delta + (eps_A - eps_B)/2 (A has +delta, B has -delta in PythTB's kane_mele;
checked from the onsite terms, see sublattice_index). Its SD is W / sqrt(12 N_sites)
= W / (2 sqrt(6) n) for an n x n supercell, so it shrinks as 1/n and may account for part of
the finite-size width of the transition. How much it explains is open: the Born shift of
delta is second order in W and comes from scattering, not from this mean (review, 21 Nov).
The global shift (a + b)/2 moves all energies equally, so it cannot change Z2 at fixed
half filling; it only matters when comparing with transport at fixed E_F.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gpu_engine.backend import Backend
from gpu_engine.batched import disorder_shifts
from gpu_engine.pipeline import run_batch
from gpu_engine.reference import TUTORIAL_PARAMS, clean_kane_mele

CLEAN_BOUNDARY_FACTOR = 3 * np.sqrt(3)  # clean transition at delta = 3 sqrt(3) soc (rashba = 0)


# ----------------------------------------------------------------------------- disorder


def sublattice_index(supercell: int, **params) -> np.ndarray:
    """0 for sublattice A (onsite +delta), 1 for B (onsite -delta), per orbital. Shape (norb,).

    PythTB's kane_mele orders orbitals A, B and make_supercell keeps that order per cell, so
    this is just alternating; we check it against the onsite signs when delta != 0.
    """
    model = clean_kane_mele(supercell, **params)
    sub = np.arange(model.norb) % 2
    onsite = np.real(np.trace(np.asarray(model.onsite), axis1=1, axis2=2)) / 2  # (norb,)
    if not np.allclose(onsite, 0):  # A carries +delta, B -delta (delta may be negative)
        assert np.allclose(onsite[sub == 0], onsite[0]) and np.allclose(
            onsite[sub == 1], -onsite[0]), "unexpected orbital order"
    return sub


def pinned_shifts(norb: int, W: float, seeds: list[int]) -> np.ndarray:
    """Like batched.disorder_shifts, but each sublattice's mean is removed per sample.

    Assumes alternating A, B orbital order (checked in sublattice_index). Returns (C, 2*norb).
    """
    eps = disorder_shifts(norb, W, seeds)[:, 0::2]  # (C, norb), one value per orbital
    sub = np.arange(norb) % 2
    for s in (0, 1):
        eps[:, sub == s] -= eps[:, sub == s].mean(axis=1, keepdims=True)
    return np.repeat(eps, 2, axis=1)


SHIFT_FUNCTIONS = {"uniform": disorder_shifts, "pinned": pinned_shifts}


def disorder_statistics(eps: np.ndarray, sub: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample statistics of the realised on-site energies.

    eps: (C, norb) one value per orbital. sub: (norb,) 0 = A, 1 = B.
    """
    eps_A, eps_B = eps[:, sub == 0].mean(axis=1), eps[:, sub == 1].mean(axis=1)
    return {
        "eps_mean": eps.mean(axis=1),
        "eps_std": eps.std(axis=1),
        "eps_A_minus_B": eps_A - eps_B,
        "eps_max_abs": np.abs(eps).max(axis=1),
    }


# ----------------------------------------------------------------------------- sweep


@dataclass
class SweepConfig:
    supercells: list[int]
    Ws: list[float]
    n_samples: int
    nk: dict[int, int]  # supercell -> k-mesh (bigger supercell, smaller zone, fewer k)
    deltas: list[float] = field(default_factory=lambda: [TUTORIAL_PARAMS["delta"]])
    soc: float = TUTORIAL_PARAMS["soc"]
    rashba: float = TUTORIAL_PARAMS["rashba"]
    disorder: str = "uniform"  # "uniform" or "pinned"
    seeds: str = "shared"  # "shared" or "independent"
    seed0: int = 0
    keep_flows: int = 0  # keep the Wannier flow of the first `keep_flows` samples per W
    spin_blocks: bool = False  # solve spin up/down separately (rashba = 0 only; ~4x cheaper)


def seeds_for(cfg: SweepConfig, iW: int) -> list[int]:
    base = cfg.seed0 if cfg.seeds == "shared" else cfg.seed0 + iW * cfg.n_samples
    return list(range(base, base + cfg.n_samples))


def run_sweep(cfg: SweepConfig, backend: Backend, progress: bool = True):
    """Run every (supercell, delta, W) point with n_samples configurations each.

    Returns (rows, flows): rows is a list of dicts, one per sample (write with save_rows);
    flows maps "L{n}_d{delta}_W{W}" -> (keep_flows, nk/2+1, n_occ) float32 Wannier flows.
    """
    shift_fn = SHIFT_FUNCTIONS[cfg.disorder]
    rows, flows = [], {}
    for n in cfg.supercells:
        nk = cfg.nk[n]
        for delta in cfg.deltas:
            params = {"delta": delta, "soc": cfg.soc, "rashba": cfg.rashba}
            sub = sublattice_index(n, **params)
            norb = len(sub)
            for iW, W in enumerate(cfg.Ws):
                seeds = seeds_for(cfg, iW)
                t0 = time.perf_counter()
                res = run_batch(
                    n, W, nk, seeds, backend, check=True,
                    keep_flows=cfg.keep_flows > 0, shift_fn=shift_fn,
                    spin_blocks=cfg.spin_blocks, **params,
                )
                wall = time.perf_counter() - t0
                stats = disorder_statistics(shift_fn(norb, W, seeds)[:, 0::2], sub)
                reliable = res.reliable
                for i, s in enumerate(seeds):
                    rows.append({
                        "L": n, "delta": delta, "soc": cfg.soc, "rashba": cfg.rashba,
                        "W": W, "sample": i, "seed": s, "disorder": cfg.disorder,
                        "seeds_mode": cfg.seeds, "nk": nk, "precision": backend.precision,
                        "z2": int(res.z2[i]), "reliable": int(reliable[i]),
                        "band_gap": res.band_gap[i], "z2_margin": res.z2_margin[i],
                        "min_overlap": res.min_overlap[i],
                        **{k: v[i] for k, v in stats.items()},
                        "delta_eff": delta + stats["eps_A_minus_B"][i] / 2,
                        "batch_ns_fallbacks": res.ns_fallbacks,
                        "batch_seconds": wall,
                    })
                if res.flows is not None:
                    flows[f"L{n}_d{delta:g}_W{W:g}"] = res.flows[: cfg.keep_flows]
                if progress:
                    z = res.z2[reliable]
                    print(f"  L={n} delta={delta:g} W={W:5.2f}: P(Z2=1)={res.z2.mean():.2f} "
                          f"(reliable only {z.mean() if len(z) else float('nan'):.2f}, "
                          f"{reliable.mean():.0%} reliable), {wall:.2f} s")
    return rows, flows


def save_rows(rows: list[dict], path: Path, cfg: SweepConfig, backend: Backend) -> None:
    """CSV with one row per sample, plus a .json sidecar with the run settings and hardware."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "config": asdict(cfg) | {"nk": {str(k): v for k, v in cfg.nk.items()}},
        "hardware": backend.describe(),
        "platform": platform.platform(),
        "disorder_convention": "eps_i ~ U[-W/2, W/2] per site, both spins (ours; Orth uses "
                               "[-W, W], i.e. W_ours = 2 W_Orth)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_rows(*paths: Path) -> dict[str, np.ndarray]:
    """Read one or more ensemble CSVs into a dict of column arrays (numbers where possible)."""
    import csv

    cols: dict[str, list] = {}
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    cols.setdefault(k, []).append(v)
    out = {}
    for k, v in cols.items():
        try:
            out[k] = np.array(v, dtype=float)
        except ValueError:
            out[k] = np.array(v)
    for k in ("L", "sample", "seed", "z2", "reliable", "nk"):
        if k in out:
            out[k] = out[k].astype(int)
    out["reliable"] = out["reliable"].astype(bool)
    return out
