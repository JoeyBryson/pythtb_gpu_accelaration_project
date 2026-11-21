"""CPU reference pipeline using plain PythTB. Every GPU version is tested against this.

Keep it slow and obviously correct; do not optimise it. Mirrors
docs/source/tutorials/kane_mele.ipynb.

The pipeline for one material ("one configuration"):
  1. build      Kane-Mele model -> n x n supercell -> random on-site energies (disorder)
  2. solve      diagonalise H(k) at every k-point             (PythTB: solve_model)
  3. wilson     Wilson loops along k_y -> Wannier centres      (PythTB: berry_phase)
  4. z2         Wannier-centre flow -> topological or not      (ours: topology.py)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pythtb import Mesh, TBModel, WFArray
from pythtb.models import kane_mele

from .topology import RELIABLE_MARGIN, z2_from_wannier_flow, z2_margin

# Tutorial parameters (topological phase), from CLAUDE.md.
TUTORIAL_PARAMS = {"delta": 1.0, "t": 1.0, "soc": 0.3, "rashba": 0.25}


# ---------------------------------------------------------------- 1. build the model


def clean_kane_mele(supercell: int = 1, **params) -> TBModel:
    """Kane-Mele model, optionally as an n x n supercell. Spinful."""
    p = {**TUTORIAL_PARAMS, **params}
    model = kane_mele(p["delta"], p["t"], p["soc"], p["rashba"])
    if supercell > 1:
        model = model.make_supercell([[supercell, 0], [0, supercell]])
    return model


def disordered_kane_mele(
    supercell: int, W: float, rng: np.random.Generator, **params
) -> TBModel:
    """Kane-Mele supercell with Anderson disorder: a random on-site energy per orbital,
    drawn uniformly from [-W/2, W/2], spin-independent (same value for both spins).

    To check (physics): that this convention for W matches Orth et al. (2016), and
    whether the paper uses per-orbital or per-site disorder (same thing here, since
    each site carries one orbital).
    """
    model = clean_kane_mele(supercell, **params)
    shifts = rng.uniform(-W / 2, W / 2, size=model.norb)
    model.set_onsite(list(shifts), mode="add")
    return model


# ---------------------------------------------------------------- 2. solve


def solve(model: TBModel, nk: int) -> WFArray:
    """Diagonalise H(k) on an nk x nk gamma-centred mesh (the ~70% CPU-time stage)."""
    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk), gamma_centered=True)
    wfa = WFArray(model.lattice, mesh, spinful=True)
    wfa.solve_model(model=model)
    return wfa


def band_gap(wfa: WFArray, n_occ: int) -> float:
    """Gap between the highest occupied and lowest empty band, over the whole mesh.
    Negative means the bands overlap (a metal at this filling)."""
    e = np.asarray(wfa.energies)  # (nk, nk, n_states)
    return float(e[..., n_occ].min() - e[..., n_occ - 1].max())


# ---------------------------------------------------------------- 3. Wilson loop


def wannier_centres(wfa: WFArray, n_occ: int, axis_idx: int = 1) -> np.ndarray:
    """Hybrid Wannier centres in [0, 1) units of the cell, for the lowest n_occ bands.

    Wilson loop along mesh axis `axis_idx`, one per point along the other axis.
    Returns: (nk_other_axis, n_occ) float.
    """
    phases = wfa.berry_phase(
        axis_idx=axis_idx, state_idx=list(range(n_occ)), contin=True, berry_evals=True
    )
    return np.asarray(phases) / (2 * np.pi)


def half_bz_flow(wfa: WFArray, centres: np.ndarray) -> np.ndarray:
    """Reorder Wannier centres (from wilson along axis 1) to run k_x = 0 ... 0.5 (pi).

    centres: (nk, n_occ). Returns (nk/2 + 1, n_occ) in [0, 1). Needs even nk so that
    k_x = pi is on the mesh.
    """
    kx = np.asarray(wfa.mesh.points)[:, 0, 0] % 1.0  # k_x of each row, in [0, 1)
    rows = np.where(kx <= 0.5 + 1e-12)[0]
    rows = rows[np.argsort(kx[rows])]
    if not (np.isclose(kx[rows[0]], 0.0) and np.isclose(kx[rows[-1]], 0.5)):
        raise ValueError("k-mesh must include k_x = 0 and k_x = 0.5; use an even nk")
    return centres[rows] % 1.0


# ---------------------------------------------------------------- full pipeline


@dataclass
class ConfigResult:
    """Everything we measure for one configuration (one imperfect material)."""

    z2: int | None  # None if the mesh can't give Z2 (odd nk)
    z2_margin: float | None  # how safe the Z2 answer is (see topology.z2_margin)
    band_gap: float  # negative = bands overlap (metal), Z2 not meaningful
    t_build: float  # seconds per stage
    t_solve: float
    t_wilson: float
    t_z2: float

    @property
    def t_total(self) -> float:
        return self.t_build + self.t_solve + self.t_wilson + self.t_z2

    @property
    def z2_reliable(self) -> bool:
        """True if the Z2 answer can be trusted: gapped, and gap tracking was safe."""
        return (
            self.z2 is not None
            and self.band_gap > 0
            and self.z2_margin is not None
            and self.z2_margin >= RELIABLE_MARGIN
        )


def run_config(supercell: int, W: float, nk: int, seed: int | None = None, **params) -> ConfigResult:
    """Run the whole pipeline for one configuration, timing each stage.

    W = 0 gives the clean model. `seed` makes the disorder reproducible: the same seed
    always produces the same material, so CPU and GPU runs can be compared exactly.
    """
    t0 = time.perf_counter()
    if W == 0:
        model = clean_kane_mele(supercell, **params)
    else:
        model = disordered_kane_mele(supercell, W, np.random.default_rng(seed), **params)
    n_occ = model.norb  # half filling: spinful, so 2*norb states and norb occupied

    t1 = time.perf_counter()
    wfa = solve(model, nk)

    t2 = time.perf_counter()
    centres = wannier_centres(wfa, n_occ)

    t3 = time.perf_counter()
    z2 = margin = None
    if nk % 2 == 0:
        flow = half_bz_flow(wfa, centres)
        z2 = z2_from_wannier_flow(flow)
        margin = z2_margin(flow)
    t4 = time.perf_counter()

    return ConfigResult(
        z2=z2,
        z2_margin=margin,
        band_gap=band_gap(wfa, n_occ),
        t_build=t1 - t0,
        t_solve=t2 - t1,
        t_wilson=t3 - t2,
        t_z2=t4 - t3,
    )


def periodic_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between Wannier centres on the circle [0, 1)."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(d, 1.0 - d)
