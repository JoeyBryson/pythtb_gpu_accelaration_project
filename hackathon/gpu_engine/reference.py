"""CPU reference pipeline using plain PythTB. Every GPU version is tested against this.

Keep it slow and obviously correct; do not optimise it. Mirrors
docs/source/tutorials/kane_mele.ipynb.
"""

from __future__ import annotations

import numpy as np
from pythtb import Mesh, TBModel, WFArray
from pythtb.models import kane_mele

# Tutorial parameters (topological phase), from CLAUDE.md.
TUTORIAL_PARAMS = {"delta": 1.0, "t": 1.0, "soc": 0.3, "rashba": 0.25}


def clean_kane_mele(supercell: int = 1, **params) -> TBModel:
    """Kane-Mele model, optionally as an n x n supercell. Spinful."""
    p = {**TUTORIAL_PARAMS, **params}
    model = kane_mele(p["delta"], p["t"], p["soc"], p["rashba"])
    if supercell > 1:
        model = model.make_supercell([[supercell, 0], [0, supercell]])
    return model


def solve(model: TBModel, nk: int) -> WFArray:
    """Diagonalise H(k) on an nk x nk gamma-centred mesh (the ~70% CPU-time stage)."""
    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk), gamma_centered=True)
    wfa = WFArray(model.lattice, mesh, spinful=True)
    wfa.solve_model(model=model)
    return wfa


def wannier_centres(wfa: WFArray, n_occ: int, axis_idx: int = 1) -> np.ndarray:
    """Hybrid Wannier centres in [0, 1) units of the cell, for the lowest n_occ bands.

    Returns: (nk_other_axis, n_occ) float.
    """
    phases = wfa.berry_phase(
        axis_idx=axis_idx, state_idx=list(range(n_occ)), contin=True, berry_evals=True
    )
    return np.asarray(phases) / (2 * np.pi)


def periodic_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance between Wannier centres on the circle [0, 1)."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(d, 1.0 - d)
