"""Spin-block split (Rashba = 0) gives the same answers as the full problem.

Kept small (2x2 supercells) so the suite stays fast. Tolerances: 1e-10 energies and 1e-8
Wannier centres in double precision.
"""

import numpy as np
import pytest

from gpu_engine import get_backend
from gpu_engine.batched import solve_configs
from gpu_engine.pipeline import run_batch
from gpu_engine.reference import clean_kane_mele
from gpu_engine.wilson import wannier_centres

NO_RASHBA = {"rashba": 0.0}


def test_split_refuses_when_rashba_couples_the_spins():
    with pytest.raises(ValueError, match="rashba = 0"):
        solve_configs(clean_kane_mele(2), 4, 1.0, [0], get_backend("double"), spin_blocks=True)


def test_same_energies_and_wannier_centres_as_full_problem():
    be, model, nk, W, seeds = get_backend("double"), clean_kane_mele(2, **NO_RASHBA), 6, 3.0, [0, 1]
    e_full, u_full = solve_configs(model, nk, W, seeds, be)
    e_blk, u_blk = solve_configs(model, nk, W, seeds, be, spin_blocks=True)
    np.testing.assert_allclose(e_blk.cpu().numpy(), e_full.cpu().numpy(), atol=1e-10)

    wcc_full, _ = wannier_centres(u_full, model)
    wcc_blk, _ = wannier_centres(u_blk, model)
    d = np.abs(wcc_blk.cpu().numpy() - wcc_full.cpu().numpy())  # .cpu(): may be on a GPU
    assert np.minimum(d, 1 - d).max() < 1e-8  # both sorted per line; distance on the circle


@pytest.mark.parametrize("precision", ["double", "single"])
def test_same_z2_as_full_problem(precision):
    be = get_backend(precision)
    for W in (0.5, 6.0, 20.0):  # topological, mixed, trivial
        full = run_batch(2, W, 8, list(range(6)), be, check=False, **NO_RASHBA)
        blk = run_batch(2, W, 8, list(range(6)), be, check=False, spin_blocks=True, **NO_RASHBA)
        assert (blk.z2 == full.z2).all()
