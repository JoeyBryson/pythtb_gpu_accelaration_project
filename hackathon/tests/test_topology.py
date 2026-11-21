"""Z2 sanity checks against cases where we know the answer.

Known answers:
  - tutorial on-site 1.0 -> topological (Z2 = 1); 2.5 -> trivial (Z2 = 0)
  - a clean supercell has the same topology as the unit cell (band folding only)
  - weak disorder keeps topology; very strong disorder destroys it (CLAUDE.md sanity check)
"""

import numpy as np
import pytest

from gpu_engine.reference import run_config
from gpu_engine.topology import largest_gap_midpoint, z2_from_wannier_flow


@pytest.mark.parametrize(
    ("supercell", "delta", "expected"),
    [(1, 1.0, 1), (1, 2.5, 0), (2, 1.0, 1), (2, 2.5, 0)],
)
def test_clean_kane_mele_z2(supercell, delta, expected):
    assert run_config(supercell, W=0, nk=8, delta=delta).z2 == expected


@pytest.mark.parametrize("nk", [8, 12, 20])
def test_z2_does_not_depend_on_mesh(nk):
    assert run_config(1, W=0, nk=nk).z2 == 1


@pytest.mark.parametrize(("W", "expected"), [(0.5, 1), (20.0, 0)])
def test_disorder_sanity(W, expected):
    results = [run_config(2, W=W, nk=8, seed=s).z2 for s in range(5)]
    assert results == [expected] * 5


def test_too_coarse_mesh_is_flagged():
    """Regression (21 Nov): clean 12x12 on a 4x4 mesh gave Z2 = 0, which is wrong.
    The margin must flag it; one step finer must be right and trusted."""
    coarse = run_config(12, W=0, nk=4)
    assert not coarse.z2_reliable
    fine = run_config(12, W=0, nk=6)
    assert fine.z2 == 1 and fine.z2_reliable


def test_near_clean_transition_coarse_mesh_is_flagged():
    """Regression (21 Nov): Orth et al. 2016 say lambda_nu < 3*sqrt(3)*lambda_SO = 1.559 is
    topological (rashba = 0). At lambda_nu = 1.50 a 12x12 mesh gives the wrong Z2; the
    overlap check must flag it, and a 72x72 mesh must give the right, trusted answer."""
    params = {"delta": 1.50, "soc": 0.3, "rashba": 0.0}
    coarse = run_config(1, W=0, nk=12, **params)
    assert coarse.z2 == 0 and not coarse.z2_reliable
    fine = run_config(1, W=0, nk=72, **params)
    assert fine.z2 == 1 and fine.z2_reliable


def test_odd_mesh_gives_no_z2():
    assert run_config(1, W=0, nk=7).z2 is None


def test_largest_gap_wraps_round_the_ring():
    mid, gap = largest_gap_midpoint(np.array([0.1, 0.2, 0.8, 0.9]))
    assert gap == pytest.approx(0.6)
    assert mid == pytest.approx(0.5)
    mid, gap = largest_gap_midpoint(np.array([0.4, 0.45, 0.5, 0.55]))
    assert gap == pytest.approx(0.85)
    assert mid == pytest.approx(0.975)  # gap crosses 1 -> 0


def test_winding_flow_is_topological():
    """Synthetic flow: one Kramers pair swaps across the ring; the other stays put."""
    k = np.linspace(0, 1, 11)
    flow = np.stack([(0.25 + 0.5 * k) % 1, (0.25 - 0.5 * k) % 1], axis=1)
    assert z2_from_wannier_flow(flow) == 1
    still = np.stack([0.25 + 0.05 * np.sin(np.pi * k), 0.25 - 0.05 * np.sin(np.pi * k)], axis=1)
    assert z2_from_wannier_flow(still) == 0
