"""Deciding "topological or not": the Z2 invariant from Wannier-centre flow.

Background (plain language)
---------------------------
The Wilson loop along k_y gives, for each k_x, the positions of the electrons along
y inside the unit cell: the hybrid Wannier centres. Picture them as beads on a ring
(the ring is the unit cell, which wraps round). As k_x goes from 0 to pi:

  - normal insulator (Z2 = 0):      the beads wobble and come back
  - topological insulator (Z2 = 1): pairs of beads swap and wind round the ring

To decide automatically we use "largest-gap tracking" (Soluyanov & Vanderbilt,
PRB 83, 235401 (2011), the method used by Z2Pack): follow the middle of the biggest
empty gap between beads, and count how many beads it jumps over from one k_x to the
next. An odd total means Z2 = 1.

Status: method chosen by the GPU-engine side; to be confirmed by the physics team.

Known weak spot: the tracking miscounts if the beads move further in one k_x step than
the gaps between them. This gets *worse* for bigger supercells, because there are more
beads (2n^2 for an n x n supercell) so the gaps are smaller. Found 21 Nov: a clean
12x12 supercell on a 4x4 mesh gave Z2 = 0 (wrong); 6x6 and finer gave 1 (right).
`z2_margin` (below) catches this: always check it before trusting a Z2 answer.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

# Below this margin, a Z2 answer is flagged as unreliable (see z2_margin). Chosen from the
# 21 Nov test: the one wrong answer had margin 0.002, every right answer had >= 0.38.
RELIABLE_MARGIN = 0.1


def largest_gap_midpoint(centres: np.ndarray) -> tuple[float, float]:
    """Middle of the biggest gap between beads on the ring [0, 1).

    centres: (n_occ,) Wannier centres in [0, 1)
    Returns (midpoint, gap_size).
    """
    x = np.sort(centres)
    # gaps between neighbours, including the wrap-around gap from the last bead to the first
    gaps = np.diff(np.append(x, x[0] + 1.0))
    i = int(np.argmax(gaps))
    return (x[i] + gaps[i] / 2) % 1.0, float(gaps[i])


def z2_from_wannier_flow(flow: np.ndarray) -> int:
    """Z2 invariant (0 or 1) from Wannier centres over half the Brillouin zone.

    flow: (n_kx, n_occ) Wannier centres in [0, 1), rows ordered k_x = 0 ... 0.5 (pi).
          The first and last rows must be exactly k_x = 0 and k_x = pi.
    """
    jumps = 0
    z_prev, _ = largest_gap_midpoint(flow[0])
    for centres in flow[1:]:
        z_next, _ = largest_gap_midpoint(centres)
        lo, hi = sorted((z_prev, z_next))
        # Beads the gap midpoint jumped over. With an even number of beads (always true
        # here: time reversal pairs them up), counting inside [lo, hi] or outside it gives
        # the same parity, so it doesn't matter which way round the ring the jump went.
        jumps += int(np.count_nonzero((centres > lo) & (centres < hi)))
        z_prev = z_next
    return jumps % 2


def z2_margin(flow: np.ndarray) -> float:
    """How safely the gap tracking worked: 1 = very safe, 0 = it may have miscounted.

    At each k_x step, take the gap midpoint from the previous step and see how close the
    beads of the new step come to it. If a bead lands right on it, we can't tell which
    side the bead passed, and the count may be wrong. The margin is that closest
    distance, relative to half the largest gap at the new step; we report the worst step.

    flow: (n_kx, n_occ) as for z2_from_wannier_flow.
    Below RELIABLE_MARGIN: don't trust the Z2 answer; rerun with more k_x points.
    """
    worst = np.inf
    for prev, nxt in pairwise(flow):  # each step: (row k, row k+1)
        z_prev, _ = largest_gap_midpoint(prev)
        _, gap_next = largest_gap_midpoint(nxt)
        d = np.abs(nxt - z_prev) % 1.0
        closest = np.minimum(d, 1.0 - d).min()  # distance on the ring
        worst = min(worst, closest / (gap_next / 2))
    return float(worst)
