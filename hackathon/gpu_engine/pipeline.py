"""The whole batched pipeline for many disorder configurations: the GPU counterpart of
reference.run_config, but for C configurations at once.

    build + solve (batched.py) -> Wilson loop (wilson.py) -> Z2 per config (topology.py)

Same seeds give the same materials as the CPU reference, so results can be compared one
configuration at a time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .backend import Backend, sync
from .batched import configs_per_chunk, k_mesh, solve_configs
from .reference import clean_kane_mele
from .topology import RELIABLE_MARGIN, RELIABLE_OVERLAP, z2_from_wannier_flow, z2_margin
from .wilson import boundary_phase, overlaps_along_ky, wannier_centres


@dataclass
class BatchResult:
    """Per-configuration answers plus stage timings for the whole batch."""

    z2: np.ndarray  # (C,) 0 or 1
    z2_margin: np.ndarray  # (C,) gap-tracking safety (see topology.z2_margin)
    min_overlap: np.ndarray  # (C,) mesh-fineness check, along k_y (see note in run_batch)
    band_gap: np.ndarray  # (C,)
    timings: dict = field(default_factory=dict)  # seconds per stage
    ns_fallbacks: int = 0

    @property
    def reliable(self) -> np.ndarray:
        return (
            (self.band_gap > 0)
            & (self.z2_margin >= RELIABLE_MARGIN)
            & (self.min_overlap >= RELIABLE_OVERLAP)
        )


def half_bz_rows(nk: int) -> np.ndarray:
    """Row indices (k_x) running 0 ... 0.5 on the gamma-centred mesh (same as half_bz_flow)."""
    kx = k_mesh(nk)[:, 0, 0] % 1.0
    rows = np.where(kx <= 0.5 + 1e-12)[0]
    return rows[np.argsort(kx[rows])]


def run_batch(
    supercell: int, W: float, nk: int, seeds: list[int], backend: Backend,
    check: bool = True, **params,
) -> BatchResult:
    """Z2 for every configuration (one per seed), timing each stage on the device.

    check=True also computes the mesh-fineness check (smallest singular value of the
    overlaps). Note: here along k_y only (incl. the boundary link), whereas the CPU
    reference uses interior links along both axes, so the numbers differ slightly.
    It is timed separately and not part of the pipeline's cost.
    """
    if nk % 2:
        raise ValueError("Z2 needs an even nk (k = pi on the mesh)")
    model = clean_kane_mele(supercell, **params)
    n_occ, N = model.norb, 2 * model.norb
    t = {"solve": 0.0, "wilson": 0.0, "z2": 0.0, "check": 0.0}
    z2, margin, gap, min_overlap, fallbacks = [], [], [], [], 0
    rows = half_bz_rows(nk)

    # Process configurations in chunks that fit in device memory. The Wilson step holds
    # several (n_occ x n_occ) matrices per link, so budget for both steps together.
    chunk = max(1, configs_per_chunk(nk, N, backend) // 2)
    for start in range(0, len(seeds), chunk):
        part = seeds[start : start + chunk]

        sync(backend)
        t0 = time.perf_counter()
        energies, u = solve_configs(model, nk, W, part, backend, chunk=len(part))
        sync(backend)
        t1 = time.perf_counter()
        wcc, info = wannier_centres(u, model)
        sync(backend)
        t2 = time.perf_counter()

        # Z2 from the Wannier-centre flow over half the zone, per configuration (on the CPU)
        flows = wcc[:, rows].double().cpu().numpy()  # (c, nk/2 + 1, n_occ)
        z2 += [z2_from_wannier_flow(f) for f in flows]
        margin += [z2_margin(f) for f in flows]
        g = energies[..., n_occ].amin(dim=(1, 2)) - energies[..., n_occ - 1].amax(dim=(1, 2))
        gap += g.double().cpu().tolist()
        t3 = time.perf_counter()
        t["solve"] += t1 - t0
        t["wilson"] += t2 - t1
        t["z2"] += t3 - t2
        fallbacks += info["ns_fallbacks"]

        if check:  # untimed diagnostic: is the mesh fine enough?
            phase = torch.as_tensor(boundary_phase(model), dtype=u.dtype, device=u.device)
            sv = torch.linalg.svdvals(overlaps_along_ky(u, phase))  # (c, nk, nk, n_occ)
            min_overlap += sv.amin(dim=(1, 2, 3)).double().cpu().tolist()
            t["check"] += time.perf_counter() - t3
        else:
            min_overlap += [np.nan] * len(part)
        del energies, u, wcc

    t["total"] = t["solve"] + t["wilson"] + t["z2"]
    return BatchResult(
        np.array(z2), np.array(margin), np.array(min_overlap), np.array(gap), t, fallbacks
    )
