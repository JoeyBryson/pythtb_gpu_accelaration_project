"""Batched pipeline, step 1: build H(k) for many disorder configurations at once and
diagonalise them all in one call. Runs on CPU or GPU (PyTorch), same code.

The trick that makes disorder cheap to batch
--------------------------------------------
Anderson disorder only adds a random number to each orbital's on-site energy. On-site
terms don't depend on k (in PythTB's convention their phase factor is exactly 1), so for
configuration c:

    H[c, k] = H_clean[k] + diag(d_c)

We build H_clean(k) once with PythTB, then every configuration is the same stack plus a
different diagonal: one broadcast addition on the device, no Python loop over
configurations and no rebuilding of models.

Array shapes (C = configurations, K = k-points, N = 4 n^2 states, spin flattened):
    H_clean  (K, N, N)      complex
    d        (C, N)         real, random on-site shift per state
    H        (C, K, N, N)   complex
"""

from __future__ import annotations

import numpy as np
import torch
from pythtb import Mesh, TBModel

from .backend import Backend


def k_mesh(nk: int) -> np.ndarray:
    """The same gamma-centred nk x nk mesh the CPU reference uses. Returns (nk, nk, 2)."""
    mesh = Mesh(["k", "k"])
    mesh.build_grid(shape=(nk, nk), gamma_centered=True)
    return np.asarray(mesh.points)


def clean_hamiltonians(model: TBModel, nk: int) -> np.ndarray:
    """H_clean(k) at every mesh point, from PythTB (built once, on the CPU).

    Returns (nk*nk, N, N) complex, k-points in the mesh's row-major order.
    """
    k = k_mesh(nk).reshape(-1, 2)
    return model.hamiltonian(k, flatten_spin_axis=True)


def disorder_shifts(norb: int, W: float, seeds: list[int]) -> np.ndarray:
    """Random on-site energies for each configuration, identical to the CPU reference.

    Same random numbers as `reference.disordered_kane_mele` for the same seed: one value
    per orbital, uniform in [-W/2, W/2], the same for both spins. Returns (C, 2*norb), in
    PythTB's flattened (orbital, spin) order: orb0 up, orb0 down, orb1 up, ...
    """
    per_orbital = np.stack(
        [np.random.default_rng(s).uniform(-W / 2, W / 2, size=norb) for s in seeds]
    )
    return np.repeat(per_orbital, 2, axis=1)  # same shift for spin up and down


def batched_hamiltonians(H_clean: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """H[c, k] = H_clean[k] + diag(shifts[c]).  (K,N,N) + (C,N) -> (C,K,N,N)."""
    return H_clean.unsqueeze(0) + torch.diag_embed(shifts.to(H_clean.dtype)).unsqueeze(1)


def solve(H: torch.Tensor, n_occ: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonalise every matrix in the batch in one call.

    H: (..., N, N). Returns
      energies: (..., N) real, ascending
      u_occ:    (..., n_occ, N) complex, occupied states as rows (PythTB's layout)
    """
    energies, vecs = torch.linalg.eigh(H)  # vecs: eigenvectors are columns
    u_occ = vecs[..., :n_occ].transpose(-1, -2)
    return energies, u_occ


def configs_per_chunk(nk: int, N: int, backend: Backend, budget_gb: float = 4.0) -> int:
    """How many configurations fit in one batch without running out of GPU memory.

    One configuration's Hamiltonians take nk^2 * N^2 * (bytes per complex number). eigh needs
    extra workspace for eigenvectors and scratch, so we allow ~4x that per configuration.
    budget_gb is deliberately well under a T4's 16 GB.
    """
    bytes_per_complex = 16 if backend.precision == "double" else 8
    per_config = 4 * nk * nk * N * N * bytes_per_complex
    return max(1, int(budget_gb * 1e9 // per_config))


def split_spin_blocks(H: torch.Tensor) -> torch.Tensor:
    """(..., N, N) -> (..., 2, N/2, N/2): the spin-up and spin-down blocks.

    Only valid when spin up and down don't mix (Rashba = 0): then H, in PythTB's
    (orbital, spin) order, has nonzero entries only between equal spins, so the even rows/
    columns (spin up) and odd ones (spin down) are two independent problems of half the size.
    Raises if the blocks are coupled, rather than silently giving wrong answers.
    """
    coupling = H[..., 0::2, 1::2].abs().amax()
    if coupling > 1e-12:
        raise ValueError(
            f"spin blocks are coupled (max |H_up,down| = {float(coupling):.2e}): "
            "the spin-block split needs rashba = 0"
        )
    return torch.stack([H[..., 0::2, 0::2], H[..., 1::2, 1::2]], dim=-3)


def solve_configs(
    model: TBModel, nk: int, W: float, seeds: list[int], backend: Backend,
    chunk: int | None = None, spin_blocks: bool = False, shift_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole step 1 for C configurations of one clean model, `chunk` configurations per batch
    (default: as many as fit in memory, see configs_per_chunk).

    Returns energies (C, nk, nk, N) and occupied states on the device:
      spin_blocks=False: (C, nk, nk, n_occ, N)
      spin_blocks=True:  (C, 2, nk, nk, n_occ/2, N/2), i.e. per spin block (Rashba = 0 only).
        Each block is half the size, so diagonalising costs ~1/4 of the full problem. With a
        gap at half filling, each spin block contributes exactly half its states (the count is
        fixed across the zone while the gap is open, and equal for both spins at k = 0, where
        time reversal maps one block onto the other), so each takes its lowest n_occ/2.

    shift_fn(norb, W, seeds) -> (C, 2*norb) on-site shifts; default `disorder_shifts`.
    Lets analysis code try other disorder ensembles (e.g. analysis/ensemble.pinned_shifts).
    """
    n_occ = model.norb  # half filling
    H0 = torch.as_tensor(clean_hamiltonians(model, nk), dtype=backend.cdtype, device=backend.device)
    N = H0.shape[-1]
    if spin_blocks:
        H0 = split_spin_blocks(H0)  # (K, 2, N/2, N/2)
    chunk = chunk or configs_per_chunk(nk, N // 2 if spin_blocks else N, backend)
    shift_fn = shift_fn or disorder_shifts

    all_e, all_u = [], []
    for start in range(0, len(seeds), chunk):  # usually just one or a few big batches
        part = seeds[start : start + chunk]
        d = torch.as_tensor(
            shift_fn(model.norb, W, part), dtype=backend.rdtype, device=backend.device
        )
        if spin_blocks:
            d = d[:, 0::2].unsqueeze(1)  # same shift for both spins: (c, 1, N/2), broadcasts
            H = H0.unsqueeze(0) + torch.diag_embed(d.to(H0.dtype)).unsqueeze(1)  # (c,K,2,.,.)
            e, u = solve(H, n_occ // 2)
            e = torch.sort(e.flatten(-2), dim=-1).values  # both spins' energies, as one list
        else:
            e, u = solve(batched_hamiltonians(H0, d), n_occ)
        all_e.append(e)
        all_u.append(u)
    C = len(seeds)
    energies = torch.cat(all_e).reshape(C, nk, nk, N)
    if spin_blocks:
        u_occ = torch.cat(all_u).reshape(C, nk, nk, 2, n_occ // 2, N // 2).movedim(3, 1)
    else:
        u_occ = torch.cat(all_u).reshape(C, nk, nk, n_occ, N)
    return energies, u_occ
