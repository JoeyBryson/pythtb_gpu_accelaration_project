"""Batched pipeline, step 2: Wilson loops for every line of every configuration at once.

What PythTB does (wfarray.py, `berry_phase` -> `_berry_loop` -> `_wilson_loop`), one line
at a time in a Python loop:
  1. overlaps between neighbouring k-points along the line   M = <u(k)|u(k+dk)>
  2. unitary part of each overlap with an SVD                U = V @ Wh
  3. multiply them in order around the closed loop          W = U_0 @ U_1 @ ... @ U_{n-1}
  4. Wannier centres = -angle(eigenvalues of W) / 2pi

What we do differently (all lines and configurations in one batch):
  2. Newton-Schulz instead of SVD: only matrix multiplications, which GPUs are best at.
     Converges in a few steps because the overlaps are already close to unitary.
     Checked, and SVD used as a fallback for any matrix where it hasn't converged.
  3. Tree product instead of a sequential loop: multiply neighbouring pairs, then pairs of
     pairs, ... (log2(n) rounds). Matrix products are associative, so the order is kept.

Closing the loop at the zone boundary: the neighbour of the last k-point is the first one
plus a reciprocal lattice vector G, and in PythTB's convention u(k + G) = D u(k) with
D = diag(exp(-2 pi i G.tau)) (tau = orbital positions). Checked numerically to 1e-15 (21 Nov).

Shapes: C configs, nk_x x nk_y mesh, n = n_occ occupied states, N = all states (spin flattened)
    u      (C, nk_x, nk_y, n, N)   occupied states as rows
    M, U   (C, nk_x, nk_y, n, n)   one per link along k_y
    W      (C, nk_x, n, n)         one Wilson matrix per line
    wcc    (C, nk_x, n)            Wannier centres in [0, 1)
"""

from __future__ import annotations

import numpy as np
import torch
from pythtb import TBModel

# Newton-Schulz settings. Overlap singular values are ~0.75-1 on the meshes we use (see
# topology.RELIABLE_OVERLAP), where it converges quadratically; 12 steps is generous.
NS_MAX_STEPS = 12
NS_TOL = {torch.complex128: 1e-12, torch.complex64: 1e-5}


def boundary_phase(
    model: TBModel, G: tuple[float, float] = (0.0, 1.0), spin_blocks: bool = False
) -> np.ndarray:
    """D = exp(-2 pi i G.tau) for each state: u(k + G) = D * u(k).
    Shape (N,) with spin flattened, or (N/2,) per spin block (one entry per orbital)."""
    tau = np.asarray(model.orb_vecs)
    if not spin_blocks:
        tau = np.repeat(tau, 2, axis=0)  # same position for both spins
    return np.exp(-2j * np.pi * tau @ np.asarray(G))


def overlaps_along_ky(u: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """M[..., j] = <u(k_j)|u(k_{j+1})> along the k_y axis (dim -3), loop closed with `phase`.

    u: (C, nk_x, nk_y, n, N). Returns (C, nk_x, nk_y, n, n).
    """
    nxt = torch.roll(u, shifts=-1, dims=-3)  # u(k_{j+1}); last one wraps round to k_0 ...
    nxt[..., -1, :, :] = nxt[..., -1, :, :] * phase  # ... which is really k_0 + G
    return u.conj() @ nxt.transpose(-1, -2)


def unitary_part(M: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Closest unitary matrix to each M (the 'U' in M = U H), by Newton-Schulz:

        X <- X (3I - X^H X) / 2

    Stops early once every matrix has converged (checked every 2 steps from step 4: each
    check costs one extra product and, on a GPU, a sync). Returns (U, number of matrices
    that needed the SVD fallback).
    """
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    tol = NS_TOL[M.dtype] * M.shape[-1]
    X = M
    for step in range(1, NS_MAX_STEPS + 1):
        XhX = X.mH @ X
        if step >= 4 and step % 2 == 0:
            residual = torch.linalg.matrix_norm(XhX - eye, ord="fro")  # how far from unitary
            if bool((residual <= tol).all()):
                break
        X = 0.5 * X @ (3 * eye - XhX)
    # final convergence check: how far is each X from unitary?
    residual = torch.linalg.matrix_norm(X.mH @ X - eye, ord="fro")
    bad = residual > tol
    n_bad = int(bad.sum())
    if n_bad:  # rare: overlaps far from unitary (coarse mesh near a gap closing)
        V, _, Wh = torch.linalg.svd(M[bad], full_matrices=False)
        X[bad] = V @ Wh
    return X, n_bad


def unitary_part_svd(M: torch.Tensor) -> torch.Tensor:
    """PythTB's way: U = V @ Wh from the SVD. Used for comparison and as the fallback."""
    V, _, Wh = torch.linalg.svd(M, full_matrices=False)
    return V @ Wh


def ordered_product(U: torch.Tensor) -> torch.Tensor:
    """U_0 @ U_1 @ ... @ U_{m-1} along dim -3, by a tree of pairwise products.

    (..., m, n, n) -> (..., n, n). Each round multiplies neighbouring pairs in order, so
    the result equals the left-to-right product exactly (up to rounding).
    """
    while U.shape[-3] > 1:
        m = U.shape[-3]
        pairs = U[..., 0 : m - m % 2 : 2, :, :] @ U[..., 1 : m - m % 2 : 2, :, :]
        if m % 2:  # odd count: carry the last matrix to the next round, still at the end
            pairs = torch.cat([pairs, U[..., m - 1 : m, :, :]], dim=-3)
        U = pairs
    return U[..., 0, :, :]


def wannier_centres(
    u: torch.Tensor, model: TBModel, method: str = "newton-schulz", eig_method: str = "hermitian"
) -> tuple[torch.Tensor, dict]:
    """Wannier centres for every line of every configuration.

    u: (C, nk_x, nk_y, n_occ, N) occupied states from batched.solve_configs, or
       (C, 2, nk_x, nk_y, n_occ/2, N/2) with spin_blocks=True (Rashba = 0). Then each spin
       block gets its own Wilson loops (half-size matrices), and the two blocks' Wannier
       centres are combined per line: the same set the full problem gives, because the
       overlaps and Wilson matrices are block-diagonal in spin.
    Returns wcc (C, nk_x, n_occ) in [0, 1) on the CPU, sorted per line, and info
    (fallback count and seconds per sub-step).
    """
    on_gpu = u.is_cuda
    clock = _Clock(on_gpu)
    spin_blocks = u.ndim == 6

    phase = torch.as_tensor(
        boundary_phase(model, spin_blocks=spin_blocks), dtype=u.dtype, device=u.device
    )
    M = overlaps_along_ky(u, phase)
    clock.lap("overlaps")
    if method == "newton-schulz":
        U, n_fallback = unitary_part(M)
    else:
        U, n_fallback = unitary_part_svd(M), 0
    clock.lap("unitary")
    W = ordered_product(U)  # (C, nk_x, n, n)
    clock.lap("product")
    if eig_method == "hermitian":
        evals, eig_stats = unitary_eigenvalues(W)
        evals = evals.cpu()
    else:  # "general": the plain non-Hermitian eigensolver, on the CPU (slow on GPUs)
        evals = torch.linalg.eigvals(W.cpu())
        eig_stats = {"eig_retried": 0, "eig_cpu": int(W.shape[:-2].numel())}
    wcc = torch.remainder(-torch.angle(evals) / (2 * torch.pi), 1.0)
    if spin_blocks:  # (C, 2, nk_x, n/2) -> (C, nk_x, n): both spins' centres on each line
        wcc = wcc.movedim(1, -2).flatten(-2)
    wcc, _ = torch.sort(wcc, dim=-1)
    clock.lap("eigvals")
    info = {
        "ns_fallbacks": n_fallback, **eig_stats, "n_wilson": int(W.shape[:-2].numel()),
        "n_links": M.shape[:-2].numel(), **clock.laps,
    }
    return wcc, info


# Mixing weights for unitary_eigenvalues. Any fixed number not 0 works; irrational-looking
# ones make accidental ties (two different phases giving the same mixed value) unlikely.
# The second is used to retry the rare matrices where the first one fails.
_MIX = 0.6180339887
_MIX_RETRY = -1.4142135623
# Accuracy bar per eigenvalue (the step always runs in double precision, see below). W is
# normal, so if |W v - lambda v| <= tol for a unit vector v, lambda is within tol of a true
# eigenvalue (Bauer-Fike), i.e. the Wannier centre is within tol / 2pi. Checked per
# eigenvector, not per matrix: a matrix-wide norm let errors of ~1e-3 through at n = 512 in
# single precision (found 21 Nov).
EIG_TOL = 1e-10


def _hermitian_eig(W: torch.Tensor, mix: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One attempt: eigenvalues of unitary W via eigh of a Hermitian mix of W and W^H.
    Returns (eigenvalues (..., n), worst per-eigenvector residual |W v - lambda v| per matrix)."""
    H = 0.5 * (W + W.mH) + mix * (W - W.mH) / 2j
    _, V = torch.linalg.eigh(H)
    lam = (V.conj() * (W @ V)).sum(dim=-2)  # diagonal of V^H W V, one per eigenvector
    per_vector = torch.linalg.vector_norm(W @ V - V * lam.unsqueeze(-2), dim=-2)  # (..., n)
    return lam, per_vector.amax(dim=-1)


def unitary_eigenvalues(W: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Eigenvalues of unitary matrices with a Hermitian eigensolver (fast and batched on GPUs).

    A unitary W has eigenvalues e^{i theta} and an orthonormal set of eigenvectors. The
    Hermitian matrix  H = (W + W^H)/2 + MIX (W - W^H)/(2i)  has the SAME eigenvectors
    (eigenvalues cos theta + MIX sin theta), so we diagonalise H with `eigh` and read each
    eigenvalue of W back as v^H W v.

    Always done in DOUBLE precision, even in a single-precision run: in single precision
    eigh can't separate eigenvalues closer than ~1e-6 and the Rayleigh read-back loses
    accuracy (1e-3 errors at n = 512, 21 Nov). The Wilson matrices are small and few
    (one n_occ x n_occ per line), so double costs little. A single-precision W is first made
    exactly unitary in double with two Newton-Schulz steps (it moves by ~1e-5, below the 1e-4
    single-precision tolerance); otherwise its tiny non-unitarity would fail the check.

    It can still go wrong when two different phases give (nearly) the same cos + MIX sin, so
    eigh mixes their eigenvectors. Every eigenvector is checked, and failures go through:
      1. retry on the same device with a different MIX (cheap: few matrices)
      2. only if that also fails: the general (non-Hermitian) solver on the CPU (slow)

    W: (..., n, n). Returns (eigenvalues (..., n) complex128, {"eig_retried", "eig_cpu"} counts).
    """
    Wd = W.to(torch.complex128)
    if W.dtype != torch.complex128:
        eye = torch.eye(W.shape[-1], dtype=Wd.dtype, device=Wd.device)
        for _ in range(2):  # Newton-Schulz polish: nearest unitary matrix, in double
            Wd = 0.5 * Wd @ (3 * eye - Wd.mH @ Wd)
    lam, residual = _hermitian_eig(Wd, _MIX)
    bad = residual > EIG_TOL
    stats = {"eig_retried": int(bad.sum()), "eig_cpu": 0}
    if stats["eig_retried"]:
        lam_retry, res_retry = _hermitian_eig(Wd[bad], _MIX_RETRY)
        still_bad = res_retry > EIG_TOL
        stats["eig_cpu"] = int(still_bad.sum())
        if stats["eig_cpu"]:
            lam_retry[still_bad] = torch.linalg.eigvals(Wd[bad][still_bad].cpu()).to(
                lam_retry.device
            )
        lam[bad] = lam_retry
    return lam, stats


class _Clock:
    """Lap timer for sub-steps; synchronises the GPU so the laps are real."""

    def __init__(self, on_gpu: bool):
        import time

        self._now, self.on_gpu, self.laps = time.perf_counter, on_gpu, {}
        self._last = self._sync_now()

    def _sync_now(self) -> float:
        if self.on_gpu:
            torch.cuda.synchronize()
        return self._now()

    def lap(self, name: str) -> None:
        now = self._sync_now()
        self.laps[f"t_{name}"] = now - self._last
        self._last = now
