"""Intermediate results of the pipeline, plotted as they come out of the code (paper-style figures).

Run from hackathon/:  uv run python presentation/pipeline_figures.py        (~5 min on a laptop CPU)
Writes to results/figures/pipeline/:

  step1_ribbon_bands.png   SOLVE. Energy bands of a ribbon (periodic along x, 24 cells wide), every
                           state coloured by which edge it lives on. Ordinary: empty gap.
                           Topological: edge states cross the gap. With disorder: they still do.
  step2_winding.png        WILSON LOOP -> INVARIANT. For 200 disordered samples per panel, the
                           spin-up polarisation sum_n ybar_n(k_x) followed across the Brillouin zone
                           (the quantity gpu_engine.spin_chern winds to get C_up). A sample is
                           topological if its line ends one cell away from where it started.
  step3_newton_schulz.png  OUR ALGORITHM. Newton-Schulz iterations on real overlap matrices from
                           disordered samples: singular values go to 1, the distance from a unitary
                           matrix falls to rounding error.

Model: Kane-Mele, t = 1, lambda_SO = 0.3, lambda_R = 0. Disorder: on-site energies uniform in
[-W/2, W/2]. Nothing here is fitted or smoothed; each figure states its parameters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from pythtb.models import kane_mele

from gpu_engine.reference import clean_kane_mele
from gpu_engine.spin_chern import boundary_phase_up, clean_up_hamiltonians, up_disorder

OUT = Path(__file__).resolve().parent.parent / "results" / "figures" / "pipeline"
# the team deck's figure colours (261121hackathon.pptx): blue - grey - salmon red, on white
BLUE, RED, GREY, INK = "#3b82f6", "#f08080", "#c9cdd4", "#000000"
LINE_BLUE = "#1d5fd0"
EDGE = LinearSegmentedColormap.from_list("edge", ["#1d5fd0", "#c9cdd4", "#e23b3b"])  # left edge, bulk, right edge


def paper_style() -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica Neue", "DejaVu Sans"], "font.size": 12,
        "axes.linewidth": 0.8, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "mathtext.default": "regular",
    })


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- step 1: ribbon bands


def ribbon_bands(lambda_nu: float, W: float = 0.0, width: int = 24, length: int = 1, nk: int = 241,
                 seed: int = 3):
    """Bands of a ribbon: periodic along x (`length` cells per period), `width` cells across.

    Returns k (nk,), E (nk, n_states), side (nk, n_states) in [-1, 1]: -1 = all of the state's
    weight within 3 cells of the left edge, +1 = the right edge, ~0 = bulk (or both edges).
    """
    model = kane_mele(lambda_nu, 1.0, 0.3, 0.0)
    if length > 1:
        model = model.make_supercell([[length, 0], [0, 1]])
    model = model.make_finite([1], [width])
    k = np.linspace(0.0, 1.0, nk)
    H = np.asarray(model.hamiltonian(k[:, None], flatten_spin_axis=True))
    eps = np.random.default_rng(seed).uniform(-W / 2, W / 2, model.norb)
    e, v = np.linalg.eigh(H + np.diag(np.repeat(eps, 2)))
    y = np.repeat(np.asarray(model.orb_vecs)[:, 1], 2)  # position across the ribbon, in cells
    prob = np.abs(v) ** 2  # (nk, orbital, state)
    left = prob[:, y < y.min() + 3, :].sum(axis=1)
    right = prob[:, y > y.max() - 3, :].sum(axis=1)
    return k, e, right - left


def fig_ribbon_bands() -> Path:
    cases = [
        ("ordinary", {"lambda_nu": 2.5}),
        ("topological", {"lambda_nu": 1.0}),
        ("topological + disorder (W = 2)", {"lambda_nu": 1.0, "W": 2.0, "length": 6}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, (title, kw) in zip(axes, cases, strict=True):
        k, e, side = ribbon_bands(**kw)
        kk = np.repeat(k[:, None], e.shape[1], axis=1)
        order = np.argsort(np.abs(side), axis=None)  # edge states drawn last, on top
        sc = ax.scatter(kk.ravel()[order], e.ravel()[order], c=side.ravel()[order], cmap=EDGE,
                        vmin=-1, vmax=1, s=1.6, linewidths=0, rasterized=True)
        ax.set_title(f"({'abc'[list(axes).index(ax)]}) {title}")
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xticks([0, 0.5, 1], ["0", "π", "2π"])
        ax.set_xlabel("momentum along the ribbon")
    axes[0].set_ylabel("energy (t)")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.015, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["left edge", "bulk", "right edge"])
    cbar.outline.set_linewidth(0.6)
    return save(fig, "step1_ribbon_bands.png")


# ---------------------------------------------------------------- step 2: winding


def polarisation_flow(L: int, lambda_nu: float, W: float, seeds: list[int], nk: int,
                      chunk: int = 20) -> np.ndarray:
    """Spin-up polarisation followed across k_x, for each sample. Returns (samples, nk + 1):
    starts at 0, ends at the integer C_up. Same steps as gpu_engine.spin_chern.chern_from_states
    (overlap determinants along k_y, wrapped differences along k_x), in double precision on the CPU.
    """
    model = clean_kane_mele(L, delta=lambda_nu, t=1.0, soc=0.3, rashba=0.0)
    H0 = torch.as_tensor(clean_up_hamiltonians(model, nk))  # (nk*nk, n, n)
    phase = torch.as_tensor(boundary_phase_up(model))
    flows = []
    for start in range(0, len(seeds), chunk):
        d = torch.as_tensor(up_disorder(model.norb, W, seeds[start : start + chunk]))
        H = H0.unsqueeze(0) + torch.diag_embed(d.to(H0.dtype)).unsqueeze(1)
        _, v = torch.linalg.eigh(H)
        n = H.shape[-1] // 2
        u = v[..., :n].transpose(-1, -2).reshape(len(d), nk, nk, n, H.shape[-1])
        nxt = torch.roll(u, shifts=-1, dims=2)
        nxt[:, :, -1] = nxt[:, :, -1] * phase
        sign = torch.linalg.slogdet(u.conj() @ nxt.transpose(-1, -2))[0]
        p = -torch.angle(sign).sum(dim=2) / (2 * torch.pi)  # (C, nk), mod 1
        step = torch.remainder(torch.roll(p, shifts=-1, dims=1) - p + 0.5, 1.0) - 0.5
        flows.append(torch.cumsum(step, dim=1).numpy())
    flow = np.concatenate(flows)
    return np.concatenate([np.zeros((len(flow), 1)), flow], axis=1)


def fig_winding(L: int = 8, lambda_nu: float = 1.65, n_samples: int = 200, nk: int = 16) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    kx = np.linspace(0, 1, nk + 1)
    for ax, W in zip(axes, (1.0, 2.0, 3.0), strict=True):
        flow = polarisation_flow(L, lambda_nu, W, list(range(n_samples)), nk)
        winds = np.abs(np.round(flow[:, -1])) == 1
        orient = np.sign(np.round(flow[winds, -1]).sum()) or 1.0  # plot the winding upwards
        for line, w in zip(flow, winds, strict=True):
            ax.plot(kx, orient * line, color=LINE_BLUE if w else "#8a8a8a", lw=0.7, alpha=0.3)
        ax.set_title(f"({'abc'[list(axes).index(ax)]}) disorder W = {W:g}")
        ax.text(0.04, 0.93, f"{int(winds.sum())} of {n_samples} wind", transform=ax.transAxes,
                va="top", color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.35, 1.35)
        ax.set_xticks([0, 0.5, 1], ["0", "π", "2π"])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(r"$k_x$")
        print(f"  winding: W = {W}: {int(winds.sum())}/{n_samples} wind", flush=True)
    axes[0].set_ylabel("shift of the electrons' centre (cells)")
    fig.subplots_adjust(wspace=0.06)
    return save(fig, "step2_winding.png")


# ---------------------------------------------------------------- step 3: Newton-Schulz


def real_overlaps(L: int = 8, lambda_nu: float = 1.65, W: float = 2.0, n_samples: int = 4,
                  nk: int = 6) -> torch.Tensor:
    """Overlap matrices <u(k_j)|u(k_j+1)> of the spin-up block, from disordered samples at the
    mesh the sweeps use. Returns (n_samples * nk * nk, n, n) complex128."""
    model = clean_kane_mele(L, delta=lambda_nu, t=1.0, soc=0.3, rashba=0.0)
    H0 = torch.as_tensor(clean_up_hamiltonians(model, nk))
    phase = torch.as_tensor(boundary_phase_up(model))
    d = torch.as_tensor(up_disorder(model.norb, W, list(range(n_samples))))
    H = H0.unsqueeze(0) + torch.diag_embed(d.to(H0.dtype)).unsqueeze(1)
    _, v = torch.linalg.eigh(H)
    n = H.shape[-1] // 2
    u = v[..., :n].transpose(-1, -2).reshape(n_samples, nk, nk, n, H.shape[-1])
    nxt = torch.roll(u, shifts=-1, dims=2)
    nxt[:, :, -1] = nxt[:, :, -1] * phase
    return (u.conj() @ nxt.transpose(-1, -2)).reshape(-1, n, n)


def fig_newton_schulz(steps: int = 10) -> Path:
    X = real_overlaps()
    eye = torch.eye(X.shape[-1], dtype=X.dtype)
    smin, resid = [], []
    for _ in range(steps + 1):
        smin.append(torch.linalg.svdvals(X)[:, -1].numpy())  # smallest singular value per matrix
        resid.append(torch.linalg.matrix_norm(X.mH @ X - eye, ord="fro").numpy())
        X = 0.5 * X @ (3 * eye - X.mH @ X)  # the Newton-Schulz step (gpu_engine.wilson)
    smin, resid = np.array(smin), np.array(resid)  # (steps + 1, matrices)
    it = np.arange(steps + 1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    a.plot(it, smin, color=LINE_BLUE, lw=0.6, alpha=0.35)
    a.set_title("(a)", loc="left")
    b.set_title("(b)", loc="left")
    a.axhline(1.0, color=INK, lw=0.6, ls=":")
    a.set_xlabel("Newton-Schulz iteration")
    a.set_ylabel("smallest singular value")
    a.set_xlim(0, steps)
    b.semilogy(it, np.maximum(resid, 1e-17), color=LINE_BLUE, lw=0.6, alpha=0.35)
    b.set_xlabel("Newton-Schulz iteration")
    b.set_ylabel(r"distance from unitary, $\|X^\dagger X - 1\|$")
    b.set_xlim(0, steps)
    fig.subplots_adjust(wspace=0.32)
    print(f"  newton-schulz: {smin.shape[1]} matrices of {X.shape[-1]}x{X.shape[-1]}, smallest "
          f"singular value at start {smin[0].min():.3f}", flush=True)
    return save(fig, "step3_newton_schulz.png")


if __name__ == "__main__":
    paper_style()
    for make in (fig_ribbon_bands, fig_newton_schulz, fig_winding):
        print(f"wrote {make()}", flush=True)
