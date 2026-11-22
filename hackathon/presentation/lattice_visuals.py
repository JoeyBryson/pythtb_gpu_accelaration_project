"""Picture slides (no axes, no numbers to read): what the material looks like, and how many we simulate.

Run from hackathon/:  uv run python presentation/lattice_visuals.py
Writes to results/figures/slides/:

  6_edge_glow.png     Where the electrons can move, on a finite flake of the material: a ring around
                      the edge if topological, nothing if ordinary, and the ring survives disorder.
                      Computed here with PythTB (finite flake, exact diagonalisation on the CPU).
  7_sample_wall.png   "Every one of these is a different material": a wall of small disordered
                      lattices, coloured by each atom's random energy, as fed to the GPU in a batch.

Edge glow: Kane-Mele, lambda_SO = 0.3, lambda_R = 0, t = 1, a 24 x 24 flake (1,152 atoms).
Brightness = summed |psi|^2 of every state with |E| < 0.3, i.e. inside the bulk gap
(the clean gap edge is at |E| = |3 sqrt(3) * 0.3 - lambda_nu| = 0.56 for lambda_nu = 1.0).
Topological: lambda_nu = 1.0; ordinary: lambda_nu = 2.5; disorder: W = 2 (energies in [-W/2, W/2]).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pythtb.models import kane_mele
from style import INK_2, new_slide, save, titled

L_FLAKE, WINDOW, SEED = 24, 0.3, 1
GLOW = LinearSegmentedColormap.from_list(
    "glow", ["#14161c", "#3a2a6b", "#b23a8a", "#f08a4b", "#fff3c4"])  # dark -> warm light
ENERGY = LinearSegmentedColormap.from_list("energy", ["#2a78d6", "#f0efec", "#e34948"])
DARK = "#0e1016"


# ---------------------------------------------------------------- physics


def flake(lambda_nu: float, W: float, L: int = L_FLAKE, seed: int = SEED):
    """Finite L x L flake: atom positions (n, 2), in-gap density per atom (n,), states in the gap."""
    model = kane_mele(lambda_nu, 1.0, 0.3, 0.0).make_finite([0, 1], [L, L])
    H = np.asarray(model.hamiltonian(flatten_spin_axis=True))
    eps = np.random.default_rng(seed).uniform(-W / 2, W / 2, model.norb)  # one per atom, both spins
    e, v = np.linalg.eigh(H + np.diag(np.repeat(eps, 2)))
    in_gap = np.abs(e) < WINDOW
    dens = (np.abs(v[:, in_gap]) ** 2).reshape(model.norb, 2, -1).sum(axis=(1, 2))
    pos = np.asarray(model.orb_vecs) @ np.asarray(model.lattice.lat_vecs)
    return pos, dens, int(in_gap.sum())


def bonds(pos: np.ndarray) -> np.ndarray:
    """Nearest-neighbour bonds as line segments (m, 2, 2)."""
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    nn = np.min(d[d > 1e-9])
    i, j = np.where(np.triu(np.abs(d - nn) < 1e-6))
    return np.stack([pos[i], pos[j]], axis=1)


# ---------------------------------------------------------------- 6. edge glow


def slide_edge_glow() -> Path:
    cases = [
        (1.0, 0.0, "Topological material", "current flows around the edge"),
        (2.5, 0.0, "Ordinary material", "nowhere for current to flow"),
        (1.0, 2.0, "Topological, with imperfections", "the edge current survives"),
    ]
    fig, axes = new_slide(ncols=3)
    fig.patch.set_facecolor(DARK)
    plt.rcParams["savefig.facecolor"] = DARK  # style.apply() resets it for the next slide
    results = [flake(lam, W) for lam, W, *_ in cases]
    vmax = max(d.max() for _, d, _ in results)
    for ax, (_, _, head, sub), (pos, dens, _) in zip(axes, cases, results, strict=True):
        ax.add_collection(LineCollection(bonds(pos), colors="#262a35", linewidths=0.6, zorder=1))
        order = np.argsort(dens)  # brightest on top
        ax.scatter(pos[order, 0], pos[order, 1], c=dens[order], cmap=GLOW, vmin=0, vmax=vmax * 0.6,
                   s=16, linewidths=0, zorder=2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(head, color="white", fontsize=17, fontweight="bold", pad=4)
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", color="#c9c7c0",
                fontsize=14)
    fig.text(0.045, 0.93, "Where electricity can flow", fontsize=28, fontweight="bold",
             color="white", ha="left", va="top")
    fig.text(0.045, 0.855, "Brightness = where electrons at the key energy sit, in a flake of "
             "1,152 atoms", fontsize=16, color="#c9c7c0", ha="left", va="top")
    fig.text(0.045, 0.03, r"Kane-Mele flake 24×24, $\lambda_{SO}$ = 0.3, $\lambda_R$ = 0; "
             r"$\lambda_\nu$ = 1.0 (topological), 2.5 (ordinary); disorder W = 2. "
             "States with |E| < 0.3 (inside the gap). Computed with PythTB.",
             fontsize=10, color="#8d8b84", ha="left", va="bottom")
    return save(fig, "6_edge_glow.png", top=0.78, bottom=0.12, left=0.02, right=0.98)


# ---------------------------------------------------------------- 7. sample wall


def slide_sample_wall(rows: int = 10, cols: int = 20, n: int = 4) -> Path:
    """A wall of rows x cols small n x n supercells, each with its own random on-site energies."""
    from gpu_engine.reference import clean_kane_mele

    model = clean_kane_mele(n, rashba=0.0)
    pos = np.asarray(model.orb_vecs) @ np.asarray(model.lattice.lat_vecs)
    pos = (pos - pos.min(0)) / np.ptp(pos, axis=0).max()  # unit-sized tile
    seg = bonds(pos)
    fig, ax = new_slide()
    W = 4.0
    for r in range(rows):
        for c in range(cols):
            off = np.array([c * 1.15, -r * 0.82])  # tile is 1 x 0.58 plus its slant
            eps = np.random.default_rng(1000 + r * cols + c).uniform(-W / 2, W / 2, len(pos))
            ax.add_collection(LineCollection(seg + off, colors="#d9d8d2", linewidths=0.5, zorder=1))
            ax.scatter(pos[:, 0] + off[0], pos[:, 1] + off[1], c=eps, cmap=ENERGY, vmin=-W / 2,
                       vmax=W / 2, s=5, linewidths=0, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    titled(fig, "Every one of these is a different material",
           f"{rows * cols} shown. One disorder study needs 16,400; the GPU takes them in batches "
           "of many at once",
           "Each tile: a 4×4 supercell (32 atoms) shown; the runs use up to 20×20 (800 atoms). "
           "Colour = each atom's random energy (blue lower, red higher).")
    fig.text(0.955, 0.035, "× 16,400", fontsize=30, fontweight="bold", color=INK_2, ha="right",
             va="bottom")
    return save(fig, "7_sample_wall.png", top=0.79, bottom=0.12, left=0.045, right=0.955)


if __name__ == "__main__":
    for make in (slide_edge_glow, slide_sample_wall):
        print(f"wrote {make()}")
