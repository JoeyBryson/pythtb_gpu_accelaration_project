"""Figures for a non-specialist audience, made from our actual model and results.

Run from hackathon/:  uv run python presentation/layman_figures.py
Writes PNGs to results/figures/:

  lattice_clean_vs_disordered.png  "Real materials are imperfect": the same honeycomb lattice,
                                   perfect vs with a random energy on every atom
  samples_grid.png                 "Thousands of samples, one answer": 100 disordered samples,
                                   each outlined by our topological-or-not answer

Parameters: Kane-Mele, lambda_SO = 0.3, lambda_R = 0 (team choice), on-site 1.0, disorder
W = 7 (our convention, energies uniform in [-W/2, W/2]); chosen because small samples there
split roughly half topological / half not, which makes the point visually.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import common  # noqa: F401  (macOS warning filter)

from gpu_engine import get_backend
from gpu_engine.batched import disorder_shifts
from gpu_engine.pipeline import run_batch
from gpu_engine.reference import clean_kane_mele

FIGURES = Path(__file__).resolve().parent.parent / "results" / "figures"
PARAMS = {"rashba": 0.0}  # tutorial delta = 1.0, soc = 0.3, with Rashba off
W = 7.0

# Colours (validated default palette, dataviz skill). Diverging blue <-> red with a grey
# midpoint for "lower / higher energy than normal"; aqua vs muted grey for the answer.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BOND = "#c3c2b7"
ENERGY = LinearSegmentedColormap.from_list("energy", ["#2a78d6", "#f0efec", "#e34948"])
TOPO, NORMAL = "#1baf7a", "#898781"


# ---------------------------------------------------------------- geometry


def atom_positions(supercell: int) -> np.ndarray:
    """Cartesian positions of the atoms in an n x n supercell (one per orbital). (norb, 2)"""
    model = clean_kane_mele(supercell, **PARAMS)
    return np.asarray(model.orb_vecs) @ np.asarray(model.lattice.lat_vecs)


def bonds(pos: np.ndarray) -> np.ndarray:
    """Nearest-neighbour bonds inside the supercell (no wrap-round), as line segments."""
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    nn = np.min(d[d > 1e-9])
    i, j = np.where(np.triu(np.abs(d - nn) < 1e-6))
    return np.stack([pos[i], pos[j]], axis=1)


def draw_lattice(ax, pos, energies=None, atom_size=60, vmax=W / 2):
    ax.add_collection(LineCollection(bonds(pos), colors=BOND, linewidths=1.2, zorder=1))
    colours = "#d9d8d2" if energies is None else ENERGY((energies + vmax) / (2 * vmax))
    ax.scatter(pos[:, 0], pos[:, 1], s=atom_size, c=colours, edgecolors="white",
               linewidths=0.8, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")


def onsite_energies(supercell: int, seed: int) -> np.ndarray:
    """The random energy added to each atom: the same numbers the simulation uses."""
    model = clean_kane_mele(supercell, **PARAMS)
    return disorder_shifts(model.norb, W, [seed])[0][0::2]  # one value per atom (spin up)


# ---------------------------------------------------------------- figure 1


def fig_clean_vs_disordered(n: int = 8, seed: int = 0) -> Path:
    pos = atom_positions(n)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.patch.set_facecolor(SURFACE)
    draw_lattice(a, pos)
    a.set_title("A perfect crystal: a regular, repeating pattern", color=INK, fontsize=13, pad=10)
    draw_lattice(b, pos, onsite_energies(n, seed))
    b.set_title("A real material: random imperfections everywhere", color=INK, fontsize=13, pad=10)

    sm = plt.cm.ScalarMappable(cmap=ENERGY, norm=plt.Normalize(-W / 2, W / 2))
    cbar = fig.colorbar(sm, ax=b, fraction=0.035, pad=0.02, ticks=[-W / 2, 0, W / 2])
    cbar.ax.set_yticklabels(["lower\nenergy", "normal", "higher\nenergy"], color=INK_2, fontsize=9)
    cbar.outline.set_visible(False)

    fig.text(0.5, 0.03, "Same honeycomb lattice (like graphene). On the right each atom gets a random "
             "energy, as in a real, imperfect sample.\nWe simulate thousands of such samples to see "
             "whether the material's special 'topological' property survives.",
             ha="center", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, "lattice_clean_vs_disordered.png")


# ---------------------------------------------------------------- figure 2


def fig_samples_grid(n: int = 4, n_samples: int = 100, nk: int = 24) -> Path:
    seeds = list(range(n_samples))
    result = run_batch(n, W, nk, seeds, get_backend("double"), **PARAMS)
    topo = result.z2.astype(bool)
    sure = result.reliable
    pos = atom_positions(n)
    pos = pos - pos.mean(axis=0)

    cols = 10
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8.4))
    fig.patch.set_facecolor(SURFACE)
    for k, ax in enumerate(axes.flat):
        if k >= n_samples:
            ax.axis("off")
            continue
        colour = MUTED if not sure[k] else (TOPO if topo[k] else NORMAL)
        tint = "#ffffff" if not sure[k] else ("#d3efe4" if topo[k] else "#e8e7e2")
        ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, zorder=0,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=tint, edgecolor=colour,
                                    linewidth=2.2 if sure[k] else 1.4,
                                    linestyle="-" if sure[k] else "--"))
        draw_lattice(ax, pos, onsite_energies(n, seeds[k]), atom_size=16)
        ax.set_xlim(pos[:, 0].min() - 0.6, pos[:, 0].max() + 0.6)
        ax.set_ylim(pos[:, 1].min() - 0.6, pos[:, 1].max() + 0.6)

    n_topo, n_norm, n_unsure = int((topo & sure).sum()), int((~topo & sure).sum()), int((~sure).sum())
    fig.suptitle(f"100 imperfect samples: {n_topo} topological, {n_norm} normal, "
                 f"{n_unsure} uncertain", color=INK, fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.5, 0.935, "Each tile is one simulated sample with its own random imperfections. "
             "Green = topological, grey = normal, dashed outline = our checks say 'not sure'.",
             ha="center", color=INK_2, fontsize=11)
    fig.text(0.5, 0.012, f"Kane-Mele model, {n}x{n} supercells, disorder W = {W:g}, "
             f"Rashba off, {nk}x{nk} k-mesh. Uncertain = flagged by our reliability checks "
             "(would be re-run on a finer mesh).", ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.04, wspace=0.06, hspace=0.12)
    print(f"samples: {n_topo} topological, {n_norm} normal, {n_unsure} uncertain")
    return save(fig, "samples_grid.png")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"wrote {fig_clean_vs_disordered()}")
    print(f"wrote {fig_samples_grid()}")
