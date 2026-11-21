"""Lowest-order Born approximation for the critical disorder Wc of the TAI transition.

Orth et al. (2016), Eq. (2) with Sigma = 0 on the right-hand side, evaluated numerically
from the clean Kane-Mele H(k) (lambda_R = 0, E_F = 0). For uncorrelated on-site disorder the
self-energy is local and diagonal:

    Sigma_alpha = <eps^2> * G_loc,alpha(E_F),   G_loc = mean over the BZ of (E_F - H(k))^-1

At E_F = 0 particle-hole symmetry gives Sigma_A = -Sigma_B, so the only effect is a
renormalised staggered potential

    lambda_nu_bar = lambda_nu + (Sigma_A - Sigma_B) / 2 = lambda_nu - <eps^2> * g,
    g = (G_B - G_A) / 2  (> 0)

and the transition (gap closing) is at lambda_nu_bar = 3*sqrt(3)*lambda_SO, i.e.

    <eps^2>_c = (lambda_nu - 3*sqrt(3)*lambda_SO) / g.

Disorder conventions: ours (and Favata's) eps in [-W/2, W/2], so <eps^2> = W^2/12.
Orth's eps in [-W, W], so <eps^2> = W_Orth^2/3 and W_ours = 2*W_Orth.

Normalisation (to check): taking Orth's prefactor (a/2pi)^2 * integral over the BZ literally,
with a = lattice constant of the honeycomb lattice, gives 2/sqrt(3) times the BZ average.
With that factor we reproduce Orth's Fig. 5 Born curve (e.g. Wc_Orth = 2.05 at 2.1t);
without it (the plain BZ average, standard lowest-order Born) Wc is ~7.5% higher.
Both are printed. Eq. (3), the analytic low-energy form, does NOT reproduce Fig. 5 as printed.

Run from hackathon/:  uv run python analysis/born.py
"""

from __future__ import annotations

import numpy as np
from pythtb.models import kane_mele

SOC = 0.3
CLEAN_BOUNDARY = 3 * np.sqrt(3) * SOC  # 1.5588 t


def local_greens(delta: float, soc: float = SOC, nk: int = 800, e_f: float = 0.0) -> np.ndarray:
    """Diagonal of G_loc(E_F) for the clean unit cell. Returns (4,) real: A up, A dn, B up, B dn."""
    model = kane_mele(delta, 1.0, soc, 0.0)
    k = (np.arange(nk) + 0.5) / nk  # offset mesh: avoids k-points exactly at a Dirac point
    kk = np.stack(np.meshgrid(k, k, indexing="ij"), -1).reshape(-1, 2)
    e, v = np.linalg.eigh(model.hamiltonian(kk, flatten_spin_axis=True))
    G = np.einsum("kin,kn,kjn->ij", v, 1.0 / (e_f - e), v.conj()) / len(kk)
    return np.real(np.diag(G))


def born_wc(delta: float, soc: float = SOC, orth_normalisation: bool = False) -> float:
    """Critical disorder in OUR convention (eps in [-W/2, W/2]). 0 if already topological."""
    m = delta - 3 * np.sqrt(3) * soc
    if m <= 0:
        return 0.0
    g_A, _, g_B, _ = local_greens(delta, soc)
    g = (g_B - g_A) / 2
    if orth_normalisation:
        g *= 2 / np.sqrt(3)
    return float(np.sqrt(12 * m / g))  # <eps^2> = W^2 / 12


if __name__ == "__main__":
    print(f"clean boundary 3*sqrt(3)*lambda_SO = {CLEAN_BOUNDARY:.4f} t\n")
    print("lambda_nu   Wc_ours(Orth norm.)  Wc_ours(BZ average)   Wc_Orth(Orth norm.)")
    for d in np.arange(1.55, 2.101, 0.05):
        w_o = born_wc(d, orth_normalisation=True)
        w_b = born_wc(d)
        print(f"  {d:.2f}        {w_o:6.3f}               {w_b:6.3f}              {w_o / 2:6.3f}")
