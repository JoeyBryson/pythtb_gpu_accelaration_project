# Part 2 goal: reproduce the disorder-driven topological transition at λν = 1.65t

## The goal

Use the GPU pipeline to measure the **fraction of disorder configurations that are topological (Z₂ = 1)**
as a function of disorder strength W, at λν = 1.65t, λ_SO = 0.3t, λ_R = 0 and E_F = 0. Run it for several
supercell sizes. Show on one plot that the material is:

1. **trivial at weak disorder**,
2. **topological from W ≈ 2** (the topological Anderson insulator), and
3. **trivial again near W ≈ 7** (Anderson insulator).

Overlay Favata & Marrazzo (2023) Fig. 7 (left) and the E_F = 0 curve of Orth et al. (2016) Fig. 2 (bottom
middle). Report how long the same dataset takes on the best CPU baseline compared with the GPU.

## Conventions

| Quantity | Value / definition |
|---|---|
| Model | `kane_mele(delta=1.65, t=1, soc=0.3, rashba=0)`. Mapping checked in `VALIDATION.md`. |
| Disorder | **Ours: εᵢ uniform in [−W/2, W/2]**, the same for both spins. This is the same convention as Favata. |
| Orth's W | Orth uses [−W, W], so **W_ours = 2 × W_Orth**. Orth's numbers below are already doubled. |
| Supercell | L × L cells (2L² sites). Same meaning of L as Favata. |
| Per-config answer | Z₂ from `topology.z2_from_wannier_flow`, kept only if `ConfigResult.z2_reliable` |

## Reference values (exact plotted data, extracted from the PDFs' vector graphics)

**Favata & Marrazzo 2023, Fig. 7 (left)**, Δ/λ_SO = 5.5 (= λν 1.65t), λ_R = 0, M = 50 to 100 configs.
The full curves are in `analysis/reference_data/favata2023_fig7_left.csv`.

| | L = 15 | L = 42 |
|---|---|---|
| **Lower transition (C_s = 0.5)** | **W ≈ 1.99** | **W ≈ 1.92** |
| TAI plateau (C_s ≥ 0.9) | 2.25 to 6.0 | 2.25 to 6.25 |
| **Upper transition (C_s = 0.5)** | **W ≈ 6.89** | **W ≈ 7.00** |

**Orth et al. 2016**, the λν = 1.65t point of Fig. 5 (`analysis/reference_data/orth2016_fig5.csv`):
**Wc = 2.14 ± 0.07** in our units. In Fig. 2, G at E_F = 0 reaches the 2G₀ plateau there and stays on it
up to the edge of the plot (W_ours = 6).

The two references agree on the lower transition to within about 10%.

## What we measure

For each (L, W), with M configurations:

- **f = fraction topological** = (number of reliable configs with Z₂ = 1) / (number of reliable configs).
  Error bar: √(f(1 − f)/M_reliable).
- **u = fraction unreliable** (no gap, mesh too coarse, or unsafe gap tracking). Always reported next to f.
- **Wc_low and Wc_high** = the W values where f crosses 0.5 (linear interpolation), with bootstrap errors.

## Run

| λν | W (ours) | L | M per point | Total configs |
|---|---|---|---|---|
| 1.65 | 0 to 8 in steps of 0.25 (33 points, on Favata's grid) | 8, 12, 16 (+24 if the GPU allows) | 200 | ≈ 20,000 |

For scale, the laptop CPU takes 0.75 s, 4.3 s and 22.5 s per config at L = 8, 12 and 16. That makes the run
about 50 h serial, or about 12 h on 10 cores.

## How we'll know we've got there

**Prerequisites**
- [ ] Clean model (W = 0) at λν = 1.65 gives Z₂ = 0 at every L used.
- [ ] Small CPU check (L = 8, W ∈ {0, 1, 2, 3, 5, 8}, 20 configs each) shows trivial → topological → trivial.
- [ ] GPU gives the **same Z₂ as `reference.py` config by config** (same seeds) on at least 20 configs at
      W ∈ {1, 2, 3}.
- [ ] Mesh convergence: doubling nk changes f by less than its error bar at W = 2 and 3.

**The result**
- [ ] f ≤ 0.1 for W ≤ 1.2 (trivial).
- [ ] **Wc_low ∈ [1.6, 2.6]** at the largest L. That is Favata's 1.92 and Orth's 2.14, ±0.4 to allow for our
      smaller L.
- [ ] f ≥ 0.9 for 2.8 ≤ W ≤ 5.5 (TAI plateau).
- [ ] **Wc_high ∈ [6.3, 7.7]** at the largest L (Favata's 7.0 ± 0.7).
- [ ] At least 2 sizes, M ≥ 200 per point, with u reported.
- [ ] **One figure:** f(W) for each L with error bars, Favata's L = 15 and L = 42 curves, Orth's E_F = 0 curve
      (G/2G₀), and Wc_low and Wc_high marked.

**Performance**
- [ ] Time for the full run on the GPU, against `cpu_baseline.py parallel` (and `serial`) on the same machine.
      Label any CPU time that is extrapolated from a subset as such.

## Caveats to state

1. **Different observables.** Orth measures edge conductance on a ribbon, Favata the mean of a single-point
   spin Chern number (not an integer at finite L), and we the fraction of configs with Z₂ = 1. We compare
   *where* the transitions are, not the shape of the curves.
2. **Spectral gap vs mobility gap.** Disorder can fill the gap with localised states. Our band-gap check flags
   these configs, so u will rise near the transitions. Decide the policy before the production run (see
   `VALIDATION.md`) and report u.
3. **Coarse meshes near transitions** can give a wrong Z₂. This is caught by `z2_margin` and `min_overlap`,
   and checked by the mesh-convergence test.
4. **Reference values come from the figures' vector data** (`analysis/extract_paper_data.py`). Neither paper
   gives a table. Favata's crossings are limited by their 0.25 W grid.
