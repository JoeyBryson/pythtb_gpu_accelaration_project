# Goal: reproduce the paper's central result with an independent GPU method

## The goal, in one line

Show that **disorder turns a trivial Kane-Mele insulator topological at W ≈ 2.0 (our units; "W ≈ t" in
Orth et al.)**. We locate this critical disorder **by finite-size scaling** (something the paper does not do),
using our batched GPU Wilson-loop pipeline, and cross-check it against two independent published methods.

**Pitch line:** "We reproduce the central result of Orth et al. (2016), that disorder makes a trivial Kane-Mele
insulator topological at W ≈ t, with an independent method. We locate the transition by finite-size scaling,
which needs thousands of disorder samples: practical on a GPU, impractical on a CPU."

## Why this is the right target

- Orth et al.'s headline finding is that the topological Anderson insulator (TAI) exists in the Kane-Mele model.
  Their demonstration is λν = 1.65t: trivial when clean, topological "at W ≈ t" (Fig. 2, middle panel), and
  quantified as Wc in Fig. 5.
- Two independent methods have measured it (conductance; single-point spin Chern number). Our Wilson-loop Z₂ is
  a third. The lowest-order Born approximation gives a theory lower bound.
- Neither paper extracts a critical point properly. Orth uses one ribbon size and does not define Wc; Favata
  only shows curves for two sizes. Finite-size scaling is a real addition.

## Fixed conditions

| Parameter | Value | PythTB / code |
|---|---|---|
| Hopping | t = 1 | `t=1` |
| Intrinsic spin-orbit coupling | λ_SO = 0.3t | `soc=0.3` |
| Staggered sublattice potential | **λν = 1.65t** (= 5.5 λ_SO; trivial when clean, 0.09t above the boundary 3√3·λ_SO = 1.5588t) | `delta=1.65` |
| Rashba | λ_R = 0 | **`rashba=0`** (the code defaults to 0.25, so always pass it) |
| Filling | half filling, E_F = 0 | n_occ = norb |
| Disorder | Anderson, εᵢ uniform in **[−W/2, W/2]**, the same for both spins | `disordered_kane_mele`, `batched.disorder_shifts` |

**Watch the factor of 2.** Orth uses [−W, W], so **W_ours = 2 × W_Orth**. All values below are in our units
unless marked. Favata uses the same convention as us.

## Target values

| Transition | Reference | Method | W (ours) |
|---|---|---|---|
| **Wc₁: trivial → TAI** (main target) | Orth et al. 2016, Fig. 5 | Two-terminal conductance, 93a × 150a ribbon, 100 configs | **2.14 ± 0.07** |
| | Favata & Marrazzo 2023, Fig. 7 (left) | Single-point spin Chern number, L = 42 / 15, M = 50 | **1.92 / 1.99** (C_s = 0.5, our interpolation) |
| | Lowest-order Born (`analysis/born.py`) | Theory, clean H(k) | 1.68 (Orth's normalisation) to 1.80 (BZ average): a lower bound |
| **Wc₂: TAI → Anderson insulator** (secondary) | Favata & Marrazzo 2023, Fig. 7 (left) | as above | **7.00 / 6.89** (L = 42 / 15) |

Expected: **Wc₁ ≈ 2.0 ± 0.15** (the two methods span 1.92–2.14). **Wc₂ ≈ 7.0 ± 0.3.**
Reference data: `analysis/reference_data/orth2016_fig5.csv` and `favata2023_fig7_left.csv`, extracted from the
PDFs' vector graphics (`analysis/extract_paper_data.py`).

## Method

**Per configuration** (`gpu_engine.pipeline.run_batch`):
1. Build the clean H(k) once, then add a random diagonal for each configuration.
2. Run a batched `eigh` over configurations × k-points.
3. Run batched Wilson loops along k_y (Newton-Schulz with SVD fallback, tree product) to get the Wannier centres.
4. Compute Z₂ from the Wannier-centre flow between k_x = 0 and π, plus the reliability checks.

**Per (L, W)**, over M configurations:
- **f** = fraction topological = (reliable configs with Z₂ = 1) / (reliable configs)
- **u** = fraction unreliable, split by reason (no gap, `z2_margin`, `min_overlap`)

Use the **same seeds at every W**: each configuration keeps its disorder pattern, scaled by W. Error bars come
from a bootstrap over seeds.

**Locating Wc (finite-size scaling):**
1. **Crossing.** f(W, L) curves for different L cross at Wc, because f → 0 below and f → 1 above as L → ∞. Each
   pair (L₁, L₂) gives W*. Report the drift of W* with L, and extrapolate it.
2. **Collapse (stronger, optional).** Fit f = F((W − Wc)·L^(1/ν)) for Wc and ν, with bootstrap errors.
   Recipe: Mildner et al. (arXiv:2312.16689), disordered Haldane model, L ≤ 17, 500 configs,
   Wc = 3.58 ± 0.02, ν = 2.42 ± 0.11. With λ_R = 0, Kane-Mele is two decoupled Chern copies, so ν ≈ 2.4–2.6
   is likely (to check).
3. **Quick estimate.** The single-L midpoint (f = 0.5) for each L. Report it, but don't quote it as the
   critical point.

## Stages

| # | Stage | Where | Done when |
|---|---|---|---|
| 0 | **Go/no-go:** `reference.run_config(8, W, nk=6, seed=s, delta=1.65, rashba=0)`, W ∈ {0, 1, 2, 3, 5, 8}, 20 seeds | CPU, about 5 min | f ≈ 0 → partial → ≈ 1 → ≈ 0, with u reported |
| 1 | **GPU equals CPU:** `run_batch` and `run_config`, same seeds, W ∈ {1, 2, 3}, single and double | GPU | Identical Z₂ for every reliable config |
| 2 | **Z₂ cross-check:** spin-up Wilson-loop winding C↑ per config. Z₂ must equal C↑ mod 2 for every reliable config. | GPU | 100% agreement, or disagreements explained |
| 3 | **Pilot:** W ∈ {1.5, 2.0, 2.5, 6.5, 7.0, 7.5}, L ∈ {8, 12}, nk = 6 vs 10, M = 50. Measure seconds per config. | GPU | Mesh fixed (f stable within error bars); time budget known |
| 4 | **Coarse scan:** W 0 to 8 in steps of 0.25 (Favata's grid), L ∈ {8, 12, 16}, M = 200 | GPU | f(W) for each L; rough Wc₁ and Wc₂ |
| 5 | **Finite-size-scaling runs:** about 10 W values within ±0.6 of each coarse Wc, L ∈ {8, 12, 16} (+24 if spin-split), M = 500 | GPU | Crossings W*(L₁, L₂) with errors |
| 6 | **Analysis:** crossings, drift, optional collapse, robustness (below) | CPU | Wc₁ (and Wc₂) ± statistical ± systematic error |
| 7 | **CPU comparison:** `cpu_baseline.py parallel` on the same machine, on a subset, extrapolated and labelled as such | same machine | Measured time to the whole dataset, GPU vs best CPU |
| 8 | **Figures and write-up** | — | See below |

**Compute estimate** (T4, single precision, from RESULTS R4a and R5; to be measured in stage 3): about 0.1, 0.5
and 1.3 s per config at L = 8, 12 and 16. Stage 4 is about 3.5 h and stage 5 about 3 h per transition on one T4,
roughly halved with both T4s. L = 24 needs the spin split (about 8x cheaper diagonalisation). Laptop CPU
equivalent, all 10 cores: roughly 15 h per transition at L ≤ 16.

**Raw output:** append every config to `results/tai_165_raw.csv` as it finishes:
`L, nk, W, seed, z2, z2_margin, min_overlap, band_gap, reliable, precision, device, timestamp`.

## Robustness checks (stage 6)

- **Reliability bias:** recompute Wc with the unreliable configs counted as Z₂ = 0, then as Z₂ = 1. The shift is
  a systematic error.
- **Mesh:** Wc at nk = 6 vs 10 at the largest L.
- **Precision:** single vs double precision on a subset near Wc₁.
- **Crossing drift:** W* for (8, 12), (12, 16) and (16, 24). Report the trend, and use it as a systematic error.
- **Gap policy:** `reliable` uses the *indirect* gap at the moment; the Wilson loop only needs a *direct* gap at
  each k. Log both and agree the policy with Oscar before stage 4.

## Figures

1. **The key-result figure.** f(W) for each L, with error bars, for W from 0 to 8.
   - On the same axes: Favata's L = 15 and 42 curves, a band for Orth's Wc = 2.14 ± 0.07, and a dashed line
     for the Born value 1.68.
   - Top axis: W_Orth = W/2.
   - Lower panel: u(W).
2. **Finite-size scaling.** A zoom on Wc₁: curves crossing, with W* against L as an inset. If the collapse is
   done, F(x) with all sizes on one curve.
3. **Performance.** Time to the whole dataset, GPU against the best CPU (and serial PythTB), same machine.

## How we'll know we've got there

**Validity (must pass first)**
- [ ] Stage 0 shows trivial → topological → trivial at L = 8.
- [ ] Clean model (W = 0) gives Z₂ = 0 at every L used.
- [ ] GPU Z₂ = CPU Z₂ for every reliable config (stage 1), and Z₂ = C↑ mod 2 for every config (stage 2).
- [ ] Mesh convergence: f is unchanged within its error bars at 2x nk.

**The key result**
- [ ] **Wc₁ from finite-size scaling in [1.8, 2.3]**, consistent with Orth (2.14 ± 0.07) and Favata (1.92), with
      statistical and systematic errors stated.
- [ ] Wc₁ above the Born lower bound (≈ 1.7).
- [ ] f ≤ 0.1 for W ≤ 1.2, and f ≥ 0.9 on the plateau (about 2.8 to 5.5).
- [ ] At least 3 sizes; the transition sharpens with L, and the crossings are resolved beyond their errors.

**Secondary**
- [ ] **Wc₂ in [6.5, 7.5]** (Favata 7.0).
- [ ] Optional: ν from the collapse, compared with ≈ 2.4 (Mildner, Haldane).

**Performance**
- [ ] Measured GPU time for the full dataset against the best CPU baseline on the same machine.

## Caveats to state

1. **Different observables.** Orth measures edge conductance on a ribbon, Favata the mean of a single-point spin
   Chern number (not an integer at finite L), and we the fraction of configs with Z₂ = 1. They should agree on
   *where* the transition is, not on the shape of the curves.
2. **The references are not rigorous critical points.** Orth: one size, Wc undefined. Favata: our interpolation
   of their curves. So "≈ 2.0" is a target with about ±0.15 uncertainty, and our finite-size-scaling value may
   be the better-defined number.
3. **Small sizes.** L ≤ 16 (24) against Orth's 28k sites. Crossings may drift because the clean critical point is
   close (1.5588t). We report the drift.
4. **Reliability flags rise near Wc,** which is exactly where finite-size scaling is sensitive. This is why the
   reliability-bias check exists.
5. **λν = 1.65t is a choice** made by Orth and reused by Favata, not a property of a real material.
6. **"First" claims.** As far as we can tell, no finite-size-scaling analysis of the Kane-Mele TAI has been
   published. Say it that way, never "first".
