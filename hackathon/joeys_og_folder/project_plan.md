# Project plan: GPU mode for PythTB, simulating thousands of disordered materials

Combined plan, merging `project_idea.md` (science scope, validation, risks) with `CLAUDE.md` (engineering
rules, measured results, timeline). Where the two disagreed, the choice made here is marked **[decision]**.
Please confirm or change these as a team. Anything uncertain is marked **(to check)**.

## One line

We build a batched GPU pipeline for PythTB's topology calculation (Hamiltonian, diagonalisation, Wilson loops,
invariant). It runs thousands of disorder configurations at once. We use it to locate the disorder-driven
topological transition in the Kane-Mele model and compare it with Orth et al. (2016).

**Pitch for non-physicists:** real materials are never perfect. Disorder can even switch a material's
topological state. Finding where that switch happens means simulating thousands of random imperfect samples.
We make that fast enough to be routine. Quote the measured "before/after" time, not an estimate.

## Two deliverables, so we don't depend on matching the paper exactly

1. **Tool:** a GPU mode for PythTB, in `hackathon/gpu_engine/`. It sits alongside PythTB and does not modify it.
2. **Demo:** use the tool to find the transition and compare it with the published result.

If the demo does not match the paper well, the tool is still the main deliverable, and an explained mismatch is
still a result. The judges explicitly value informative negative results.

## Key decisions

| # | Decision | Why |
|---|---|---|
| 1 | **[decision] Core regime λ_R = 0, invariant = spin Chern number C_s per spin block**, computed from Wannier-centre winding. Report Z₂ = C_s mod 2 alongside it. | S_z is conserved at λ_R = 0, so each spin block is a Chern problem (simple, robust). The blocks are half-size, so diagonalisation is roughly 8x cheaper. For λ_R = 0, Z₂ and C_s mod 2 agree. λ_R > 0 (real Z₂) is a stretch goal. |
| 2 | **Whole pipeline on GPU**, including building H(k). | Build H₀(k) once in PythTB and add the disorder diagonal on the GPU. This avoids thousands of `TBModel` objects and host→device copies. |
| 3 | **Newton-Schulz polar factor plus tree-reduced product**, with a convergence guard and SVD fallback. | Measured 6.5x (c128) and 58x (c64) over CPU, while GPU SVD was *slower* than CPU. That negative result is worth explaining in the pitch. |
| 4 | **Different observable from the paper, on purpose.** | Orth uses Kwant conductance on an open sample. We use a bulk invariant on a periodic supercell. We validate *where the transition is*, not the conductance. |
| 5 | **[decision] Finite-size study: multiple L is core, the collapse fit is a stretch goal.** Core = at least 3 sizes, with a Wc estimate from the curve crossings. Stretch = the full collapse fit for Wc and ν. | Multiple sizes are needed anyway to tell real transitions from finite-size smearing, and they show off the GPU. The collapse fit is the first thing to drop if short of time. |
| 6 | **Honest baselines.** The headline is against PythTB as-is. The *main* claim is against the best CPU version (batched NumPy plus multiprocessing). | Judges will look for a credible baseline. We quote the speed-up against each. |
| 7 | **Careful novelty language.** "As far as we can tell", never "first". | BerryEasy already does GPU Wilson loops for disordered supercells (CuPy, PyBinding). Our angle: PythTB-based, Newton-Schulz and tree reduction, and batching over disorder × k × lines. |

## Physics setup

- **Model:** `pythtb.models.kane_mele(esite, t, soc, rashba)`, with `esite` the staggered potential (Orth's λν),
  t = 1, soc = λ_SO = 0.3, rashba = 0. Note that `reference.py` currently defaults to the tutorial parameters
  (rashba = 0.25), so the demo must override them. Whether PythTB's `soc` matches Orth's λ_SO **(to check)**:
  verify by checking the clean gap closes at λν = 3√3·λ_SO ≈ 1.56t.
- **Disorder:** Anderson, εᵢ ~ U[-1, 1], H(k) = H₀(k) + W·diag(ε), as in Orth Eq. (1). `spinv` uses
  [-w/2, w/2], so w = 2W. Keep the convention explicit in every comparison.
- **Supercell:** L x L cells (2L² orbitals per spin block), with k acting as twisted boundary conditions. Larger L
  needs fewer k-points. The required mesh at each L is **(to check)** by convergence tests.
- **Gap check:** E_F = 0 must lie in a gap in each realization. We count and report realizations where it does
  not, instead of silently dropping them.

### Targets from the paper (λ_SO = 0.3t, λ_R = 0)

- λν = 1.45t: topological at W = 0 and stays topological.
- λν = 1.65t: trivial at W = 0, topological around W ≈ t. **This is the main demo.**
- λν = 1.85t: the transition happens, but the topological phase is fragile.
- Wc vs λν (Fig. 5) and the lowest-order Born approximation (Eqs. 2–3, computable from the clean H(k)
  without disorder). Fig. 5 values are read off the PDF **(to check)**.

### How we differ from the paper

| Aspect | Orth et al. | Ours | Consequence |
|---|---|---|---|
| Observable | Two-terminal conductance (Kwant) | Wilson-loop spin Chern number | Compare transition location, not G |
| Geometry | 93a x 150a open sample with leads | Periodic L x L supercell | Bulk invariant stands in for edge conductance |
| Size | About 28k sites | L ≈ 8–24 core (maybe larger) | Finite-size broadening, studied with several L |
| Averaging | Mean G over 100 configs | Fraction topological / mean C_s over 100s–1000s | Our own statistic, with error bars |

## The pipeline (all on GPU, batched, no Python loops)

```
H0(k) from PythTB (once) ─► + W·diag(ε) per realization ─► split spin blocks
  ─► batched torch eigh over (realization × k) ─► occupied states
  ─► overlaps between neighbouring k ─► Newton-Schulz polar factor (guarded, SVD fallback)
  ─► tree-reduced product along each line ─► eigenvalue phases = Wannier centres
  ─► winding → C_s per spin block, + gap check → classify or flag each realization
```

Batch over realizations × W × λν, and chunk to fit GPU memory. Support complex128 and complex64 through a flag.

### Engineering rules (from CLAUDE.md, still apply)

- Keep data on the GPU between steps. Leading batch dimensions throughout.
- Newton-Schulz: `X <- 0.5 X (3I - XᴴX)`, about 8 iterations. Monitor ‖XᴴX - I‖. Near gap closings the singular
  values drop and Newton-Schulz slows down, which is exactly where the transition is. Report the fallback rate.
- Tree product: pairwise rounds, order preserved, odd lengths carried.
- Reproduce PythTB's loop closure at the zone boundary exactly, and test against it.
- Spinful states: `flatten_spin_axis=True`. Compare only unique quantities (eigenvalues, projectors, Wannier
  centres modulo 1), never raw eigenvectors.
- Memory: elements × 16 B (c128) × 2–4 for workspace. Chunk accordingly.

## Validation

1. **Clean limit:** C_s on the correct side of λν = 3√3·λ_SO.
2. **GPU vs CPU:** Wannier centres and invariants against `reference.py` (PythTB `berry_phase`) on 20+ single
   realizations. Tolerances 1e-8 (c128) and 1e-4 (c64), written next to each test.
3. **Newton-Schulz vs SVD** on sampled realizations, especially near Wc.
4. **Precision:** fraction of realizations whose classification differs between c64 and c128, by W.
   Wannier-centre agreement on clean cases is not enough.
5. **External cross-check:** `spinv` (Favata, Baù & Marrazzo, PythTB-based) on a few small cases.
6. **Born approximation:** gap edges vs the numerical gap at small W.
7. **Sanity checks:** weak disorder at λν = 1.45t stays topological. Very strong disorder becomes trivial.

## Benchmarks

| Baseline | Role |
|---|---|
| PythTB as-is (per-realization `TBModel`, per-line Wilson loops) | Headline "before" number |
| PythTB `use_tensorflow=True` | What the existing GPU option gives |
| Batched NumPy (vectorised over lines and k) | Fair serial CPU baseline |
| Batched NumPy plus multiprocessing over realizations | Fair parallel CPU baseline |
| BerryEasy (stretch) | Closest competitor |

Rules: synchronise before stopping timers (`timed()` does this), warm up first, include host↔device copies,
and state the hardware. Compare at **equal accuracy** (the same converged k-mesh), not equal mesh. Report the time
breakdown (eigh vs matmul vs transfer), realizations per GPU-hour, maximum batch size, and **time to the full
phase diagram**, best CPU vs best GPU. Re-benchmark on the event GPU before choosing a precision strategy
(the T4 has very weak FP64).

## Measured so far (pre-event, Kaggle T4, clean supercells, 31x31 mesh)

- CPU time split is roughly 70% solve, 30% Wilson loop. At 8x8: solve 31.5 s, Wilson 13.5 s.
- eigh at 8x8: NumPy 24.6 s; torch c128 13.3 s (1.9x), c64 2.5 s (10x). CuPy is slower at large sizes.
- Wilson loop at 8x8: CPU 8.7 s; GPU SVD 15.2 s (slower); Newton-Schulz 1.33 s c128 (6.5x), 0.15 s c64 (58x).
- c64 errors: eigenvalues about 4e-4 (8x8), Wannier centres about 1e-7 (up to 4x4). Untested near a transition.
- End-to-end numbers are sums of separately timed parts, not yet measured as one run.

## Timeline and owners

Workstreams follow CLAUDE.md: **GPU engine** (engine and benchmarks), **Models & analysis** (disordered model,
averaging, finite-size study, plots), **Physics validation** (parameters, invariant, paper comparison).
The validation track runs *in parallel* with the GPU track on Saturday morning, so the go/no-go doesn't block
the engine.

| When | GPU engine | Models & analysis | Physics validation | Done when |
|---|---|---|---|---|
| **Sat AM** | Clean Kane-Mele end-to-end on GPU (batched eigh + Newton-Schulz Wilson loop) | `anderson_kane_mele(L, W, delta, seed)`; H₀(k) + diagonal construction | Clean gap closes at ≈1.56t; CPU invariant on one disordered supercell; k-mesh convergence | Wannier centres match PythTB; measured speed-up. **Go/no-go:** invariant is correct in the clean limit and changes near W ≈ t at λν = 1.65t |
| **Sat PM** | Disorder diagonal and spin split on GPU; guard plus SVD fallback | Test harness vs `reference.py` | `spinv` cross-check; gap-check rule | One disordered system gives correct topological/trivial answers |
| **Sat eve** | Batching over realizations, chunking, c64 flag | Averaging plus error bars | c64 vs c128 classification study | Stable averages; CPU vs GPU speed-up measured |
| **Sun AM** | Baselines and benchmark table | Production sweep: W grid × λν ∈ {1.45, 1.65, 1.85} × ≥3 L, 400+ realizations per point, saved to disk as it runs | Compare with paper cases, Fig. 5, Born | Phase-diagram plot, Wc with error bars, time-to-result CPU vs GPU |
| **Sun PM** | Live demo plus recorded backup | Figures from saved data (script) | Assumptions and limitations slide | **Code freeze about 15:00**, pitch |

Sizes and realization counts come from the measured seconds per realization at each L (fill in on Saturday).
Pre-event estimates for one spin block, unmeasured: about 0.3 ms at L = 8, 20 ms at L = 16, 0.25 s at L = 24.

**Go/no-go fallback:** if the invariant is noisy or unconverged at small L, lead with the tool plus a clean-limit
and single-realization demo.

## Risks

| Risk | Mitigation |
|---|---|
| Speed-up is small, or only appears in c64 | Report both. Explain FP64 throughput on the event GPU. |
| Weak baseline undermines the headline | Quote vs every baseline, including multiprocessing. |
| Transition too blurred at small L | Use several L and finite-size analysis. Report the mismatch honestly. |
| Our invariant disagrees with the paper (different observable) | `spinv` cross-check; compare trend and approximate Wc. |
| Gap closes at E_F, or k-mesh too coarse | Count and report flagged realizations; mesh convergence test. |
| Newton-Schulz fails near Wc | Guard plus SVD fallback; report the rate. |
| Parameter mapping off | Clean-gap check before any disorder run. |
| Memory or time limits at large L | Spin split, chunking; quote the practical limit as a result. |

## Cut list (drop in this order)

λ_R > 0 → BerryEasy benchmark → local Chern marker / sign-function projector → collapse fit (keep the crossings)
→ Wc vs λν (keep only 1.65t) → extra sizes (keep 2).

**Minimum viable project:** the batched GPU pipeline validated against PythTB and benchmarked against the best
CPU, plus λν = 1.65t showing trivial → topological near W ≈ t.

## Stretch goals

- Collapse fit for Wc and ν with bootstrap errors (compare ν with Mildner et al.'s 2.42 for Haldane, arXiv:2312.16689).
- BerryEasy benchmark.
- Local Chern marker; eigensolver-free sign-function projector.
- λ_R > 0 with a true Z₂ invariant (Kramers-pair winding) or spin Bott index.

## Open questions (settle first)

- Event GPU model, memory and FP64 rate.
- Does PythTB's `kane_mele` convention match Orth's (λν, λ_SO)? Check through the gap closing.
- Exact winding → C_s algorithm on a supercell, and the k-mesh needed at each L.
- Time for one realization with PythTB as-is at each L on our hardware.

## References (local copies in `papers/`)

- Orth, Sekera, Bruder, Schmidt, arXiv:1512.03233: the target paper.
- Favata & Marrazzo, arXiv:2301.02612: single-point spin Chern number on supercells, disordered Kane-Mele.
- Favata, Baù & Marrazzo, arXiv:2503.23188; Baù & Marrazzo, arXiv:2404.04598: local markers, larger supercells.
- Assunção, Ferreira & Lewenkopf, arXiv:2401.03028; Mildner et al., arXiv:2312.16689: finite-size scaling recipes.
- Huang & Liu, arXiv:1810.00081: spin Bott index.
- `spinv`: https://github.com/roberta-favata/spinv
- Prior art: BerryEasy, TBPLaS 2.0, WannierBerri.
