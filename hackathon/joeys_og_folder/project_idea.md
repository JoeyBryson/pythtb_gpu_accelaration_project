# Project: GPU Mode for PythTB, Simulating Thousands of Disordered Materials

## One-sentence pitch

We build a **GPU mode for PythTB** that runs the whole Wilson-loop topology pipeline (Hamiltonian, diagonalisation, Wilson loops) on the GPU and batches it over thousands of independent disorder configurations. We then use it to **find the disorder-driven topological transition in the Kane-Mele model** and check the result against Orth et al. (2016).

## Team decisions (20 Nov 2026)

1. **Two parts, so we don't depend on matching the paper perfectly.**
   - **Tool:** GPU mode for PythTB.
   - **Demo:** use the tool to simulate the disorder-driven phase transition and compare with the published result.
2. **Whole pipeline on the GPU.** PythTB already has a TensorFlow option for diagonalisation and `berry_flux`, but the main Wilson-loop function has no GPU path. We batch that too, using Newton-Schulz in place of SVD.
3. **Different method from the paper, on purpose.** The paper computes the Kwant conductance. PythTB diagnoses topology with Wilson loops, so we use a Wilson-loop invariant. We validate the transition, not the conductance.
4. **Focus on one section of the paper** (the disorder-driven transition), and reproduce intermediate results rather than every figure.
5. **Sell it with** a big, honest speed-up number, a simple explanation of why it matters, and validation against the paper.
6. **Analyse the PythTB-versus-paper differences carefully** (for example, the Wilson loop exists only on the PythTB side). See the comparison table below.

Per Oscar, the hackathon's own examples are mostly known science made faster. The novelty here is on the engineering side: batching disorder configurations and running Wilson loops on the GPU.

## Why this is a good hackathon project

| Judging criterion | How this project answers it |
|---|---|
| Correctness and assumptions | Ground truth from the clean phase boundary (Δ = 3√3·λ_SO), the Born approximation (Orth Eqs. 2-3), and an independent reference (`spinv`, built on PythTB). The method difference from the paper is stated openly. GPU results are checked against PythTB's CPU path, and FP32 against FP64. |
| Fundamental reshaping | One serial, one-model-at-a-time pipeline becomes batched dense linear algebra over realization x ky x W. Wilson loops use Newton-Schulz polar factors and a pairwise product tree, with no SVD or per-line loops. |
| Meaningful performance improvement | A headline speed-up against PythTB as-is, plus honest comparison against stronger CPU baselines (vectorised, multiprocessing). We explain where the time goes. |
| Scientific value | We locate the transition with error bars from many realizations, and compare with the paper's Wc and the Born approximation. Negative results are explained. |

## How PythTB and the paper differ (to analyse carefully)

| Aspect | Orth et al. (paper) | PythTB / our approach | Consequence |
|---|---|---|---|
| Observable | Two-terminal conductance G (Landauer, Kwant) | Wilson-loop (Wannier-centre) invariant | We validate the *transition*, not G or the 2G₀ plateau |
| Geometry | 93a x 150a open sample with metallic leads | Periodic L x L supercell, no leads | No edges and no transport. The bulk invariant stands in for the edge conductance |
| Size | About 28k sites | L of roughly 6-30 (dense linear algebra) | Finite-size broadening, so we study it with several L |
| Disorder | εᵢ ~ U[-1, 1] x W, 100 configurations per point | The same distribution. `spinv` uses [-w/2, w/2], so w = 2W | Keep conventions explicit in every comparison |
| Averaging | Conductance average and error bars | Fraction topological or mean invariant over realizations | A statistical quantity we define ourselves |
| Gap | Conductance window in E_F | Needs a gap at E_F = 0 in every realization | We check and count realizations where the gap closes |
| Theory curve | Born approximation (black lines) | Computed separately from the clean H(k) | A cheap comparison that needs no disorder |

## Scope and assumptions (stated up front)

- Disorder: static Anderson on-site disorder, as in Orth et al. Thermal fluctuations are not claimed.
- Observable: a topological invariant from Wilson loops (spin Chern number for λ_R = 0), **not** the conductance. A nonzero invariant predicts edge-state conductance, but it is a different observable. We do not reproduce Figs. 2-3.
- Regime: λ_R = 0 (S_z conserved) as the core case. λ_R > 0 is a stretch goal.
- Fermi level at half filling (E_F = 0), which must lie in a gap in each realization. We check this and report the fraction that fail.
- Agreement in trend and approximate location of Wc is the success criterion, not exact numbers.
- Finite-size broadening is part of the study, not something we hide.

## Part 1: the tool (GPU mode for PythTB)

### Why Wilson loops

A Wilson loop is the ordered product of overlap matrices between occupied states along a closed k-path. Its eigenvalue phases are the hybrid Wannier centres. Tracking how they wind as the second momentum goes around the zone gives the Chern number. With S_z conserved (λ_R = 0), the same calculation per spin block gives the spin Chern number, which is non-zero in the topological phase. This is how PythTB diagnoses topology, and its `berry_phase` function with `berry_evals=True` already does it on the CPU.

On a disordered supercell, k labels twisted boundary conditions, so the same machinery applies. The size of the (small) k-mesh is a convergence parameter that we must check.

### Pipeline, all on the GPU

1. **Build once, disorder on the diagonal.** Use PythTB once to build the clean Kane-Mele supercell (L x L cells, 2L² orbitals) and its H₀(k). Disorder only changes the diagonal, so each realization is H(k) = H₀(k) + diag(W·ε). This avoids constructing thousands of `TBModel` objects and the host-to-device copies that our profiling flagged as the next bottleneck. PythTB remains the model definition and the CPU reference.
2. **Spin split (λ_R = 0).** Each spin block is a smaller matrix, which cuts the O(N³) diagonalisation cost per realization by roughly 8x.
3. **Batched diagonalisation.** Torch `eigh` over (realization x k-points). Torch beat CuPy at large sizes in our T4 tests.
4. **Batched Wilson loops.** Overlaps between neighbouring k-points, a Newton-Schulz polar factor in place of SVD (matmuls only), then a pairwise product tree along each line. Batched over all lines and realizations.
5. **Invariant.** Wannier-centre winding to get the Chern number per spin block, plus a gap check at E_F = 0 so that every realization is classified or flagged.
6. **Batch over everything independent:** realizations (hundreds to thousands), W, Δ and L. For large L, size batches to the GPU memory.
7. **Precision as a knob.** FP64 and FP32. Report the error in the invariant and in the located Wc, especially near the transition.

### Independent cross-check

`spinv` (Favata, Baù & Marrazzo) computes a single-point spin Chern number for the same kind of PythTB-built, Anderson-disordered Kane-Mele supercell. We use it as an external reference on small cases, to confirm that our Wilson-loop invariant agrees with a different method.

### Stretch goals

- Eigensolver-free projector (sign-function iteration), with a convergence-versus-gap study.
- Local Chern marker maps.
- λ_R > 0 with a Z₂ invariant (Kramers-pair winding of the Wannier centres) or the Bott index.

## Part 2: the demo (disorder-driven transition)

Orth et al. (2016) show that Anderson disorder W can drive Kane-Mele from a trivial insulator (λν = 1.65t, λ_SO = 0.3t, λ_R = 0) to a topological one at W ≈ t. They use Kwant conductance on a 93a x 150a sample.

Our question: **where is the critical disorder Wc as a function of the staggered potential Δ (λν), as seen by a Wilson-loop invariant, and does it agree with the paper?** Because the tool is fast, we can answer it with many realizations and several system sizes, which gives error bars and shows how the transition sharpens with L.

Targets from the paper to compare against:

- λν = 1.45t: topological at W = 0, and stays topological.
- λν = 1.65t: trivial at W = 0, topological near W ≈ t.
- λν = 1.85t: the transition occurs, but the topological phase is fragile.
- Wc versus λν (Fig. 5) and the Born-approximation line (Eqs. 2-3).

The clean boundary is Δ = 3√3·λ_SO, about 1.56t for λ_SO = 0.3t. It lies between the 1.45t and 1.65t cases.

Follow-up literature (Favata & Marrazzo 2023; Baù & Marrazzo 2024; Favata, Baù & Marrazzo 2025) studies the same transition with supercell invariants.

### Finite-size scaling (the core analysis)

A supercell has finite L, so the transition is smeared. Instead of fighting this, we use it. Near Wc the disorder-averaged invariant depends on W and L only through one variable, c̄(W, L) = f̃((W - Wc)·L^(1/ν)). This is the recipe of Mildner et al. (arXiv:2312.16689, Haldane, L up to 17, 500 realizations, Wc = 3.58 ± 0.02, ν = 2.42 ± 0.11) and Assunção et al. (arXiv:2401.03028, BHZ, local Chern marker via KPM, 400 realizations, scaling variable (W^μ - Wc^μ)·L).

Steps:

1. **Data.** Disorder-averaged invariant (or fraction topological) versus W for at least 4 sizes (for example L = 8, 12, 16, 24), at fixed Δ, with 400-1000 realizations per point.
2. **Crossing.** Estimate Wc from where curves for different L cross, and report how the crossing drifts with L.
3. **Collapse.** Fit Wc and ν by collapsing all curves onto one function, with bootstrap error bars over realizations.
4. **Second observable.** Local Chern marker at the sample centre, to compare with the Wilson-loop invariant (they may scale differently).
5. **Second transition.** The TAI to Anderson-insulator transition at larger W should be L-independent in Assunção et al., which is a useful consistency check.

No finite-size-scaling collapse for Kane-Mele or the TAI phase was found in our literature search, so a clean collapse is a new result for us, and a failed collapse is still reportable (we explain why). Compare ν with Mildner's 2.42 for Haldane and with the clean mass-driven value of about 1.

Why GPU matters here: the collapse needs many sizes x many W x hundreds of realizations, which is the regime where CPU runs become impractical. Benchmark metric: **time to a finite-size-scaling result**, best CPU against best GPU.

## Evidence we already have

From `GPU_experimentation/test_scripts/pythtb_gpu_tests.ipynb` (Kaggle T4, clean Kane-Mele supercell, 31x31 mesh):

- About 70% of PythTB's time is diagonalisation and about 30% Wilson loops.
- Torch `eigh` beats CuPy at large sizes. At 8x8, it is about 1.9x faster than numpy in FP64 and about 10x in FP32.
- Batched Wilson loops with Newton-Schulz polar decomposition replace SVD: about 6.5x in FP64 and about 58x in FP32, with identical results.
- The FP32 error in wannier centres was about 1e-7 on small cases, but is untested near a transition.

These show that the batched-linear-algebra approach pays off. They are not yet disorder-averaged science results, and the end-to-end figures in the notebook are sums of separately timed parts, not a single run.

## What the literature tells us about scale

From the papers in `papers/related_work/`:

- Favata & Marrazzo (2301.02612) use single-point (one diagonalisation) spin Chern numbers on supercells with N = 2L² sites, with L up to 51. They validate on Kane-Mele with Anderson disorder, and their code is public.
- Favata, Baù & Marrazzo (2503.23188) use supercells up to L = 78 for Haldane (12168 sites), with 200-300 realizations at some points, and L = 72 for Kane-Mele.

So the interesting regime is L of about 40-100 with hundreds of realizations over a grid. Dense diagonalisation scales as O(N³), so this is exactly the regime where a CPU baseline becomes expensive and a batched GPU approach could change what is practical. I have not verified how those authors ran their calculations or what their wall-clock times were, so the baseline comparison must be measured by us.

## Deliverables

1. **Validated model.** Kane-Mele with Anderson disorder from PythTB. Verify the parameter mapping to the papers and that the clean gap closes at Δ = 3√3·λ_SO.
2. **GPU mode (the tool).** A batched pipeline over disorder configurations (build, eigh, Wilson loops, invariant) with FP64 and FP32, validated against PythTB's CPU `berry_phase` on single realizations.
3. **Demo result.** Invariant versus W for several Δ and L, Wc(Δ) with error bars, and comparison with the paper's cases, Fig. 5 and the Born approximation. A finite-size-scaling collapse if the data support it.
4. **Benchmark report.** Time per realization versus L for each implementation and precision, "realizations per GPU-hour", where the time goes, and the memory limit on batch size.
5. **Cross-check.** Agreement with `spinv` on small cases.
6. **Pitch material.** One headline speed-up number, one simple explanation of why it matters, and one validation plot.

## Benchmark design (credible baselines)

| Baseline | Purpose |
|---|---|
| PythTB as-is (numpy, per-realization `TBModel`, per-line Wilson loops) | The headline "before" number |
| PythTB with `use_tensorflow=True` where available | Shows what the existing GPU option gives, and what we add on top |
| Vectorised numpy, batched over lines | The fair serial CPU baseline |
| Same, multiprocessing across realizations on all cores | The fair parallel CPU baseline (judges will look for it) |
| `spinv` on one small case | External correctness reference, not a speed baseline |

Report: speed-up against each baseline, where the time goes (diagonalisation versus matmuls versus transfers), and the memory-bound versus compute-bound regimes. Explain, rather than merely state, any speed-up.

## Validation checks

- Clean limit: the invariant is ±1 or 0 on the correct side of Δ = 3√3·λ_SO.
- GPU versus CPU: eigenvalues, Wannier centres and invariants against PythTB's `berry_phase`, for single realizations.
- Newton-Schulz versus SVD polar factors: agreement, and convergence checks near the transition, where overlap matrices can be ill-conditioned.
- Disorder: Born-approximation gap edges versus the numerical gap at small W.
- Precision: FP32 versus FP64 invariants and Wc, specifically near the transition.
- Statistics: standard error of the mean over realizations, and a check that no realization has an E_F-crossing state that makes the invariant ill-defined.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| The speed-up is smaller than hoped, or only appears in FP32 | Report FP64 and FP32 separately, and explain the gap (the T4 has weak FP64). Confirm the hackathon GPU early. |
| The headline number is not credible (weak baseline) | Include the strong CPU baselines. Quote speed-up against each, not only the weakest. |
| Transition is too blurred at small L, so the demo does not match the paper | The tool is still the deliverable. Use several L and finite-size scaling, and report any mismatch honestly. |
| Our Wilson-loop invariant does not match the paper's transition (different observable) | Cross-check with `spinv`. State that the observables differ, and compare at the level of trend and approximate Wc. |
| Wilson loop breaks down for some realizations (gap closes at E_F, or k-mesh too coarse) | Count and report those realizations, and test mesh convergence. |
| Newton-Schulz fails to converge on ill-conditioned overlaps | Compare with SVD on sampled cases, use more iterations or fall back to SVD where needed, and report the failure rate. |
| Parameter mapping between PythTB and the papers is off | Verify the clean boundary first, before any disorder run. |
| Dense eigh at large L does not fit memory or time | Spin-split first. Use moderate L for the full grid, and quote the practical limit as a result. |
| Building supercells in PythTB is slow | Build once and add the diagonal on the GPU. |
| Scope creep (conductance, Kwant, λ_R > 0) | Kept out of scope. See the cut list. |

## Step-by-step plan (adapt durations to the hackathon length)

Each step has a **done when** check. Do not move on until it passes. Steps 1-2 are the go/no-go.

### Step 0. Setup (all three)

- Confirm the hackathon GPU, its memory, and FP64 rate. Set up a shared repo and a common environment (torch, numpy, PythTB from this fork).
- Agree roles: **A** model and validation, **B** GPU pipeline, **C** benchmarks, analysis and pitch.
- Done when: everyone can run a `torch.linalg.eigh` on the GPU and import PythTB.

### Step 1. Model and parameter check (A)

- Write `anderson_kane_mele(L, W, delta, soc=0.3, rashba=0, seed)` using `kane_mele`, `make_supercell`, and `set_onsite(..., mode="add")` with ε ~ U[-1, 1] x W.
- Check the clean gap closes at δ = 3√3·λ_SO ≈ 1.56t (unit cell, W = 0). Check the hopping and SOC conventions against the paper.
- Done when: the gap closes at about 1.56t and the model builds for L = 4-8.

### Step 2. CPU reference and go/no-go (A)

- Run PythTB's `berry_phase` (Wilson loop) on a disordered supercell, using a small twist mesh per spin block, and turn the Wannier-centre winding into a spin Chern number.
- Test at δ = 1.45t, 1.65t, W = 0 and W ≈ t. Check mesh convergence, gap at E_F = 0, and noise at L = 8-16.
- Time one realization at each L (this fills in the compute-budget table).
- Done when: the invariant is correct in the clean limit and shows a trivial-to-topological change near W ≈ t at δ = 1.65t, or we have decided on the fallback (lead with the tool).

### Step 3. External reference (A)

- Clone `spinv`, run its Kane-Mele Anderson example, and check it works with this PythTB fork. Remember w = 2W.
- Done when: `spinv` and our CPU invariant agree on a few small cases.

### Step 4. GPU pipeline v1 (B, with A)

- Build H₀(k) once and add the disorder diagonal on the GPU. Spin split. Batched `eigh` over (realization x k).
- Batched overlaps, Newton-Schulz polar factor, pairwise product tree, winding, and invariant. Add the ‖X†X - I‖ guard with SVD fallback.
- Done when: eigenvalues, Wannier centres and invariants match the CPU reference on 20+ single realizations, in FP64.

### Step 5. Precision and robustness (B)

- Add FP32. Measure the fraction of realizations whose classification differs from FP64, by W, especially near the transition.
- Count realizations with a closed gap or failed Newton-Schulz.
- Done when: we know where FP32 is safe, and the failure rates are reported.

### Step 6. Benchmarks (C)

- Implement the baselines: PythTB as-is, `use_tensorflow`, vectorised numpy, multiprocessing.
- Measure time per realization versus L, realizations per hour, the time breakdown, and the maximum batch size. Use equal-accuracy settings.
- Done when: there is one table with the speed-up against each baseline, and a plain explanation of the biggest speed-up.

### Step 7. Production run (B and C)

- Pick L (for example 8, 12, 16, 24) and realization counts from the budget table. Grid over W for δ = 1.45t, 1.65t, 1.85t, with 400-1000 realizations per point. Save raw results to disk as the run goes.
- Run the same grid on the best CPU baseline for the "time to result" metric (or extrapolate from a subset, labelled as such).
- Done when: raw invariant data exist for every grid point.

### Step 8. Analysis (A and C)

- Plot fraction topological versus W per L. Estimate Wc from the crossing, then fit the collapse (Wc, ν) with bootstrap errors.
- Compare Wc at δ = 1.65t with the paper's W ≈ t, and Wc(δ) against Fig. 5 and the Born approximation (read the figure from the PDF).
- Done when: one figure each for invariant versus W, the collapse, and Wc(δ) against the paper. Mismatches are explained.

### Step 9. Extras, only if time remains (follows the cut list)

- More `spinv` cross-checks, local Chern marker at the centre, sign-function projector backend, λ_R > 0.

### Step 10. Pitch and write-up (C, with all)

- Headline speed-up with the stronger baselines beside it, the simple explanation, and the validation plot.
- A short list of assumptions and limitations (different observable from the paper, finite L, few twists, FP32 caveats).
- Reproducible script that regenerates the figures from the saved data.
- Done when: a 3-minute pitch can be given from the figures alone.

### Time guide

Roughly: steps 0-3 take the first 20-25%, steps 4-5 the next 25%, step 6 about 10-15%, steps 7-8 about 25%, and step 10 the last 10-15%. Start the pitch outline during step 6, not at the end.

## Cut list if time runs short

Drop in this order: λ_R > 0, the eigensolver-free projector, local-marker maps, the collapse fit (keeping the crossing-point estimate of Wc over several L), then Wc versus Δ. The minimum viable project is the GPU mode (batched eigh and Wilson loops), validated against PythTB and benchmarked, plus the demo at λν = 1.65t showing the transition against the paper's trivial-to-topological claim.

## How we sell it (the "so what")

- **Simple explanation:** real materials are imperfect, and disorder can switch a material's topological state. Finding the switching point means simulating thousands of random imperfect samples. We make that fast enough to be routine.
- **Headline number:** an honest end-to-end speed-up (and "realizations per hour") against PythTB as-is, with the stronger baselines shown beside it.
- **Credibility:** the demo agrees, in trend and approximate Wc, with a published result, computed with a different method.
- **What it enables:** tighter error bars on Wc, larger systems for a sharper transition, a dense (Δ, W) phase-diagram scan, and a reusable batched toolkit for disordered topological models built on PythTB.

## Robustness measures

- **Day-1 go/no-go.** One disordered realization on the CPU, checking that the clean boundary is at about 1.56t, the invariant is stable under k-mesh refinement, and a few-twist invariant is not noisy at L = 8-16. Agree pass criteria in advance. Fallback: lead with the tool and a clean-limit plus single-point demo.
- **Three invariant routes behind one interface:** determinant (FHS) on a twist mesh, single-point Γ (as in `spinv`), and the Newton-Schulz Wilson loop. Cross-check them against each other.
- **Baseline fairness.** The PythTB default uses a dense mesh, while we may use few twists. Benchmark equal-accuracy settings, not equal-mesh ones, and apply Amdahl's law to the non-GPU part.
- **Compute-budget table.** Estimated and then measured seconds per realization by L (my unrun estimates for one spin block: about 0.3 ms at L = 8, 20 ms at L = 16, 0.25 s at L = 24, 15 s at L = 48). Choose L range and realization counts from this.
- **Precision protocol.** Fraction of realizations whose classification differs between FP32 and FP64, by W, with extra attention near Wc. The FP64 gain may vanish against a multicore CPU.
- **Newton-Schulz guard.** Monitor ‖X†X - I‖, fall back to SVD on failure, and report the failure rate.
- **Second backend (optional, clearly labelled).** Sign-function or Chebyshev projector, matmul-only, compared against `eigh`. PythTB has no sparse code.
- **Roles.** Split work across: model and validation, GPU pipeline, benchmarks and analysis, pitch.

## Open questions to settle first

- What GPU do we actually get, and how much memory?
- Does PythTB's `kane_mele(delta, t, soc, rashba)` convention match the papers' Δ, t, λ_SO, λ_R conventions? Verify numerically through the clean gap closing.
- How exactly do we turn Wannier-centre winding into an invariant per spin block and per realization, and what k-mesh size is enough on a supercell? Check PythTB's `berry_phase` usage and test convergence.
- How long does a single realization take with PythTB as-is at each L on our machine? This sets the baseline numbers.

## Key references (local copies in `papers/`)

- Orth, Sekera, Bruder, Schmidt, *The topological Anderson insulator phase in the Kane-Mele model*, arXiv:1512.03233 (the original paper).
- Favata & Marrazzo, *Single-point spin Chern number in a supercell framework*, arXiv:2301.02612.
- Baù & Marrazzo, *Theory of local Z₂ topological markers for finite and periodic two-dimensional systems*, arXiv:2404.04598.
- Favata, Baù & Marrazzo, *Fluctuations and correlations of local topological order parameters in disordered two-dimensional topological insulators*, arXiv:2503.23188.
- Huang & Liu, *Theory of spin Bott index for quantum spin Hall states in non-periodic systems*, arXiv:1810.00081.
- Assunção, Ferreira & Lewenkopf, *Phase transitions and scale invariance in topological Anderson insulators*, arXiv:2401.03028.
- Mildner et al., disordered Haldane model, finite-size scaling and transfer matrix, arXiv:2312.16689.
- `spinv` (MIT licence, PythTB-based): <https://github.com/roberta-favata/spinv>
