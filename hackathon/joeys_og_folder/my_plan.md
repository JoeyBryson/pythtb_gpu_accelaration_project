> **Progress annotations, 21 Nov 2026, about 18:00** (up to commit `e7ff650` "GPU code").
> Status labels: **DONE**, **PARTLY**, **TODO**. Paths are relative to `hackathon/`.
> Detailed sources: `benchmarks/README.md` (benchmarks and competitors) and `VALIDATION.md` (physics checks).

PART 1 - Pythtb building

GPU speedup for PythTB WIlson Loop calculator(Newton-Shultz)
> **PARTLY.** Batched Newton-Schulz Wilson loops exist only in the pre-event Kaggle notebook
> (`joeys_og_folder/test_scripts/pythtb_gpu_tests.ipynb`). They give about 6.5x in double and 58x in single
> against the CPU at 8×8, and GPU SVD was *slower* than the CPU. They have **not yet been ported into
> `gpu_engine/`**. The GPU code written today (`gpu_engine/batched.py`) covers the other stage: building H(k)
> and diagonalising it, which is about 70% of the time. Still needed: a batched Wilson loop module in
> `gpu_engine/`, with the ‖XᴴX − I‖ guard, SVD fallback and correct loop closure at the zone boundary, tested
> against `reference.py`.

Basic benchmarks completed
> **DONE (CPU side).** `benchmarks/cpu_baseline.py` times the `clean`, `scaled`, `serial` and `parallel`
> scenarios. Results on Oscar's MacBook Air (10 cores) are in `results/cpu_*.csv`, with plots in
> `results/figures/`.
> - Solve is about 70% and Wilson about 30% of the time at every size.
> - One clean config with the scaled mesh (nk = 6): 0.75 s at 8×8, 4.3 s at 12×12, 22.5 s at 16×16.
>
> No GPU benchmark numbers exist yet from today's code (see the GPU line below).

Building the Kane-Mele Setup 
> **DONE.** `reference.clean_kane_mele(n, delta=..., soc=..., rashba=...)` builds an n×n supercell
> from PythTB's `kane_mele`. The parameter mapping is checked (`VALIDATION.md`): `delta` = λν and `soc` = λ_SO,
> and the clean gap matches 2·|3√3·λ_SO − λν| to 4 decimal places.
> Note that the *default* parameters are still the tutorial ones (λ_R = 0.25). The paper runs must pass
> `delta=1.45/1.65/1.85, soc=0.3, rashba=0`.

Calculating wilson loop with basic settup?
> **DONE (CPU reference).** `reference.wannier_centres` uses PythTB's `berry_phase`, and
> `topology.py` turns the Wannier flow into **Z₂ using largest-gap tracking** (the Soluyanov-Vanderbilt /
> Z2Pack method). This settles the invariant question, so it is Z₂ rather than the spin Chern number.
> Tests pass: tutorial δ = 1.0 gives Z₂ = 1, δ = 2.5 gives Z₂ = 0, and supercells give the same answer as the
> unit cell. There is a plot in `results/figures/wannier_flow.png`.
> **Lesson learned:** near the clean transition (λν = 1.50 to 1.55, where the gap is tiny) Z₂ is wrong unless
> the mesh is very fine (48×48 to more than 120×120). Two reliability checks now flag doubtful answers:
> `z2_margin` (tracking safety) and `min_overlap` (whether the mesh is fine enough, threshold 0.8).

Building the Kane-Mele Setup with Anderson Disorder
> **DONE.** On the CPU, `reference.disordered_kane_mele` draws a random on-site energy per orbital, the same
> for both spins. On the GPU side, `batched.py` builds the clean H(k) once and then adds a random
> diagonal for each config (H[c,k] = H_clean[k] + diag(d_c)). The random numbers are identical to the CPU for the
> same seed, and this is tested.
> **Disorder convention:** we use [−W/2, W/2], the same as Favata & Marrazzo and `spinv`. Orth et al. use
> [−W, W], so **W_ours = 2 × W_Orth**. Double every W quoted from Orth before comparing.

Can we again figure out the topological insulation properties in some way at this step(verifying understanding)?
> **PARTLY.** Done: weak disorder keeps Z₂ = 1 and very strong disorder (W = 20) gives Z₂ = 0, both
> tested at 2×2. Done: 40 configs at W = 2 all gave Z₂ = 1, but that used the *tutorial* parameters,
> which are deep in the topological phase, so it is not yet a test of the transition.
> **TODO, the key physics check** (`VALIDATION.md` checks 4 to 6), all with λ_SO = 0.3 and λ_R = 0:
> - λν = 1.45 stays topological as W increases.
> - **λν = 1.65 is trivial at W = 0 and becomes topological near W ≈ 2 (our convention).** This is the headline check.
> - λν = 1.85 gives a weak, unstable topological phase.
>
> Open decision: what to do when disorder fills the gap (`band_gap ≤ 0`). The difference between a spectral gap
> and a mobility gap is in `VALIDATION.md`, and the physics team needs to decide it.

Parallelizing to simulate many setups at once with CPU- Monte-Carlo simulation
> **DONE.** The `parallel` scenario runs one config per core, with 1 BLAS thread each. It uses the same seeds as
> `serial` and gives identical Z₂ answers. The gain is only about 4.2x on 10 cores, probably because the laptop
> mixes fast and slow cores (not yet checked).
> **TODO:** rerun on the CPU of the machine that hosts the GPU, because CPU and GPU numbers must come from the
> same machine.

Parallelizing to simulate many setups at once with GPU(second layer of GPU improvements) - Monte-Carlo simulation
> **PARTLY.** Done for building and diagonalisation: `batched.solve_configs` runs many configs × all k-points
> in one `torch.linalg.eigh` call, chunked to fit GPU memory, and matches PythTB config by config
> (eigenvalues and occupied-state projectors, `tests/test_batched.py`).
> So far it has **only been timed on the laptop CPU**: 1.1 to 1.3x faster than serial PythTB in double precision
> and 1.8 to 2.3x in single (`results/step1_batched_solve.csv`).
> **Next steps:**
> 1. Run `bash kaggle/gpu_run.sh`; `kaggle/run.py` is already set up to run `gpu_step1.py` on a T4.
> 2. Add the batched Wilson loop (see the first line), so that the whole pipeline stays on the GPU.
> 3. Add a batched Z₂ calculation, or copy the Wannier centres back to the CPU, since this step is cheap.

Benchmark the difference (between unoptimized WIlson Loop calculator and also unoptimized batching, just simple loop) - this should give a hefty speed up number
> **PARTLY.** Both baselines exist (`serial` is the naive loop, and `parallel` is the strongest CPU baseline), and
> `gpu_step1.py` already compares against serial PythTB on the same seeds. It only reports a speed-up when the
> answers match.
> Quote speed-ups against *both* baselines and say which one each refers to (rule in `benchmarks/README.md`).
> The competitor analysis is done (`benchmarks/README.md`):
> - PythTB's `berry_phase` loops in Python over lines, which is our Wilson-loop speed-up opportunity.
> - BerryEasy runs one k-point and one line at a time on the GPU.
> - `spinv` uses a single Γ-point diagonalisation per config, on the CPU, and needs PythTB 1.x.

	PART2 - Calculating the topological phase transition
> **TODO (all of Part 2).** Nothing is in `analysis/` yet. This depends on the GPU pipeline being complete
> end to end.

Simulate several system sizes over a range of disorder coefficients 
> **TODO.** The compute budget is now concrete from the CPU numbers. On the laptop, 16×16 takes about 22.5 s per config.
> A sweep of 400 configs × 10 W values × 3 λν values = 12,000 configs, which is about 75 h serial or 18 h on 10 cores.
> That shows why we need the GPU, and that the GPU must be fast at matrices of 1024 and above.
> Watch out for coarse meshes near the transition (see the Wilson-loop line). Measure the fraction of
> flagged-unreliable answers versus W, and check stability at 2× nk near Wc.

Use Finite size scaling (Binder Parameter?) to get an estimate for critical value?
> **TODO.** A Binder cumulant isn't needed. The fraction topological, or the mean Z₂, is already dimensionless, so
> curves for different L should cross near Wc (Mildner et al. 2312.16689; Assunção et al. 2401.03028).
> Core: estimate Wc from the crossings. Stretch: a scaling-collapse fit for Wc and ν.

Compare against values derived from paper todo: figure out what this exact benchmark is. Make it well defined. We are using a different method so there is a high chance of it being off.
> **DEFINED** in `VALIDATION.md`. Headline: λν = 1.65t, λ_SO = 0.3t, λ_R = 0. It is trivial at W = 0 and
> becomes topological near **W ≈ 2 in our units**. Orth (W ≈ t, doubled) and Favata ("about W/t = 2") agree once the
> factor of 2 is applied. Secondary: Wc against λν (Orth Fig. 5, figure only, with no table) and the cases
> λν = 1.45 and 1.85. Success = trend and approximate Wc, not exact numbers.
> The mismatch sources are written down: edge conductance (Orth) versus a bulk invariant (ours), finite L, and
> the spectral versus mobility gap question.

This creates an argument for GPU parallelization being useful
> **TODO.** Make it measurable: report the time to produce the full finite-size dataset, best CPU (`parallel`)
> against the GPU.

We should also mention/ compare against BerryEasy or spinV if we have loads of extra credits. But would be really good.
> **PARTLY.** Both have been analysed from their source code (`benchmarks/README.md`), so "how they differ
> from us" is already written up fairly.
> - BerryEasy is a speed comparison. It issues GPU work one k-point and one line at a time, but it is more
>   general than our code.
> - `spinv` is a correctness cross-check. It is cheaper per config (one Γ-point diagonalisation) but needs large
>   supercells, and it **needs its own PythTB 1.x environment**.
>
> Neither has been run yet.
