# Results log

Every number we might quote, with where it came from. Rule: **never compare numbers from
different machines** without saying so. Raw data is in the file named under each result.
Protocol (what each benchmark measures and why): [`benchmarks/README.md`](benchmarks/README.md).
Physics checks against the papers: [`VALIDATION.md`](VALIDATION.md).

## Machines

| ID | Machine | CPU | GPU | Software |
|---|---|---|---|---|
| **MAC** | Oscar's MacBook Air | Apple Silicon (arm), 10 cores, Apple Accelerate BLAS | none (CPU only) | Python 3.12, numpy 2.5.5, torch 2.15 (CPU) |
| **KAG** | Kaggle notebook, "GPU T4 x2" | x86_64, 4 vCPUs | 2 × Tesla T4, 15.6 GB each (**only 1 used so far**) | Python 3.13, numpy 2.1.3, torch 2.11+cu130 |

T4 note: double precision is ~15× slower than single on this GPU (measured below, R5).

## Headline so far

> **Full pipeline (diagonalise → Wilson loop → Z2) of a 16×16 supercell, Rashba = 0, same
> Kaggle machine (R10): PythTB on all 4 CPU cores 575 s, ours on one T4 in single precision
> 31 s: 18.3× faster** (16 configs); **15.9× at 12×12** (40 configs). Z2 identical to PythTB for
> every configuration. Wannier centres accurate to ~1e-8 vs exact (eigenvalue step in double).
> Speed-up grows with size: 4.4× at 8×8, 15.9× at 12×12, 18.3× at 16×16.
> Baseline timed fairly (no diagnostics, Filip's fix). Superseded: R7 (8.9×, inflated baseline),
> R9 (14.3× at 12×12, Rashba 0.25, eigenvalues ~1e-3 off at 16×16).
> **Time to a full disorder sweep (R12, estimated from measured per-sample rates, same Kaggle
> machine, 16,400 samples):** at 16×16, PythTB on all 4 cores **~172 h (7 days)** vs our C_up
> algorithm on one T4 **~2.1 h: 83× faster** (8.8× from the algorithm, 9.4× more from the GPU);
> at 12×12, ~32 h vs ~55 min (35×).
> Not yet included: the second T4. On a T4, double precision overall gives only 1.6–2.9× (R8).

---

## R1. CPU baseline, clean model, fixed 31×31 mesh (MAC)

What: one clean Kane-Mele supercell, PythTB 2.0 `solve_model` + `berry_phase`, best of 3.
Why: same settings as the prep runs in `CLAUDE.md`.
Command: `uv run python benchmarks/cpu_baseline.py clean --sizes 2 4 6 8` · 21 Nov 16:41 · file `results/cpu_clean.csv`

| Supercell | Matrix | Solve | Wilson loop | Total |
|---|---|---|---|---|
| 2×2 | 16 | 0.021 s | 0.013 s | 0.035 s |
| 4×4 | 64 | 0.56 s | 0.24 s | 0.80 s |
| 6×6 | 144 | 3.54 s | 1.26 s | 4.81 s |
| 8×8 | 256 | 13.9 s | 4.6 s | 18.6 s |

Solve is 62–75% of the time, Wilson loop 25–38%.

## R2. CPU baseline, clean model, scaled mesh (MAC)

What: as R1 but mesh ≈ 48/n per direction (min 6), the meshes we'd actually use. Includes Z2.
Command: `uv run python benchmarks/cpu_baseline.py scaled --sizes 4 8 12 16` · 21 Nov 16:49 · file `results/cpu_scaled.csv`

| Supercell | Matrix | Mesh | Solve | Wilson loop | Total | Z2 (expected 1) |
|---|---|---|---|---|---|---|
| 4×4 | 64 | 12×12 | 0.081 s | 0.038 s | 0.12 s | 1 ✓ |
| 8×8 | 256 | 6×6 | 0.50 s | 0.23 s | 0.75 s | 1 ✓ |
| 12×12 | 576 | 6×6 | 3.02 s | 1.25 s | 4.30 s | 1 ✓ |
| 16×16 | 1024 | 6×6 | 16.6 s | 5.8 s | 22.5 s | 1 ✓ |

## R3. CPU baseline, 40 disorder configurations, serial vs all cores (MAC)

What: full pipeline (build, solve, Wilson loop, Z2) for 40 configurations at W = 2,
tutorial parameters. Serial = Python loop; parallel = 10 processes, 1 BLAS thread each.
Command: `uv run python benchmarks/cpu_baseline.py both --size {6,8} --configs 40 --W 2.0` · 21 Nov 16:49 · files `results/cpu_serial.csv`, `results/cpu_parallel.csv`

| Supercell | Mesh | Serial | Parallel (10 procs) | Parallel gain | Fraction topological |
|---|---|---|---|---|---|
| 6×6 | 8×8 | 13.4 s (0.34 s/config) | 3.2 s (0.080 s/config) | 4.2× | 1.00 (0 unreliable) |
| 8×8 | 6×6 | 30.1 s (0.75 s/config) | 7.1 s (0.18 s/config) | 4.2× | 1.00 (0 unreliable) |

Serial and parallel gave identical Z2 for every configuration.

⚠ Wall times here include the reliability diagnostics (`run_config(check=True)`): 15% of wall time measured on KAG, ~20% on MBP.
Fixed 21 Nov ~19:30, see "Timing fix" under R7. The per-stage sums are unaffected.

## R4. Step 1 (batched diagonalisation): ours vs PythTB, 20 disorder configs

What: H(k) for every configuration and k-point, then diagonalise. PythTB = serial loop
(build model + `solve_model`). Ours = build H_clean once, add each config's random diagonal,
one batched `eigh`, timed end to end incl. copies to/from the GPU. Same seeds, W = 2,
tutorial parameters. Every result checked against PythTB's eigenvalues.
Command: `python benchmarks/gpu_step1.py --sizes 4 8 12 --configs 20`

### R4a. On KAG (Kaggle, T4), 21 Nov 17:57

File `results/kaggle/20261121-174859/out/step1_batched_solve.csv` (log alongside).

| Supercell (mesh) | PythTB serial | Ours CPU double | Ours CPU single | Ours GPU double | **Ours GPU single** |
|---|---|---|---|---|---|
| 4×4 (12×12) | 5.36 s | 1.77 s (3.0×) | 1.35 s (4.0×) | 11.84 s (0.45×) | 2.16 s (2.5×) |
| 8×8 (6×6) | 24.57 s | 7.59 s (3.2×) | 4.99 s (4.9×) | 10.01 s (2.5×) | **1.97 s (12.5×)** |
| 12×12 (6×6) | 146.41 s | 48.74 s (3.0×) | 29.30 s (5.0×) | 35.19 s (4.2×) | **6.84 s (21.4×)** |

Max eigenvalue error vs PythTB: ≤ 3e-14 (double), ≤ 2e-5 (single). All 24 tests passed on KAG.

### R4b. On MAC (CPU only), 21 Nov 17:46

File `results/step1_batched_solve.csv`.

| Supercell (mesh) | PythTB serial | Ours CPU double | Ours CPU single |
|---|---|---|---|
| 4×4 (12×12) | 1.81 s | 1.41 s (1.3×) | 0.78 s (2.3×) |
| 8×8 (6×6) | 10.31 s | 9.48 s (1.1×) | 5.69 s (1.8×) |

### What R4 tells us

- GPU gain grows with matrix size (single precision: 2.5× → 12.5× → 21×).
- Small matrices are bad for GPU `eigh`: at 4×4 the GPU in double is *slower* than the CPU.
- The T4's double precision is weak: big gains need single precision, so single-precision
  correctness must be shown on the final Z2, not just eigenvalues.
- Ours-on-CPU beats PythTB by 3× on KAG but only 1.1–1.8× on MAC. **Probably a maths-library
  effect** (PyTorch vs NumPy on Intel), not our algorithm (to check). Don't claim it as ours.

## R5. Raw GPU check (KAG, standalone script, no project code)

What: `torch.linalg.eigh` on random Hermitian batches vs NumPy on the same Kaggle machine.
Kaggle kernel `pythtb-gpu-check`, 21 Nov 17:38. Log in scratch only (numbers copied here).

| Batch | NumPy CPU (double) | T4 double | T4 single |
|---|---|---|---|
| 961 × 64×64 | 1.54 s | 4.08 s | 0.71 s |
| 961 × 144×144 | 7.76 s | 7.30 s | 1.38 s |
| 961 × 256×256 | 26.81 s | 13.25 s | 2.49 s |
| 64 × 576×576 | 12.08 s | 3.04 s | 0.56 s |
| 36 × 1024×1024 | 31.87 s | 5.69 s | 1.05 s |

T4 matmul throughput: 3.50 TFLOP/s single, 0.24 TFLOP/s double.

## R6. Full pipeline (solve + Wilson loop + Z2), 40 disorder configs (MAC, CPU only)

What: every configuration taken all the way to Z2. PythTB serial and all-cores (10 procs)
vs our batched pipeline (`gpu_engine.pipeline.run_batch`: batched solve, Newton-Schulz
Wilson loop with tree product). Same seeds, W = 2, tutorial parameters. Reliability checks
excluded from all timings.
Command: `uv run python benchmarks/gpu_pipeline.py --sizes 6 8 --configs 40` · 21 Nov ~18:25 · file `results/full_pipeline.csv`

| Supercell (mesh) | PythTB serial | PythTB 10 cores | Ours CPU double | Ours CPU single | Z2 agrees |
|---|---|---|---|---|---|
| 6×6 (8×8) | 13.5 s | **3.1 s** | 13.2 s (solve 8.3, Wilson 4.9) | 7.1 s (solve 4.9, Wilson 2.2) | 40/40 both |
| 8×8 (6×6) | 29.8 s | **8.4 s** | 28.0 s (solve 18.5, Wilson 9.5) | 16.0 s (solve 11.5, Wilson 4.4) | 40/40 both |

No Newton-Schulz fallbacks, no configurations flagged unreliable.

**Honest negative result:** on the CPU, our batched code is *not* faster than PythTB on all
cores (0.2–0.5×), and in double precision our Newton-Schulz Wilson loop is *slower* than
PythTB's SVD (4.9 s vs 3.9 s at 6×6). Expected: on a CPU the SVD is fast and Newton-Schulz
does extra matrix multiplications (12 fixed steps; could stop early). Our method is designed
for the GPU; on the CPU, PythTB on all cores remains the better choice. Worth saying in the pitch.

## R7. Full pipeline, CPU vs GPU, same machine, 40 disorder configs (KAG)

What: as R6, on Kaggle, with the GPU. PythTB serial and all-cores (4 procs) vs ours on CPU
and T4, double and single. Same seeds, W = 2, tutorial parameters. Code as of 21 Nov ~18:05,
i.e. **before** the Wilson-loop speed-ups (Newton-Schulz always 12 steps, GPU eigenvalues).
Command: `python benchmarks/gpu_pipeline.py --sizes 8 12 --configs 40 --W 2.0` · Kaggle run
21 Nov 18:21–19:10 (version 4) · files `results/kaggle/20261121-180746/out/*.csv` + log

| Supercell (mesh) | PythTB 1 core | PythTB 4 cores | Ours CPU double | Ours CPU single | Ours GPU double | **Ours GPU single** |
|---|---|---|---|---|---|---|
| 8×8 (6×6) | 84.1 s | 37.6 s | 36.0 s (1.0×) | 20.8 s (1.8×) | 37.5 s (1.0×) | **9.8 s (3.8×)** |
| 12×12 (6×6) | 465.8 s | 321.3 s | 251.7 s (1.3×) | 135.9 s (2.4×) | 155.4 s (2.1×) | **36.1 s (8.9×)** |

Speed-ups in brackets are vs **PythTB on all 4 cores** (the credible baseline). PythTB 1-core
time is the sum of per-config pipeline times (excl. reliability checks). Z2 agreed with PythTB
for **40/40 configurations in every column**; no Newton-Schulz fallbacks; none unreliable.

GPU single, stage split: 8×8 solve 4.0 s / Wilson 5.8 s; 12×12 solve 15.8 s / Wilson 20.3 s.
The Wilson loop taking longer than the diagonalisation is the bottleneck the later fix targets.

**Timing fix (21 Nov ~19:30).** In this run the **PythTB 4-core times are too slow**: `bench_parallel`
timed `run_config` with its default `check=True`, so its wall clock included the reliability
diagnostics (an SVD over every link). The 1-core column was already right: it is the sum of
per-config pipeline times, which exclude the diagnostics. Size of the effect, from this run's own
`cpu_serial.csv` (wall clock vs sum of stage times, same configs): 99.2 vs 84.1 s at 8×8 and
545.9 vs 465.8 s at 12×12, i.e. **15% of wall time**. Assuming the same share in the parallel run
(not measured), the 4-core baseline is ~31.9 s (8×8) and ~274 s (12×12), and GPU single becomes
**~3.3× and ~7.6×** (not 3.8× and 8.9×). `cpu_baseline.py` now times with `check=False` and takes
the flags from one separate untimed pass (`reliability_pass`); `gpu_pipeline.py` uses it. On MBP
the fix cut serial wall time 9.29 → 7.81 s and parallel 1.48 → 1.32 s (6×6, 20 configs), with
identical Z2 and flags. **Re-run this table's PythTB columns before quoting it.**

Notes: Kaggle's 4 cores only give 1.4–2.2× over 1 core (memory bandwidth? not checked).
Kaggle's CPU is ~2–4× slower than MAC for PythTB (e.g. 8×8 serial 84 s vs 30 s): never mix the
two machines' numbers in one comparison.

## R8. Full pipeline with the faster Newton-Schulz, GPU vs PythTB all cores (KAG)

What: as R7 (40 configs, W = 2, tutorial parameters, same seeds), GPU only vs PythTB on all
4 cores. Code as of 21 Nov ~19:37: **with** early-stopping Newton-Schulz and the reliability
check only in single precision; **without** Filip's baseline timing fix (so PythTB's 4-core time
still includes its untimed diagnostics, ~15% high) and **without** the Hermitian eigenvalue step
(eigenvalues still computed on Kaggle's CPU). Run cancelled at the start of 16×16 (outdated by
then), so only the log was saved; numbers below are copied from it.
Command: `python benchmarks/gpu_pipeline.py --sizes 8 12 --configs 40 --W 2.0 --devices cuda --skip-serial`
· Kaggle run 21 Nov 19:37–19:52 (version 7) · log `results/kaggle/20261121-193743/pythtb-gpu-run.log`

| Supercell (mesh) | PythTB 4 cores (as measured) | Ours GPU double | **Ours GPU single** | Z2 agrees |
|---|---|---|---|---|
| 8×8 (6×6) | 40.6 s | 25.3 s (1.6×) | **6.6 s (6.2×; ~5.2× corrected)** | 40/40 |
| 12×12 (6×6) | 337.9 s | 117.8 s (2.9×) | **32.7 s (10.3×; ~8.8× corrected)** | 40/40 |

"Corrected" = PythTB time × 0.85 (Filip's measured ~15% diagnostic share; not re-measured here).
No Newton-Schulz fallbacks.

Where the GPU time goes (single precision):

| | Solve | Wilson total | of which: overlaps | Newton-Schulz | product | **eigenvalues (on CPU)** |
|---|---|---|---|---|---|---|
| 8×8 | 3.97 s | 2.61 s | 0.015 | 0.046 | 0.005 | **2.54** |
| 12×12 | 15.86 s | 16.83 s | 0.125 | 0.735 | 0.079 | **15.89** |

**Finding:** early stopping made Newton-Schulz almost free (0.05–0.7 s), so the Wilson loop is
now almost entirely the eigenvalue step on Kaggle's slow CPU: **~half of the total GPU time at
12×12**. Fixed afterwards (commit 66f7a14): eigenvalues via a Hermitian `eigh` on the GPU, 3–4×
faster even on MAC's CPU; to be measured on KAG (R9).

Also seen: PythTB on 4 cores at 16×16 took 184 s for its first 4 configs (~46 s per config per
worker), so 16 configs ≈ 12 min on KAG.

## R9. Final speed run: fair baseline + all Wilson-loop speed-ups (KAG)

What: full pipeline, GPU single precision vs PythTB on all 4 cores, same machine, same seeds,
W = 2, tutorial parameters. **PythTB timed without its diagnostics** (Filip's fix, fb0cd23) and
our Wilson loop with early-stopping Newton-Schulz and Hermitian eigenvalues (66f7a14).
Command: `kaggle/run.py` JOBS at 66f7a14 (`gpu_pipeline.py --devices cuda --precisions single
--skip-serial --no-reliability`, sizes 8 and 12 with 40 configs, 16 with 16) · Kaggle run
21 Nov 19:57–20:17 (version 9) · files `results/kaggle/20261121-195518/out/*.csv` + log

| Supercell (mesh) | Configs | PythTB 4 cores | **Ours T4 single** | **Speed-up** | Z2 agrees |
|---|---|---|---|---|---|
| 8×8 (6×6) | 40 | 34.3 s | 4.6 s | **7.4×** | 40/40 |
| 12×12 (6×6) | 40 | 280.2 s | 19.5 s | **14.3×** | 40/40 |
| 16×16 (6×6) | 16 | 607.2 s | 51.6 s | **11.8×** | 16/16 |

No Newton-Schulz fallbacks, no unreliable answers. All 38 tests passed on KAG first.

Where the GPU time goes:

| | Solve | Wilson total | overlaps | Newton-Schulz | product | eigenvalues |
|---|---|---|---|---|---|---|
| 8×8 | 3.97 s | 0.63 s | 0.015 | 0.044 | 0.004 | 0.57 |
| 12×12 | 15.73 s | 3.79 s | 0.127 | 0.606 | 0.062 | 2.99 |
| 16×16 | 36.26 s | 15.36 s | 0.237 | 1.185 | 0.122 | **13.81** |

Findings:
- Fair baseline confirmed: PythTB's 4-core time fell 338 → 280 s at 12×12 (−17%) vs R8, in line
  with Filip's ~15% estimate.
- The Hermitian eigenvalue step cut that sub-step 15.9 → 3.0 s at 12×12 (R8 → R9).
- **Why 16×16 is lower than 12×12:** with 512 eigenvalues per Wilson matrix in single
  precision, ~1 in 6 matrices fails the eigenvalue safety check (checked on MAC: 2 of 12 at
  16×16, residuals 0.11 vs tolerance 0.05) and falls back to the general solver on Kaggle's CPU,
  which is slow at 512×512. Answers stay correct; time doesn't. Fix planned: retry failures on
  the GPU in double precision with a different mixing value before any CPU fallback.
- Diagonalisation is now 70–85% of GPU time: the spin-block split (λ_R = 0) targets it.

## R10. Rashba = 0, spin-block split, accurate eigenvalues (KAG)

What: full pipeline, GPU single precision vs PythTB on all 4 cores, same machine, same seeds,
W = 2, **Rashba = 0 on both sides** (team choice; other parameters tutorial), PythTB timed
without diagnostics. Ours with the spin-block split (two half-size problems) and the eigenvalue
step in double precision with a per-eigenvector check (c610abe, 2ef37ae).
Command: `kaggle/run.py` preset "speed" at 2ef37ae (`gpu_pipeline.py --devices cuda --precisions
single --skip-serial --no-reliability --rashba 0 --spin-blocks`) · Kaggle run 21 Nov 20:34–20:52
(version 13) · files `results/kaggle/20261121-205243/out/*.csv` + log

| Supercell (mesh) | Configs | PythTB 4 cores | **Ours T4 single** | **Speed-up** | Z2 agrees |
|---|---|---|---|---|---|
| 8×8 (6×6) | 40 | 32.8 s | 7.4 s | 4.4× | 40/40 |
| 12×12 (6×6) | 40 | 267.3 s | 16.8 s | **15.9×** | 40/40 |
| 16×16 (6×6) | 16 | 574.6 s | 31.3 s | **18.3×** | 16/16 |

No Newton-Schulz fallbacks; eigenvalue retries 0/480, 5/480, 6/192, none needing the CPU.
One test failed on KAG (`test_spin_blocks`: `.numpy()` on a GPU tensor, a test bug, fixed in
315bff3); the engine's results are unaffected.

Where the GPU time goes:

| | Solve | Wilson total | of which eigenvalues (double) |
|---|---|---|---|
| 8×8 | 5.34 s (R9: 3.97) | 2.05 s | 2.02 |
| 12×12 | 12.24 s (R9: 15.73) | 4.55 s | 4.33 |
| 16×16 | 27.33 s (R9: 36.26) | 3.98 s | 3.66 (R9: 13.81) |

Findings:
- **Spin-block split on the GPU helps only for big supercells:** solve 1.3× faster at 12×12 and
  16×16, but 1.3× *slower* at 8×8 (128×128 blocks are too small for the GPU's batched `eigh`).
  On MAC's CPU it was 2.6× faster even at 8×8. The sweep uses it from 12×12 up.
- **Accurate eigenvalues cost T4 time at small sizes** (2 s at 8×8, where the old method took 0.6 s),
  because T4 double precision is slow; at 16×16 it's 3.7× *faster* than R9's step, which had CPU
  fallbacks. Worth re-checking on an A100/H100 (fast double).

## R11. First disorder sweep: pilot, 8×8, both λ_ν (KAG)

What: fraction of disorder samples that are topological vs disorder strength W, for λ_ν = 1.0
(topological when clean) and λ_ν = 1.65 (normal when clean); λ_SO = 0.3, λ_R = 0, 8×8
supercells, 6×6 k-mesh (uncertain samples re-run at 10×10), 100 samples per point, W from 0 to 10
in steps of 0.5. Invariant: Joey's spin-up Chern number C_up (Z2 = C_up mod 2), GPU single
precision, cuSOLVER determinant fix (d823336).
Command: `PRESET=sweep_pilot bash kaggle/gpu_run.sh` (`analysis/sweep.py --size 8 --samples 100
--W-step 0.5`) · Kaggle run 21 Nov 21:23–21:30 · data `results/sweep/pilot_8x8_c_up.csv`,
plot `results/sweep/pilot_8x8_c_up.png` (raw: `results/kaggle/20261121-212251/`)

**Result:**
- λ_ν = 1.65 (normal when clean): 0% topological up to W = 1, then **disorder switches it on**:
  13% at W = 1.5, **60% at W = 2.0**, 80% at 2.5, ~100% for W = 3.5–4.5; then it is destroyed again,
  crossing 50% near W ≈ 7.
- λ_ν = 1.0 (topological when clean): 100% up to W = 4.5, crossing **~50% at W ≈ 7**, ~3% at W = 10.
- Above W ≈ 5 the two curves coincide within error bars: strong disorder washes out the
  difference between the clean starting points.
- Uncertain (after refinement): 0–7 of 100 per point; gap closed: 0 everywhere.
- **GPU time for the whole sweep: 5.1 min** (42 points, 4,200 samples, 0.074 s per sample incl. refinement).

Cross-check with the independent gap-tracking Z2 (cancelled 8×8 run, version 3, 21:00–21:14,
numbers from its log; same λ_ν = 1.0, W = 0–6.5, 100 samples): 100% up to W = 3.5, 99% at 4–4.5,
98% at 5.0, 92% at 5.5, 81% at 6.0, 58% at 6.5, vs C_up 100/100/99/96/87/64%. Agrees within
~1–2 standard errors, with far fewer uncertain samples for C_up (3–6 vs 22–28 per point at W = 6–6.5)
and ~7–15× less time per point (6.5–9 s vs 44–135 s).

Not yet: bigger sizes (transitions at 8×8 are smeared by finite size), the CPU time for the same
sweep (the main sweep runs the CPU subset), and comparison with the papers (on hold).

## R12. What a full sweep costs: PythTB vs C_up on CPU vs C_up on GPU (KAG)

What: per-sample time of three ways to answer "topological or not?" on the SAME Kaggle machine
(4 CPU cores + 1 T4), scaled to a full sweep of 16,400 samples (41 W values × 2 λ_ν × 200).
Measured at λ_ν = 1.65, W = 2.0 (the transition: the GPU path refines the most samples there;
the CPU baselines don't refine at all, so the GPU figures are conservative). Rashba 0, 6×6 mesh.
  - PythTB as-is: Wilson loops + gap-tracking Z2, one process per core (24 / 12 / 8 samples)
  - C_up on CPU: Joey's spin-up Chern number in NumPy, one process per core (same samples)
  - C_up on GPU: the same algorithm batched on the T4, single precision, incl. refinement (100 samples)
Command: `PRESET=cost bash kaggle/gpu_run.sh` (`benchmarks/sweep_cost.py --sizes 8 12 16`) ·
Kaggle run 21 Nov 21:38–21:49 · files `results/kaggle/20261121-213825/out/sweep_cost.csv` + log

| Supercell | PythTB (s/sample) | C_up CPU | C_up GPU | Sweep: PythTB | C_up CPU | **C_up GPU** | Algorithm | GPU on top | **Total** |
|---|---|---|---|---|---|---|---|---|---|
| 8×8 | 0.884 | 0.140 | 0.098 | 4.0 h | 38 min | 27 min | 6.3× | 1.4× | 9.0× |
| 12×12 | 7.015 | 0.888 | 0.199 | 32.0 h | 4.0 h | **55 min** | 7.9× | 4.5× | **35.2×** |
| 16×16 | 37.73 | 4.287 | 0.456 | 171.9 h | 19.5 h | **2.1 h** | 8.8× | **9.4×** | **82.7×** |

Findings:
- **Both gains are real and separate:** the better algorithm gives ~6–9× on any hardware; the GPU
  adds 1.4× at 8×8 rising to 9.4× at 16×16. Bigger systems (what the physics needs) favour the GPU.
- A 16×16 sweep goes from about a week of PythTB on 4 cores to about two hours on one T4.
- These are **estimates** (measured rate × samples), not full runs; the CPU is Kaggle's 4 cores. A
  bigger CPU narrows the gap, a bigger GPU (A100/H100) likely widens it: not measured.

## R13. Finite-size scaling: 12×12 and 16×16 sweeps (KAG)

What: as R11 (C_up, λ_SO = 0.3, λ_R = 0, 100 samples per point, 6×6 mesh, uncertain re-run at
10×10) at bigger supercells: 12×12 for both λ_ν over W = 0–10 (step 0.5); 16×16 for λ_ν = 1.65
over W = 0–5 (step 0.25), around the switch-on.
Command: `PRESET=fss bash kaggle/gpu_run.sh` · Kaggle run 21 Nov 21:58–22:19 · data
`results/sweep/fss_12x12_c_up.csv`, `results/sweep/fss_16x16_c_up.csv` (raw:
`results/kaggle/20261121-215811/`) · slides `results/figures/slides/3_disorder_switch.png`,
`4_sizes.png`

**Switch-on at λ_ν = 1.65 (fraction topological):**

| W | 1.5 | 1.75 | 2.0 | 2.25 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| 8×8 (R11) | 0.13 | | 0.60 | | 0.80 | 0.97 |
| 12×12 | 0.06 | | 0.60 | | 0.90 | 1.00 |
| 16×16 | 0.02 | 0.20 | 0.65 | 0.87 | 0.98 | 1.00 |

- **The switch sharpens with system size** (bigger: lower before, higher after) and the curves
  cross near **W ≈ 1.9–2.0** (rough, from linear interpolation between points; a proper
  crossing/collapse analysis is still to do). The hallmark of a genuine transition.
- Switch-off (12×12): crosses 50% near W ≈ 7.0 (λ_ν = 1.0) and W ≈ 6.7 (λ_ν = 1.65), as at 8×8.
- Uncertain after refinement: 0–11 of 100 per point (most at 16×16, W = 1.75); gap closed: 0.
- GPU time: 12×12, 4,200 samples in 9.6 min (0.14 s/sample); 16×16, 2,100 samples in 10.2 min
  (0.29 s/sample). By R12's rates the same work with PythTB on 4 CPU cores would take ~8 h + ~22 h.

---

## Still to measure

- [x] PythTB **all-cores** baseline on KAG (R7)
- [x] Step 2: batched Wilson loop vs PythTB `berry_phase` (tests; R6, R7)
- [x] Full pipeline end to end, GPU vs best CPU (R7)
- [x] Single precision: Z2 agrees config by config (R7: 40/40 at 8×8 and 12×12)
- [x] Re-run R7 with the Newton-Schulz speed-up (R8)
- [x] R9: final speed run with the timing fix **and** the Hermitian eigenvalue step (headline updated)
- [x] Eigenvalue accuracy/fallbacks (R10: double step, per-eigenvector check)
- [x] Spin-block split for λ_R = 0 (R10: helps from 12×12)
- [ ] Use both T4s
- [x] First sweep, 8×8 pilot (R11): disorder switches topology on at λ_ν = 1.65
- [x] 12×12 and 16×16 sweeps (R13): the switch sharpens with size, crossing near W ≈ 1.9–2.0
- [x] CPU time estimate for whole sweeps (R12)
- [ ] Proper crossing / data-collapse estimate of the critical W (finite-size scaling analysis)
