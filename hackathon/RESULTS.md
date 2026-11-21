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

> **Full pipeline (diagonalise → Wilson loop → Z2), 40 disorder configurations of a 12×12
> supercell, same Kaggle machine (R7): PythTB on all 4 CPU cores 321 s, ours on one T4 in
> single precision 36 s: 8.9× faster** (12.9× vs PythTB on one core), with Z2 identical to
> PythTB for all 40 configurations. Speed-up grows with size (3.8× at 8×8).
> Not yet included: the faster Wilson loop (early-stopping Newton-Schulz, CPU eigenvalues),
> the second T4, or the spin-block split. Double precision on the T4 gives only 2.1×.
>
> ⚠ **Baseline correction pending:** the 321 s all-cores time included PythTB's untimed
> reliability diagnostics (~15% of wall time on KAG, see "Timing fix" under R7). Estimated
> corrected baseline ~274 s, i.e. **~7.6×** rather than 8.9× (~3.3× rather than 3.8× at 8×8).
> Re-run R7's PythTB parallel baseline with the fixed `cpu_baseline.py` before quoting.
>
> **Latest (R8, 12×12):** with faster Newton-Schulz, GPU single 32.7 s: ~8.8× vs the corrected
> all-cores baseline, and about half of that GPU time was one fixable step (eigenvalues on the
> CPU). Final numbers: R9.

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

---

## Still to measure

- [x] PythTB **all-cores** baseline on KAG (R7)
- [x] Step 2: batched Wilson loop vs PythTB `berry_phase` (tests; R6, R7)
- [x] Full pipeline end to end, GPU vs best CPU (R7)
- [x] Single precision: Z2 agrees config by config (R7: 40/40 at 8×8 and 12×12)
- [x] Re-run R7 with the Newton-Schulz speed-up (R8)
- [ ] R9: final speed run with the timing fix **and** the Hermitian eigenvalue step; update the headline
- [ ] Use both T4s
- [ ] Full phase-diagram sweep: CPU time vs GPU time (the headline for the pitch)
