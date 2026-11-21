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

> **Diagonalisation step, 20 disorder configurations of a 12×12 supercell (KAG): PythTB 146 s,
> ours on the T4 in single precision 6.8 s: 21× faster, eigenvalues matching PythTB to 2e-5.**
> Of that 21×: batching on the same CPU gives 5.0×, moving to the GPU a further 4.3×.
> Caveat: baseline is PythTB on one core; the all-cores baseline on KAG is still to run.

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

---

## Still to measure

- [ ] PythTB **all-cores** baseline on KAG (same machine as the GPU numbers)
- [ ] Step 2: batched Wilson loop (Newton-Schulz + tree product) vs PythTB `berry_phase`
- [ ] Full pipeline (steps 1+2+Z2) end to end, GPU vs best CPU
- [ ] Single precision: does Z2 still agree config by config?
- [ ] Use both T4s
- [ ] Full phase-diagram sweep: CPU time vs GPU time (the headline for the pitch)
