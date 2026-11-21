# Results log

Every number we might quote, with where it came from. Rule: **never compare numbers from
different machines** without saying so. Raw data is in the file named under each result.
Protocol (what each benchmark measures and why): [`benchmarks/README.md`](benchmarks/README.md).
Physics checks against the papers: [`VALIDATION.md`](VALIDATION.md).
One-page charts of R7–R11 (and the earlier runs): `results/benchmark_comparison.html` (open in a browser).

## Machines

| ID | Machine | CPU | GPU | Software |
|---|---|---|---|---|
| **MAC** | Oscar's MacBook Air | Apple Silicon (arm), 10 cores, Apple Accelerate BLAS | none (CPU only) | Python 3.12, numpy 2.5.5, torch 2.15 (CPU) |
| **MBP** | Filip's MacBook Pro | Apple M2 Max, 12 cores (8 performance + 4 efficiency), 64 GB, Apple Accelerate BLAS | none (CPU only; Metal tested in R8) | Python 3.12, numpy 2.5.5, torch 2.15 (CPU); TF runs: tensorflow 2.18.1 + tensorflow-metal 1.2.0, numpy 2.0.2 |
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

## R7. Where PythTB's `solve_model` spends its time (MBP)

What: line profile of `WFArray.solve_model` on a clean supercell, 24×24 mesh (576 k-points), plus
variants of the diagonalisation on the same H(k) stack.
Command: `uv run --with line_profiler --with scipy python benchmarks/profile_solve_model.py` · 21 Nov 19:07 · log `results/profile_solve_model.txt`

| Supercell | Matrix | Total | `eigh` | Hermiticity check (`np.allclose`) | Everything else |
|---|---|---|---|---|---|
| 1×1 | 4 | 2.1 ms | 51% | 8% | 41% (building H(k), Python overhead) |
| 3×3 | 36 | 122 ms | 88% | 6% | 6% |
| 6×6 | 144 | 2.56 s | 91% | 5% | 4% |
| 8×8 | 256 | 10.7 s | 93% | 5% | 2% |

Diagonalisation variants at 8×8 (576 matrices of 256×256): `eigh` 9.74 s; with 1 BLAS thread
10.0 s (so **NumPy's batched `eigh` already runs on one core**); eigenvalues only (`eigvalsh`)
2.96 s (eigenvectors are ~70% of the cost); SciPy occupied-states-only `eigh` in a loop 11.3 s
(no gain). PyTorch CPU `eigh` (384 matrices of 144×144): 1.63 s with 8 threads, 1.67 s with 1,
NumPy 1.52 s.

## R8. PythTB's TensorFlow path on Apple Metal (MBP)

What: PythTB's built-in `solve_model(use_tensorflow=True)` with Apple's Metal plugin.
Command: `uv run --with "tensorflow==2.18.1" --with tensorflow-metal python benchmarks/tf_metal_vs_numpy.py` · 21 Nov 19:10 · log `results/tf_metal_vs_numpy.txt`

- **Metal runs none of our maths on the GPU.** With soft placement off, only float32 matmul has a
  Metal kernel; `eigh`, `svd`, `eig` (any dtype) and every complex op fall back to the CPU.
- Yet `solve_model` is faster with `use_tensorflow=True` (NumPy vs TF, same environment, energies equal to ≤ 1.1e-13):

| Supercell | 2×2 | 3×3 | 4×4 | 6×6 | 8×8 |
|---|---|---|---|---|---|
| NumPy | 0.015 s | 0.119 s | 0.362 s | 2.50 s | 10.8 s |
| TensorFlow | 0.006 s | 0.027 s | 0.087 s | 0.61 s | 3.29 s |
| TF speed-up | 2.6× | 4.4× | 4.2× | 4.1× | 3.3× |

- **The gain is CPU threads, not Metal.** 6×6 `solve_model`: TF with 1 thread 3.33 s (slower than
  NumPy's 2.49 s), 4 threads 1.04 s, default 0.65 s; without the Metal plugin installed 0.64 s.
- Metal float32 matmul vs NumPy (576 × M × M, incl. copy back): 0.4× at M = 64, 2.0× at 256, 2.7× at 512.
- Same comparison end to end on the small disorder sweep (`scripts/sample_cpu_sim.py --tensorflow`,
  3×3, 54 configs): 4.3 s → 2.1 s, Z2 identical for all 54. Files `results/sample_cpu_sweep_{numpy,tf}.csv`.

## R9. Disorder sweep through every engine, same configurations (MBP)

What: 9 disorder strengths (W = 0–24, our convention) × 6 configurations, tutorial parameters,
`scaled_nk` meshes, every engine on the same seeds (engines defined in `benchmarks/README.md`).
PythTB timed with `check=False`; reliability flags from a separate untimed pass.
Command: `uv run python benchmarks/team_tools_sweep.py --sizes 3 6 --configs 6` · 21 Nov 18:47 · file `results/team_tools_sweep.csv` (rows with `NS-*`/`SVD-*` engines; earlier rows are the run before `wilson.py` existed)

Total time for all 54 configurations, and speed-up vs PythTB serial:

| Engine | 3×3 (16×16 mesh) | 6×6 (8×8 mesh) | 6×6 solve / Wilson |
|---|---|---|---|
| PythTB serial | 4.19 s | 20.5 s | – |
| **PythTB, 12 processes** | **0.88 s (4.8×)** | **4.17 s (4.9×)** | – |
| batched solve + PythTB Wilson, double | 3.36 s (1.2×) | 20.3 s (1.0×) | 15.0 / 5.2 s |
| batched solve + PythTB Wilson, single | 2.80 s (1.5×) | 13.0 s (1.6×) | 7.7 / 5.2 s |
| `run_batch` (NS Wilson), double | 3.36 s (1.2×) | 18.7 s (1.1×) | 15.1 / 3.6 s |
| `run_batch` (NS Wilson), single | 2.40 s (1.7×) | 10.0 s (2.0×) | 7.7 / 2.3 s |
| batched + SVD Wilson, double | 3.11 s (1.3×) | 19.5 s (1.1×) | 14.9 / 4.6 s |
| batched + SVD Wilson, single | 2.28 s (1.8×) | 11.2 s (1.8×) | 7.7 / 3.6 s |

Every engine gave the same Z2 as PythTB serial for all 108 configurations, band gaps within
1e-8 (double) / 1e-3 (single); 0 Newton-Schulz fallbacks. Physics (same for every engine):
3×3 is 100% topological up to W = 4, 75% at W = 6, 20% at W = 8, 0% from W = 12; 6×6 is 100% up
to W = 6 and 0% from W = 8. Unreliable flags cluster at W = 6–8, i.e. at the transition.

## R10. The batched Wilson loop step by step (MBP)

What: `wilson.py` stages vs PythTB's `berry_phase`, on the same states: 6 configurations at W = 4.
Command: `uv run python benchmarks/profile_wilson.py` · 21 Nov 19:06 · log `results/profile_wilson.txt`

| Supercell (overlap size) | PythTB `berry_phase` | Ours NS, double | Ours NS, single | Ours SVD, double | Ours SVD, single |
|---|---|---|---|---|---|
| 3×3 (18×18) | 0.142 s | 0.140 s (1.0×) | 0.096 s (1.5×) | 0.115 s (1.2×) | 0.086 s (1.6×) |
| 6×6 (72×72) | 0.589 s | 0.388 s (1.5×) | 0.248 s (2.4×) | 0.502 s (1.2×) | 0.405 s (1.5×) |
| 8×8 (128×128) | 1.484 s | 0.811 s (1.8×) | 0.514 s (2.9×) | 1.183 s (1.3×) | 0.817 s (1.8×) |

Stages of the NS path, double, 8×8 (ms): Newton-Schulz (12 steps) 467, `eigvals` 266, overlaps 40,
convergence check 22, tree product 17. Newton-Schulz is 58% at 8×8 (84% at 3×3); `eigvals` grows
from 4% at 3×3 to 33% at 8×8. Smallest overlap singular value: 0.963 (3×3), 0.979 (6×6, 8×8).
torch thread count makes no difference: NS 504 ms with 1 thread vs 467 ms with 8; SVD 870 vs 861 ms.

## R11. Newton-Schulz: how many steps are needed (MBP)

Same run as R10. Worst-link residual ‖XᴴX − I‖ after each step, 6×6 double:
1e-3, 1e-6, 1e-12, 3e-15, then flat. Steps to reach `wilson.NS_TOL`: 3–4 (double), 2 (single, whose
rounding floor is ~1e-6). `wilson.py` always runs `NS_MAX_STEPS = 12`.

| Case | 12 steps + check | 4 steps + check | 3 steps + check | max \|U − U_svd\| at 12 / 4 / 3 steps |
|---|---|---|---|---|
| 6×6 double | 294 ms | 105 ms | 80 ms | 4e-15 / 4e-15 / 4e-13 |
| 6×6 single | 198 ms | 71 ms | 56 ms | 2e-6 / 1e-6 / 1e-6 |
| 8×8 double | 492 ms | 177 ms | 138 ms | 4e-15 / 3e-15 / 3e-13 |
| 8×8 single | 290 ms | 104 ms | 81 ms | 2e-6 / 1e-6 / 1e-6 |

Projected (component swap from R10/R11, **not measured end to end**): stopping at 4 steps would take
the Wilson step to ~3.0× faster than PythTB at 6×6 and 8×8 in double (from 1.5–1.8×) and ~4.5–4.9×
in single (from 2.4–2.9×).

### What R7–R11 tell us

- **Diagonalisation is the cost.** It is ~90% of `solve_model` from 3×3 up (R7), and still ~80% of
  our batched pipeline after the Wilson loop got faster (R9: 15.1 of 18.7 s at 6×6 double).
  Speeding up the Wilson loop alone cannot move the total much; the GPU `eigh` (R4a) is what matters.
- **On a CPU, "batched" is not "parallel".** NumPy's and PyTorch's batched `eigh`, and PyTorch's
  batched matmul and SVD, each run on one core here (R7, R10). PythTB on 12 processes is therefore
  2.4–4.5× faster than our batched pipeline on the same CPU (R9). Our design pays off only on a GPU.
  This matches Oscar's R6 on a different Mac.
- **PythTB's existing TensorFlow switch is a free ~4× on a CPU** (R8), because TF threads its
  batched `eigh`. It is a fairer "best CPU" reference for solve than single-core PythTB, alongside
  the multi-process baseline. On a Mac it never touches the GPU.
- **Newton-Schulz beats SVD where it should**: from 72×72 overlaps up, and more so in single
  precision (R10). At 18×18 the batched SVD is faster. Both give the same Z2 everywhere (R9).
- **Cheap win: stop Newton-Schulz when it has converged** (R11). 3–4 steps instead of 12 gives the
  same answer and roughly halves the Wilson step. Check convergence every step or two
  with 12 as the cap, so the safeguard near gap closings stays.
- **Next Wilson-loop cost: `eigvals`** of the Wilson matrices (33% at 8×8, R10).
- **Baseline correction:** R3's wall times include the untimed reliability diagnostic (~20%); see
  `benchmarks/README.md`, "How a comparison is made".

---

## Still to measure

- [ ] PythTB **all-cores** baseline on KAG (same machine as the GPU numbers)
- [ ] Step 2: batched Wilson loop (Newton-Schulz + tree product) vs PythTB `berry_phase` (CPU done: R10; GPU still to do)
- [ ] Run `benchmarks/team_tools_sweep.py` on KAG: same sweep, adds CUDA engines automatically
- [ ] Newton-Schulz with early stopping, measured end to end (R11 projection)
- [ ] Full pipeline (steps 1+2+Z2) end to end, GPU vs best CPU
- [ ] Single precision: does Z2 still agree config by config?
- [ ] Use both T4s
- [ ] Full phase-diagram sweep: CPU time vs GPU time (the headline for the pitch)
