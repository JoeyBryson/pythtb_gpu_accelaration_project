# Part 2 plan: from where we are to `goal.md`

**Goal** (`goal.md`): show that disorder makes the trivial Kane-Mele insulator (λν = 1.65t, λ_SO = 0.3t,
λ_R = 0) topological at **Wc₁ ≈ 2.0 (our units; "W ≈ t" in Orth)**. Locate it by finite-size scaling with the GPU
pipeline, check it against Orth (2.14 ± 0.07) and Favata (1.92), and report the time-to-result against the best
CPU baseline.

Updated Sat 21 Nov, about 22:30. Code freeze Sun about 15:00, so **about 17 h left**; data is needed by about
08:00.

## Where we are

### Done

| | Status |
|---|---|
| GPU pipeline | ✓ **18.3x vs PythTB on all CPU cores** at 16×16 (R10: T4, single precision, Rashba 0, spin blocks), 15.9x at 12×12. Z₂ identical to PythTB. |
| **Stage 0 go/no-go** (L = 8, CPU) | ✓ **Physics GO**: trivial → TAI → trivial. The gap-tracking Z₂ is not converged near W ≈ 2: this is the counting method, not the physics. |
| **Decision A: invariant** | ✓ **Spin-up Chern number C↑ from the winding of arg det(Wilson loop), Z₂ = C↑ mod 2.** Same answer at nk = 6 and 10 for 137/140 samples (L = 8, 12; W = 1–8); all 3 changes are flagged by largest step > 0.4. Rule: nk = 6, re-run flagged samples on a finer mesh. |
| GPU version of C↑ | ✓ `gpu_engine/spin_chern.py`. All tests pass on the T4 (63 passed, R11 run), with Oscar's cuSOLVER fix for a crash in PyTorch's default MAGMA LU routine on the T4 (d823336). |
| Production runner | ✓ `analysis/sweep.py` (Oscar): saves after each point, resumes, refines flagged samples, CPU estimate. C↑ by default (`--invariant c_up`). |
| **First sweep (R11, 8×8, T4)** | ✓ λν = 1.65: 0% → 60% (W = 2) → about 100% (W = 3.5–4.5) → 50% near W ≈ 7. **5.1 min of GPU time** for 4,200 samples (0.074 s per sample). The CPU C↑ on the same 100 seeds at W = 2 gives identical counts (56/94). |
| **Fair cost comparison (R12, T4 + 4 CPU cores, same machine)** | ✓ Per sample: PythTB 0.88 / 7.0 / 37.7 s, C↑ CPU 0.14 / 0.89 / 4.3 s, **C↑ GPU 0.098 / 0.199 / 0.456 s** at L = 8 / 12 / 16 (GPU incl. refinement). **Total 9× / 35× / 83×**, split into the algorithm (6–9×, any hardware) and the GPU (1.4× → 9.4×, growing with L). A 16×16 sweep: about a week of PythTB on 4 cores against about 2 h on one T4 (estimates from measured rates). |
| Coarse finite-size sweep preset | ✓ Kaggle preset `fss` (Oscar): 12×12 over the full W range (both λν, 100 samples) and 16×16 at λν = 1.65 for W ≤ 5 (100 samples). Not yet run, as far as the repo shows. |
| Slide charts | ✓ `presentation/style.py`, `results/figures/slides/` (Oscar): speed, algorithm vs GPU split, disorder switch |
| Ensemble figures (Filip) | ✓ `analysis/ensemble*.py`, `sample_size.py`: per-sample data, population and "why many samples" figures, including a 16×16 ensemble (72 samples per W). For the pitch. |
| Reference data | ✓ Orth Fig. 5 and Favata Fig. 7, extracted exactly from the PDFs (`analysis/reference_data/`) |

### Decision B (made 22:00): Kaggle only, Oscar's account

- **Modal:** two runners exist (`modal_runs/` for many GPUs, Filip's `modal_gpu/` for one GPU), but we couldn't get
  the workspace working. Parked.
- **Joey's Kaggle account:** jobs stay queued (more than 30 min) and never start, probably phone verification.
  Tonight's runs go on **Oscar's account**, which starts jobs promptly.
- **Hardware:** "GPU T4 x2", so 2 T4s per session if we run one process per GPU.
- **Scope:** L = 8, 12, 16 at 1,000 samples near each transition; **L = 24 on the second T4** if the overnight time
  allows (about 5 s per sample, estimated from R12 by scaling with matrix size). L = 32 dropped.

## Remaining work

### Phase 2b: build (Sat 22:15–23:15)

| # | Task | Owner | Status |
|---|---|---|---|
| 1 | **Two-GPU helper:** run two `sweep.py` jobs at once, one per T4 (`CUDA_VISIBLE_DEVICES=0/1`) | GPU engine | To do |
| 2 | **Presets `wc1` and `wc2`** in `kaggle/run.py`, added only (existing presets untouched): one T4 runs L = 8, 12, 16; the other runs L = 24 (or splits the sizes if L = 24 is dropped). One CSV per size. | GPU engine | To do |
| 3 | **Refinement mesh:** nk = 12 for L = 8 and 16 (Dirac point on the mesh when 3 doesn't divide L, Filip's finding), 10 for L = 12 | Physics validation | **To agree** |
| 4 | **Upload race in `gpu_run.sh`:** a job pushed seconds after an upload ran the previous code version (21:40, Joey's account). Wait for the new version before pushing, or print a version stamp in the log. | Oscar | **To fix before the overnight runs** |
| 5 | **Finite-size-scaling analysis** (no GPU): f(W, L) with bootstrap errors, crossings and drift, reliability-bias check, overlay with Favata, Orth and Born | Models and analysis | To do |
| 6 | Joey's uncommitted `gpu_run.sh` fix (jobs point at the account's own code dataset; no change for Oscar) | Oscar / Joey | Decide |

### Phase 3: production runs (Oscar's account, unattended)

Speed on a T4 with C↑ (**measured, R12**): 0.098 / 0.199 / 0.456 s per sample at L = 8 / 12 / 16, including
refinement at the transition. L = 24 is estimated at about 5 s.

| Order | Run | Sizes | W | Samples per point | T4 time |
|---|---|---|---|---|---|
| 1 | **`fss`** (Oscar's preset): coarse curves at 12 and 16 | 12, 16 | full range / W ≤ 5 | 100 | ≈ 30 min |
| 2 | **`wc1`, core goal** | 8, 12, 16 | 1.0–3.0, step 0.2 (11) | 1,000 | ≈ 2.3 h on one T4 (8: 18 min, 12: 36 min, 16: 84 min) |
| 2b | `wc1`, largest size (second T4, at the same time) | 24 | 1.0–3.0, step 0.2 | 300 | ≈ 4.5 h (estimate) |
| 3 | **`wc2`** | 8, 12, 16 | 6.0–8.0, step 0.2 (11) | 1,000 | ≈ 2.3 h |
| 4 | Fig. 5 equivalent (optional) | 8, 12 | 7 λν × about 11 W around Orth's Wc | 300 | ≈ 1.5 h |

Expected precision with L = 8–16 and 1,000 samples: Wc₁ **± about 0.15**, about 0.1 if L = 24 works. That tests
[1.8, 2.3]; separating Orth (2.14) from Favata (1.92) is marginal at best.

### Phase 4: analysis (Sun 08:00–11:00; code written tonight against R11 and partial data)

- f(W, L) and u(W, L) with bootstrap errors. Crossings W*(L₁, L₂) and their drift. Collapse only as a bonus.
- **Robustness:** unreliable samples counted as 0 and then as 1, refinement mesh (10 vs 12), largest-step threshold
  0.35 / 0.45.
- **Compare:** Wc₁ in [1.8, 2.3]? Above Born (about 1.7)? Wc₂ in [6.5, 7.5]?
- **Figures:**
  1. f(W) for each L, with Favata's curves, Orth's band, the Born line and a u(W) panel;
  2. the finite-size-scaling zoom (crossings, W* against L);
  3. time-to-result from R12 (`cost`): PythTB vs C↑ CPU vs C↑ GPU, same machine. Oscar's slide charts already
     cover this.
- **Decision C** (about 10:00): if the crossings aren't resolved, show the midpoint at each size against L instead.

### Phase 5: write-up and pitch (Sun 11:00–15:00)

- `RESULTS.md` entries for `cost`, `wc1` and `wc2`; tick the criteria in `goal.md`.
- Pitch: "We reproduce Orth et al.'s central result, that disorder makes a trivial Kane-Mele insulator topological
  at W ≈ t, with an independent method, and locate it by finite-size scaling: N thousand samples, X h on the GPU
  against Y h on the best CPU."
- **Speed claims:** quote the per-GPU speed-up against all cores of the same machine. Separate the algorithm's gain
  (C↑ vs PythTB's gap tracking) from the GPU's (from `cost`).
- Caveats: the different observables, the references aren't rigorous critical points, small L (≤ 16), the
  handling of unreliable samples, λ_R = 0 only.
- Code freeze about 15:00. Live demo plus a recorded backup.

## Timeline

| When | Milestone |
|---|---|
| Sat 22:15–23:15 | Phase 2b (tasks 1–5); Oscar fixes the upload race |
| About 23:15 | `fss` (about 30 min), as a quick check that 12 and 16 behave |
| About 23:45 → Sun 04:30 | `wc1` then `wc2` on one T4 (about 4.6 h); L = 24 on the other T4 |
| Sun 04:30 → 07:00 | Fig. 5 runs if wanted; slack for one re-run |
| Sun 08:00–11:00 | Phase 4 |
| Sun 11:00–15:00 | Phase 5 and the code freeze |

## Risks

| Risk | Status / mitigation |
|---|---|
| Physics doesn't show the transition | ✓ Retired (stage 0, R11) |
| Invariant not converged | ✓ Retired (C↑) |
| **Kaggle access** (one working account, queues) | Run everything on Oscar's account; save as results arrive; resume rather than restart |
| **Stale code on Kaggle** (upload race) | Task 4 before the overnight runs |
| L = 24 slower than estimated | It runs on its own T4, so it can't delay L ≤ 16; stop it at 07:00 and use what's saved |
| Crossings not resolved at L ≤ 16 | Decision C: midpoints against L |

## Housekeeping

- **Update `goal.md`** with decision B (Kaggle, L ≤ 16, 500 samples) and the expected ± 0.2 precision.
- **`VIABILITY.md` is out of date** (written before C↑ and before Kaggle-only). Mark it superseded by this plan.
- **Label pre-event and on-the-day work as SoTA requires.** Everything in Part 2 is on-the-day work.
