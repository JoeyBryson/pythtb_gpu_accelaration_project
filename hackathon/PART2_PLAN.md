# Part 2 plan: from where we are to `goal.md`

**Goal** (`goal.md`): show that disorder makes the trivial Kane-Mele insulator (λν = 1.65t, λ_SO = 0.3t,
λ_R = 0) topological at **Wc₁ ≈ 2.0 (our units; "W ≈ t" in Orth)**. Locate it by finite-size scaling with the GPU
pipeline, check it against Orth (2.14 ± 0.07) and Favata (1.92), and report the time-to-result against the best
CPU baseline.

Written Sat 21 Nov, about 20:00. Code freeze Sun about 15:00, so **about 19 h left**. Costs and timings come from
`VIABILITY.md`.

## Where we are

| | Status |
|---|---|
| GPU pipeline (batched `eigh` → Newton-Schulz Wilson loop → Z₂) | ✓ Matches PythTB 40/40. **8.9x vs PythTB on all CPU cores** (R7, T4, 12×12, single precision) |
| Faster Wilson loop | Written, not yet timed on the GPU. A further speed-up of up to 10x may be possible (source to confirm) |
| Reference data (Orth Fig. 5, Favata Fig. 7) | ✓ `analysis/reference_data/` |
| **Stage 0 go/no-go** (L = 8, CPU) | ✓ **Physics GO**: f = 0 → 0.2 → 0.7 → 0.8 → 0.15 over W = 0 to 8 |
| Gap-tracking Z₂ near the transition | ✗ Not converged: 85–95% flagged at W ≈ 2, and up to 6/20 answers change with nk. **The cause is the gap-tracking count, not the physics or the mesh.** |
| **Decision A: invariant** | ✓ **Spin-up Chern number C↑ from the winding of arg det(Wilson loop), Z₂ = C↑ mod 2.** Same answer at nk = 6 and 10 for 137/140 samples (L = 8 and 12, W = 1–8). All 3 changes are flagged by largest step > 0.4. Rule: nk = 6, re-run flagged samples at nk = 10. (`analysis/stage0_chern_up.py`, `results/stage0*_chern_up*.csv`) |
| GPU version of C↑ | ✓ `gpu_engine/spin_chern.py`. All tests pass on the T4 (63 passed, R11 run), after Oscar's cuSOLVER fix for a crash in the default LU routine (d823336). |
| Production runner | ✓ `analysis/sweep.py` (Oscar): saves per point, resumes, refines, CPU estimate. Uses C↑ by default (`--invariant c_up`). |
| **First sweep (R11, 8×8, T4)** | ✓ λν = 1.65: 0% → 60% (W = 2) → about 100% (W = 3.5–4.5) → 50% near W ≈ 7. **GPU time 5.1 min** for 4,200 samples. Cross-checked: the CPU C↑ on the same 100 seeds at W = 2 gives identical counts (56/94). |
| Ensemble analysis (Filip) | ✓ `analysis/ensemble*.py`, `sample_size.py`: per-sample data, population and sample-size figures (L = 2–4, gap-tracking Z₂, CPU double). |
| Many-GPU runner | ✓ `modal_runs/app.py` (Modal, separate from `kaggle/`). Presets test → timing → pilot → wc1 → wc2 → fig5 → big. **Blocked:** the workspace spend limit must be raised in the Modal dashboard. |
| Gap policy (direct vs indirect) | ✓ Not an issue: identical in every sample |

## The plan in five phases

### Phase 1: make the invariant trustworthy at the transition ✓ mostly done (decision A above)

What remains: confirm the GPU C↑ matches the CPU version sample by sample (the Kaggle tests), and check that nk = 10
is enough for refinement at L = 16. The steps below were the original plan, kept for the record.

1. **Mesh convergence on the GPU.** W ∈ {2, 3}, nk ∈ {6, 10, 14, 18}, L ∈ {8, 12}, the same 20 seeds as stage 0.
   Minutes of GPU time.
2. **FHS spin-up Chern number C↑** (λ_R = 0, half-size block). Use the determinants of the same overlap matrices
   (Fukui-Hatsugai-Suzuki). Run it on the same samples, with Z₂ = C↑ mod 2. First on the CPU against the stage 0
   CSVs, then batched on the GPU.
3. **Choose the invariant and nk.** Use whichever reaches both of these more cheaply:
   - stable when nk goes up a step, for at least 95% of samples;
   - at most about 20% unreliable at W ≈ 2.

   Gap-tracking Z₂ and FHS C↑ must agree on every sample where both are reliable.

**Decision point A.** If neither converges at an affordable nk, fall back: run fewer sizes at a finer mesh, and
present Wc₁ as the midpoint at each size instead of a crossing.

### Phase 2: production tooling (Sat evening, in parallel with Phase 1, about 2 h)

- A **runner** around `run_batch` (or the FHS path):
  - parameters (`delta=1.65, rashba=0` explicitly), a W × L grid and the chosen nk;
  - **append every sample to `results/tai_165_raw.csv`**, and resume after a crash or timeout;
  - memory budget raised to about 10 GB per T4;
  - **one process per GPU** (`CUDA_VISIBLE_DEVICES`).
- Time the new Wilson loop and **find out where the extra speed-up comes from**, and whether it includes the spin
  split, so it isn't counted twice.
- Measure the CPU-side share: Wilson-matrix eigenvalues and Z₂ counting. After a 10x GPU speed-up these become the
  bottleneck (Amdahl's law), so move them to the GPU or parallelise them if needed.
- Check the remaining Kaggle GPU quota and session limits.

**Decision point B: base or stretch scope.** Decide from the measured seconds per *reliable* sample at the chosen
nk:

| | **Base scope** (current speed, 1–2 T4s) | **Stretch scope** (about 10x and 2+ GPUs, spin split working) |
|---|---|---|
| Sizes | L = 6, 8, 10, 12 (+16 reduced) | L = 8, 12, 16, 24 |
| Samples near Wc₁ | 500 | 1,000 |
| Expected Wc₁ precision | ± about 0.2 | ± about 0.1 |
| Exponent ν from the collapse | bonus only | a real result (compare with about 2.4–2.6) |
| Wc₂ ≈ 7 | optional, 2 sizes | finite-size scaling as well |
| Extra λν (partial Fig. 5) | none | 3–4 values |
| GPU time (overnight) | about 3–6 h | about 6–8 h |

### Phase 3: production runs (Sat night → Sun morning, unattended)

1. **Coarse scan:** W from 0 to 8 in steps of 0.5, M = 100, two sizes, to locate both transitions. Under 1 h.
2. **Finite-size-scaling runs for Wc₁:** about 10 W values within ±0.6 of the coarse value, at the chosen sizes and
   M (table above).
3. **Then, if time allows:** Wc₂, then the extra λν (stretch scope only).
4. **CPU comparison:** a short PythTB all-cores run at the production parameters, on the same machine.
   Extrapolate it to the full dataset and **label it as extrapolated**.

Save results as they come in. If a run dies, resume it rather than restarting.

### Phase 4: analysis (Sun 08:00–11:00, written Sat night against partial data)

- f(W, L) and u(W, L) with bootstrap errors. Crossings W*(L₁, L₂) with their drift. Collapse if the data allow it.
- Robustness checks: unreliable samples counted as 0 and then as 1, nk ± one step, single vs double precision on a
  subset.
- **Compare:** Wc₁ in [1.8, 2.3]? Above the Born value of about 1.7? Wc₂ in [6.5, 7.5]?
- **Figures:**
  1. f(W) for each L, with Favata's curves, Orth's band, Born's line and a u(W) panel;
  2. the finite-size-scaling zoom (crossings, W* against L, and the collapse if available);
  3. time-to-result, GPU against the best CPU on the same machine.

**Decision point C** (Sun about 10:00): if the crossings aren't resolved, present the midpoint at each size against L.
That still tests the [1.6, 2.6] band and shows the transition sharpening.

### Phase 5: write-up and pitch (Sun 11:00–15:00)

- Update `RESULTS.md` (new R-entries for the production runs) and `goal.md` (ticked criteria).
- Pitch: "We reproduce Orth et al.'s central result, that disorder makes a trivial Kane-Mele insulator topological
  at W ≈ t, with an independent method, and locate it by finite-size scaling. That took N thousand samples, X h on
  the GPU against Y h on the best CPU."
- Caveats slide: the different observables, the references aren't rigorous critical points, small L, and how we
  handled unreliable samples.
- Code freeze about 15:00. Live demo plus a recorded backup.

## Who does what (CLAUDE.md workstreams)

| Workstream | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|---|---|---|---|---|---|
| **GPU engine** | Mesh convergence on the GPU; FHS batched | Runner, multi-GPU, timing, CPU-side share | Run and monitor | CPU comparison | Performance slide |
| **Models and analysis** | FHS on the CPU against the stage 0 data | Analysis code on partial data | — | Crossings, collapse, figures | Result figures |
| **Physics validation** | Choose the invariant and nk (decision A) | Agree base or stretch scope (decision B) | Spot-check samples | Robustness checks, decision C | Caveats, science framing |

## Timeline

| When | Milestone |
|---|---|
| Sat 20:00–22:30 | Phases 1 and 2. **Decisions A and B by about 22:30.** |
| Sat 22:30 → Sun 07:00 | Phase 3 runs unattended (base scope about 3–6 h, stretch about 6–8 h, leaving 1–3 h slack) |
| Sun 08:00–11:00 | Phase 4. Re-run one failed block if needed. Decision C. |
| Sun 11:00–15:00 | Phase 5 and the code freeze |

## Housekeeping

- **Before Phase 3, update `goal.md`:** stage 0 done, the new convergence stage, the "≤ 20% unreliable at Wc₁"
  criterion, and the chosen base or stretch scope.
- **Commit** stage 0 (`analysis/stage0_go_no_go.py` and the two CSVs), `VIABILITY.md` and this plan.
- **Label pre-event and on-the-day work as SoTA requires.** Everything in Part 2 is on-the-day work.
