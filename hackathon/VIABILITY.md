# Viability report: can we hit `goal.md` before the code freeze?

Written 21 Nov 2026, about 19:30, after R7 (`RESULTS.md`). The goal is to find Wc₁ ≈ 2.0 (our units) at
λν = 1.65t by finite-size scaling, and cross-check it against Orth (2.14 ± 0.07) and Favata (1.92).
Numbers are **measured** where marked and are otherwise **rough estimates** with the basis stated.
About 19.5 h remain until the code freeze (Sun about 15:00), including the night.

## Verdict

**Viable, with a smaller scope than `goal.md` currently lists.** The pipeline works end to end on the GPU and
matches PythTB exactly (R7: 40/40 Z₂). It is fast enough for a finite-size-scaling dataset **at L = 6 to 12**
(about 2–3 h on one T4), with L = 16 as a reduced add-on.

Three things decide it:

1. **The physics hasn't been seen yet.** Every run so far used the tutorial parameters, which are deep in the
   topological phase. Stage 0 (5 min, CPU) must show trivial → topological at λν = 1.65 **tonight**. If it
   fails, use the fallback below.
2. **L = 16 costs about 4x more than L = 12 per sample**, and L = 24 is out of reach without the spin split.
3. **Statistical resolution.** With 500 samples per point, the crossing gives Wc₁ to about ±0.1 statistical,
   plus about ±0.1–0.2 systematic from drift at small L. That is enough to test "≈ 2.0, inside [1.8, 2.3]".
   It is **not enough to tell Orth (2.14) from Favata (1.92)**, which differ by about 1σ.

## 1. Where we are

| Needed for the goal | Status | Evidence |
|---|---|---|
| Batched build + `eigh` on the GPU | ✓ | R4a: 21x vs PythTB serial (12×12, single) |
| Batched Newton-Schulz Wilson loop + Z₂ | ✓ | R7: Z₂ identical to PythTB for 40/40 configs, single and double |
| End-to-end GPU vs best CPU, same machine | ✓ | R7: **8.9x vs PythTB on 4 cores** at 12×12, 3.8x at 8×8 |
| Faster Wilson loop (early-stopping Newton-Schulz, eigenvalues on CPU) | written, **not yet timed on the GPU** | commit `e501526` |
| Runs at λν = 1.65, λ_R = 0 | ✗ | Only tutorial parameters so far |
| Stage 0 go/no-go | ✗ | — |
| Production runner (sweep, CSV append, resume) | ✗ | `run_batch` exists; it needs a wrapper |
| Spin-up C↑ cross-check / spin split | ✗ | — |
| Analysis (f, bootstrap, crossings, plots) | ✗ | Reference CSVs ready |

## 2. Cost per sample (T4, single precision)

Measured per-sample costs from R7 (40 configs, nk = 6):
- 8×8: 0.245 s (solve 0.10, Wilson 0.145);
- 12×12: 0.90 s (solve 0.40, Wilson 0.51).

The Wilson loop was the larger stage before the latest fix. Other sizes are scaled by N³ for `eigh` and n³ for
the Wilson matrix products, where N = 4L² and n = 2L².

| L | N (states) | Seconds per sample, pessimistic (R7 code) | Optimistic (Wilson halved; assumption) | Basis |
|---|---|---|---|---|
| 6 | 144 | 0.10 | 0.07 | extrapolated (nk = 8 here, so 64 k-points) |
| 8 | 256 | **0.245** | 0.17 | **measured** |
| 10 | 400 | 0.40 | 0.30 | interpolated |
| 12 | 576 | **0.90** | 0.65 | **measured** |
| 16 | 1024 | 4.5 | 2.5 | extrapolated (×5.6 from 12; R5 raw `eigh` suggests the solve part may be about half this) |
| 24 | 2304 | 35–50 | — | extrapolated: **not feasible** for hundreds of samples |
| 16, spin-up block only | 512 | ≈ 0.6 | ≈ 0.4 | ÷8 (one half-size block; C↑ gives Z₂ = C↑ mod 2) |
| 24, spin-up block only | 1152 | ≈ 5 | ≈ 3 | ÷8; feasible for a reduced run |

Correction to my earlier note: the spin split saves 8x only if we diagonalise **one** spin block (enough at
λ_R = 0, since C↓ = −C↑). Doing both blocks saves 4x.

**Memory.** `configs_per_chunk` budgets 4 GB, and `run_batch` halves it. At L = 16 that leaves **1 config per
chunk**: it still works, but batching over configs is lost. Raising the budget to about 10 GB (T4 = 15.6 GB)
gives about 3 configs per chunk. At L = 24, one full config needs about 6 GB, so it is only viable spin-split.

## 3. Compute budget for the plan

Recommended scope (changes from `goal.md` marked ★):

| Stage | Runs | Configs | T4 time, pessimistic | T4 time, optimistic |
|---|---|---|---|---|
| 0 Go/no-go (CPU, MAC) | L = 8, 6 W × 20 | 120 | 5 min (CPU) | — |
| 1 GPU = CPU at λν = 1.65 | L = 8, 3 W × 20, single + double | 120 | 5 min | 3 min |
| 3 Pilot (incl. nk = 10, ≈ 2.8x cost) | 6 W × 50 × L ∈ {8, 12} × nk ∈ {6, 10} | 1,200 | 25 min | 18 min |
| 4 Coarse scan ★ (W step 0.5, M = 100, L ≤ 12) | 17 W × 100 × L ∈ {8, 12} | 3,400 | 32 min | 23 min |
| 5 FSS for Wc₁ ★ (sizes 6–12) | 10 W × 500 × L ∈ {6, 8, 10, 12} | 20,000 | **2.3 h** | 1.7 h |
| 5 FSS add-on ★ (L = 16, reduced) | 6 W × 200 × L = 16 | 1,200 | 1.5 h | 50 min |
| 5b Wc₂, optional ★ | 10 W × 300 × L ∈ {8, 12} | 6,000 | 1.0 h | 45 min |
| 7 CPU comparison | reuse R7 per-config rates + 40-config check at λν = 1.65 | 80 | 10 min | — |
| **Total GPU** | | ≈ 32,000 | **≈ 6 h on one T4 (≈ 3 h on both)** | ≈ 4.5 h (≈ 2.3 h) |

Splitting the two T4s by L (for example L ∈ {6, 8, 10, 16} on one and L = 12 on the other) roughly halves the
wall time. Running both T4s isn't wired up yet; the simplest way is two processes with `CUDA_VISIBLE_DEVICES`.

**Equivalent CPU time** (Kaggle, PythTB on 4 cores, from R7: 0.94 s per config at 8×8 and 8.0 s at 12×12;
others scaled by N³): stage 5 alone ≈ 17 h for L ≤ 12 and ≈ 13 h more for the L = 16 add-on, about 30 h,
against about 3.8 h on one T4. **That is about 8x, the time-to-result headline** (consistent with R7's 8.9x).
On Oscar's MacBook (10 cores, faster per core) the CPU would take roughly half that. Don't mix machines in one
comparison.

**Kaggle limits (to check):** sessions time out, and there's a weekly GPU quota. Check the hours remaining
before starting stage 5. Save to the CSV as results come in, so a timeout loses nothing.

## 4. Can the crossing be resolved? (rough statistics)

- At M = 500, the error on f near f = 0.5 is about ±0.022.
- **Transition width:** Favata's L = 15 curve rises from 0.15 to 0.98 over about 0.5 in W. Scaling the width
  as L^(−1/ν) with ν ≈ 2.5 gives about 0.65 at L = 8 and about 0.55 at L = 12, so slopes of about 1.5 and 1.8
  per unit W.
- **One crossing (8 vs 12):** δW ≈ √2 × 0.022 / (1.8 − 1.5) ≈ **0.1**. Fitting all 10 W points and 4 sizes
  together should bring this to about **0.07**.
- **Drift systematic:** unknown, guessed at 0.1–0.2 at these sizes, because the clean critical point is close
  (1.5588).
- **Final:** Wc₁ ± about 0.2 total. That tests the pass band [1.8, 2.3]. It doesn't separate Orth from Favata.
- **The collapse fit for ν** at L ≤ 12 will be loose. Treat it as a bonus, not a pass criterion.
- **If the crossings aren't resolved,** fall back to the midpoint (f = 0.5) at each size against L. That still
  tests the [1.6, 2.6] band from earlier, and shows the sharpening.

## 5. Developer time still needed

| Task | Estimate | Blocks |
|---|---|---|
| Stage 0 script (CPU) | 30 min | everything |
| Gap policy: log direct and indirect gaps, base `reliable` on the direct gap (agree with Oscar) | 30 min | stage 4 |
| Production runner: parameters (`delta=1.65, rashba=0`), W/L grid, per-config CSV append, resume, memory budget, two GPUs | 1–1.5 h | stages 3–5 |
| Analysis: f, u, bootstrap, crossings, figures with the reference overlays | 2 h | stage 6 |
| Spin-up block path (C↑ winding): cross-check **and** 8x speed-up for L = 16/24 | 2–3 h | optional; makes L = 16 cheap |
| Write-up and pitch figures | 2 h | Sunday |

That is about 6–9 person-hours, which fits across three people tonight and Sunday morning.

## 6. Proposed timeline

| When | What |
|---|---|
| Sat 19:30–20:00 | **Stage 0** (CPU). Go/no-go decision. Gap-policy fix agreed. |
| Sat 20:00–22:30 | Stages 1 and 3 on Kaggle. Re-time the new Wilson loop. Write the runner and start the analysis code. |
| Sat 22:30 → overnight | Stages 4 and 5 (about 3–6 h), both T4s, saving to CSV as it goes |
| Sun 08:00–11:00 | Analysis: crossings, robustness checks, figures. Re-run anything that failed (about 2–3 h slack). |
| Sun 11:00–15:00 | Optional Wc₂ / spin-split L = 16 if time allows. Pitch, write-up, code freeze. |

Slack: about 6 h if nothing breaks. That allows **one** re-run of stage 5, not two.

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The TAI isn't visible at L ≤ 12 (no switch at λν = 1.65) | low–moderate. Favata sees it at L = 15 with a similar bulk method; Orth says W ≈ t. | fatal for the goal | Stage 0 tonight. Fallback below. |
| Indirect-gap check flags the TAI plateau as unreliable | moderate | high (hollows out f) | Use the direct gap (30 min) |
| nk = 6 too coarse near Wc₁ (wrong Z₂) | moderate | high | Pilot at nk = 10. If needed, about 2.8x cost: drop L = 16 |
| Crossings not resolved beyond the errors | moderate | medium | Fall back to midpoints against L; M = 1000 at L = 8–10 is cheap |
| Overrun / Kaggle timeout or quota | low–moderate | medium | Save as results come in; split across 2 T4s; drop L = 16 first |
| Single precision wrong near the transition | low (R7: 40/40) | medium | Stage 1 double vs single at λν = 1.65 |
| The new Wilson code is slower or different on the GPU | low | low | Re-time it in stage 3; R7 code as fallback |

## 8. Fallback if stage 0 or the crossings fail

The minimum result that is still worth pitching:
- f(W) at λν = 1.65 for L = 8 and 12, overlaid on Favata and Orth, with Wc₁ taken as the midpoint at each size;
- the R7 speed-up (8.9x vs PythTB on all cores, same machine, identical Z₂) and the time-to-result for that
  dataset.

If stage 0 shows no TAI at all, report it honestly as a negative result (the judges value these). Then the
pitch leads with the tool and the speed-up, plus the clean-limit check and a single-λν curve.

## 9. Changes to `goal.md` this implies

- Sizes: **L ∈ {6, 8, 10, 12}** as the core, with **L = 16 reduced**. L = 24 only with the spin split.
- Coarse scan: **W step 0.5, M = 100, L ≤ 12** (not step 0.25 × M = 200 × L ≤ 16).
- Wc₂: **optional**, L ∈ {8, 12}, M = 300.
- The collapse/ν fit becomes a bonus, not a pass criterion.
- Add: "Wc₁ ± about 0.2 is expected; we can't separate Orth from Favata."
