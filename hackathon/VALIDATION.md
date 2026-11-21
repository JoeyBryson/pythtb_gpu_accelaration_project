# Validation targets: what the papers say, and what we check against

So that "we got the physics right" is a list of checks with numbers, not a feeling.

## Our model vs the papers' model

Both papers use the same Kane-Mele Hamiltonian as PythTB's `kane_mele(delta, t, soc, rashba)`:

| Paper symbol | Meaning | PythTB argument | Checked? |
|---|---|---|---|
| t | nearest-neighbour hopping | `t` | ✓ |
| λ_SO | intrinsic spin-orbit coupling | `soc` | ✓ clean gap matches 2·\|3√3·λ_SO − λ_ν\| to 4 d.p. (21 Nov) |
| λ_ν (Orth) = Δ (Favata) | staggered sublattice potential, ±λ_ν on A/B | `delta` | ✓ same check |
| λ_R | Rashba spin-orbit coupling | `rashba` | not yet checked separately |
| W | Anderson disorder strength | `W` in `disordered_kane_mele` | **conventions differ, see below** |

### Disorder convention: a factor of 2

| Source | On-site energy per site | In our units |
|---|---|---|
| **Our code** | uniform in [−W/2, W/2] | W |
| Favata & Marrazzo 2023 (arXiv:2301.02612, Eq. 14) | w_i ∈ [−W/2, W/2] | **same as ours** |
| Orth et al. 2016 (arXiv:1512.03233, Eq. 1) | W·ε_i, ε_i uniform in [−1, 1], i.e. [−W, W] | **W_ours = 2 × W_Orth** |

We keep Favata's convention (it's the PythTB-based method closest to ours) and
**double every W quoted from Orth** before comparing.

## The papers

**Orth, Sekera, Bruder, Schmidt (2016)**, *The topological Anderson insulator phase in
the Kane-Mele model*, arXiv:1512.03233. In `GPU_experimentation/papers/`.
- Method: conductance of a ribbon (Kwant), w = 93a × l = 150a. Not Wilson loops, so we
  compare *where* the topological region is, not the method.
- Main parameters: λ_SO = 0.3t, λ_R = 0, λ_ν ∈ {1.45t, 1.65t, 1.85t}.
- Numbers are mostly in figures (Fig. 2, 4, 5). Fig. 5 (critical W vs λ_ν) has no table.

**Favata & Marrazzo (2023)**, *Single-point spin Chern number in a supercell framework*,
Electron. Struct. 5 014005, arXiv:2301.02612. Not yet in `papers/`.
- Method: spin Chern number at the Γ point only, on L×L supercells (2L² sites),
  L = 15 and 42, M = 50 configurations (M = 100 for 5 ≤ W/t ≤ 10).
- Code: [spinv](https://github.com/roberta-favata/spinv), **built on PythTB**. This is the
  closest existing tool to ours and a natural second baseline ("spinV" in Joey's plan).
- TAI parameters (Fig. 7): λ_SO = 0.3t; left Δ/λ_SO = 5.5, λ_R = 0; right Δ/λ_SO = 5.3, λ_R/λ_SO = 1.

## Checks, in order of difficulty

| # | Check | Expected (our W convention) | Source | Status |
|---|---|---|---|---|
| 1 | Tutorial params: δ = 1.0 → Z2 = 1, δ = 2.5 → Z2 = 0 | 1, 0 | PythTB tutorial | ✓ tested |
| 2 | Clean gap, λ_R = 0, λ_SO = 0.3 | 2·\|3√3·0.3 − λ_ν\| | Orth p.2 / Kane-Mele | ✓ matches to 4 d.p. |
| 3 | Clean transition, λ_R = 0, λ_SO = 0.3 | Z2 = 1 for λ_ν < 1.559, 0 above | Orth p.2 | ✓ **only on fine meshes** (λ_ν = 1.50 needs ≥ 48×48; 1.55 needs ~120×120) |
| 4 | λ_ν = 1.45: topological, stays so with disorder | Z2 = 1 for small W | Orth Fig. 2 | to do |
| 5 | **TAI**: λ_ν = 1.65, λ_R = 0: trivial at W = 0, topological from W ≈ 2 | Z2: 0 → 1 near W ≈ 2 | Orth (W_Orth ≈ 1) and Favata ("TAI appears at about W/t = 2"), **agree after the factor of 2** | to do: the headline check |
| 6 | λ_ν = 1.85: TAI weak/unstable | fraction topological well below 1 | Orth Fig. 2 | to do |
| 7 | Very strong disorder → trivial | Z2 = 0 | both | ✓ tested (2×2, W = 20) |
| 8 | Finite Rashba narrows the TAI | narrower W range | Orth Fig. 3, Favata Fig. 7 | stretch |

## Caveats to keep in mind

- **Mesh near transitions.** Wrong Z2 on coarse meshes near a gap closing, and the
  check marks some cases as unreliable (`ConfigResult.z2_reliable`). The TAI is a gap closing
  and reopening, so expect to need fine meshes exactly there.
- **Spectral gap vs mobility gap (to check with the physics team).** In the TAI the
  disorder can fill the gap with *localised* states. Conductance doesn't see them, but
  a supercell band gap may close, and our Wilson loop needs a gap at every k. Favata
  handles this by tracking a minimum gap (their Fig. 7 inset). We need to decide what
  to do when `band_gap <= 0`.
- **Different observables.** Orth measures edge conductance; Favata and we measure a bulk
  invariant. They should agree on where the topological region is (bulk-boundary
  correspondence), but transitions may sit at slightly different W at finite size.
