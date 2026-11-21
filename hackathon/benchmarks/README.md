# Benchmark protocol

What we measure, how, and why, so every speed-up we quote is well defined.

## The unit of work: one configuration

One **configuration** is one imperfect material, taken all the way to a yes/no answer:

| Stage | What happens | Code |
|---|---|---|
| build | Kane-Mele model → n×n supercell → random on-site energy per site (strength W) | `reference.disordered_kane_mele` |
| solve | diagonalise H(k) at every k-point (matrix size 4n² × 4n²) | `reference.solve` (PythTB `solve_model`) |
| wilson | Wilson loop along k_y at every k_x → Wannier centres | `reference.wannier_centres` (PythTB `berry_phase`) |
| z2 | Wannier-centre flow → Z2 = 0 (normal) or 1 (topological) | `topology.z2_from_wannier_flow` |

`reference.run_config(n, W, nk, seed)` runs all four and times each stage. The same
`seed` always gives the same material, so CPU and GPU runs can be checked against
each other config by config.

## The scenarios

All run with `uv run python benchmarks/cpu_baseline.py <scenario> ...` from `hackathon/`.

| Scenario | What | Why it's there |
|---|---|---|
| `clean` | one clean supercell per size, fixed 31×31 mesh | matches the prep runs in `CLAUDE.md` (Kaggle T4 numbers) |
| `scaled` | one clean supercell per size, mesh ≈ 48/n per direction | the meshes we'd actually use: bigger supercells need fewer k-points |
| `serial` | N disorder configs, one after another in a Python loop | the naive way a PythTB user would do it today |
| `parallel` | N disorder configs, one per CPU core (1 BLAS thread each) | **the strongest simple CPU baseline**: what a judge will ask "did you compare against?" |
| `both` | `serial` then `parallel` on the same configs | also checks both give identical Z2 for every config |

**Which number do we quote?** The GPU speed-up for the disorder sweep should be quoted
against `parallel` (honest) and may also be shown against `serial` (what users do
today). Always say which.

## Rules (from CLAUDE.md)

- Compare against PythTB 2.0's own vectorised code, not a hand-written slow loop.
- Warm up before timing (the first call pays import and allocation costs).
- **Never time the reliability diagnostics.** `run_config(check=True)` (the default) adds an SVD
  over every link, ~15–20% of the pipeline. Timed runs use `check=False`; the `z2_reliable` flags
  come from one untimed pass (`cpu_baseline.reliability_pass`, shared by serial and parallel;
  skip it with `--no-reliability`). Fixed 21 Nov ~19:30: earlier serial/parallel wall times
  (R3, R7's 4-core column) include the diagnostics.
- Clean/scaled: best of 3 runs (least affected by background noise); median also saved.
- GPU timings must call `torch.cuda.synchronize()` before stopping the clock (use `gpu_engine.timed`).
- Include host↔device copies in end-to-end GPU numbers.
- **Always state the hardware.** Every saved row records CPU, core count, BLAS library and numpy version.
- CPU and GPU numbers being compared must come from the **same machine**.

## Output

One CSV per scenario in `results/` (`cpu_clean.csv`, `cpu_scaled.csv`, `cpu_serial.csv`,
`cpu_parallel.csv`). Rows are appended, so re-runs build up a history; each has a timestamp.

## Results so far

MacBook Air (Apple Silicon, 10 cores, Accelerate BLAS, numpy 2.5.5), 21 Nov. Raw rows in `results/`.

**One clean supercell** (best of 3):

| Supercell | Matrix | Mesh | Solve | Wilson | Total |
|---|---|---|---|---|---|
| 8×8 | 256 | 31×31 (prep comparison) | 13.9 s | 4.6 s | 18.6 s |
| 8×8 | 256 | 6×6 (scaled) | 0.50 s | 0.24 s | 0.75 s |
| 12×12 | 576 | 6×6 | 3.0 s | 1.2 s | 4.3 s |
| 16×16 | 1024 | 6×6 | 16.6 s | 5.8 s | 22.5 s |

**40 disorder configs at W = 2** (all Z2 = 1, none flagged unreliable, serial and parallel identical):

| Supercell | Mesh | Serial | Parallel (10 procs) | Parallel gain |
|---|---|---|---|---|
| 6×6 | 8×8 | 13.5 s (0.34 s/config) | 3.2 s (0.080 s/config) | 4.2× |
| 8×8 | 6×6 | 30.1 s (0.75 s/config) | 7.1 s (0.18 s/config) | 4.2× |

Solve is ~70% and Wilson ~30% of the time at every size, matching the prep profile.
Parallel gains only ~4× on 10 cores, probably because this laptop mixes fast and slow
cores and the workers share memory bandwidth (not yet checked).

**Found while benchmarking:** a coarse mesh can give the wrong Z2 on big supercells (clean
12×12 on 4×4 gave 0). Now caught by `topology.z2_margin` and flagged as unreliable; mesh
minimum raised to 6. See `topology.py`.

## Competitors: what their code actually does

**PythTB 2.0** (read in `pythtb/wfarray.py`, 21 Nov): diagonalisation is batched over all
k-points (one `eigh` on the stack). The Wilson loop is not: `berry_phase` loops in Python
over every line (`for idx in it: ... self._berry_loop(wf_line)`). No disorder ensembles.

**BerryEasy 0.2.1** (Tyner 2024, arXiv:2312.13051; read `src/BerryEasy/BerryEasyGPU.py`
from the PyPI sdist, sha256 `579077cf…4add`, 21 Nov):
- One k-point at a time: `HamBlk` builds H(k) on the CPU with PyBinding
  (`solver.set_wave_vector([kx,ky,kz])`), copies it to the GPU (`cp.array(... .todense())`),
  and calls `cp.linalg.eigh(H)` on that single matrix (lines 25–38).
- Wilson loop as a Python loop of matrix products (`for i in kpts: ft=cp.dot(ft,proj(...))`,
  lines 72–73), one line at a time (`for kk1 in range(ds2+1)`, line 60).
- Uses **projectors in the full orbital space** (N×N, N = all orbitals), each built by a
  Python loop of outer products over bands (`for i in bnds: ft=ft+cp.outer(...)`, lines 43–44),
  instead of overlap matrices in the occupied-band space (n_occ × n_occ, half the size here).
  No SVD.
- The start point of each line is diagonalised twice (lines 67 and 70); final `eigvals`
  runs on the CPU (`np.linalg.eigvals(cp.asnumpy(...))`, line 75).
- No disorder loop in the package: left to the user.

So "BerryEasy issues its GPU work call by call" is now verified from source, not inferred.
It is also more general than us (any PyBinding/Kwant model, nested and spin-resolved
Wilson loops), which is worth saying fairly.

**spinv** (Favata & Marrazzo 2023; read github.com/roberta-favata/spinv at commit
`3a530f9`, May 2023, 21 Nov):
- **One diagonalisation per configuration**, at Γ only (`model.solve_one([0.,0.])`,
  `_pythtb/singlepoint_invariant.py` line 59). Then one small `eigh` of P·Sz·P (n_occ × n_occ,
  line 66) and four small matrix inverses (`common_func.dual_state_spin`). Spin Chern number
  C_s; Z2 = C_s mod 2. Not an integer at finite size: converges as the supercell grows
  (paper uses L = 15 to 42).
- CPU NumPy only, one configuration at a time, no GPU. Disorder is a user loop.
- **Written for PythTB 1.x** (`pythtb.tb_model`, `get_num_orbitals`, `solve_one`) and always
  imports `tbmodels` too, so it won't run against our PythTB 2.0. Needs its own environment.
- Disorder convention `0.5*w*(2*np.random.random()-1.0)` = uniform [−w/2, w/2]: **same as ours**.
  Uses the global `np.random` state (seed with `np.random.seed`).
- Kane-Mele parameters are given in units of λ_SO (`esite = esite*spin_orb`), and orbitals sit
  at [0,0], [1/3,1/3] (ours: [1/3,1/3], [2/3,2/3]). Same physics, different bookkeeping.
- Cost per configuration at L×L: one `eigh` of size 4L². Our pipeline: nk² of them plus Wilson
  loops. At 8×8 with a 6×6 mesh that's 1 vs 36 diagonalisations of 256×256, so **spinv does
  far less work per configuration** at the same size. Its catch: it needs big supercells to be
  accurate, where one diagonalisation is large (L = 42 means 7056×7056).
- **It would batch well too**: many configurations' Γ-point matrices in one GPU `eigh`.

## Open questions (to confirm)

- **k-mesh rule** (`common.scaled_nk`): nk ≈ 48/n, even, minimum 4. Is Z2 stable at these meshes,
  especially near the transition? Check by re-running a few configs at 2× the mesh.
- **Disorder convention**: on-site energies uniform in [−W/2, W/2]. Does this match Orth et al. (2016)?
- **Which supercell sizes and W range** the physics needs for the sweep and finite-size scaling.
