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

### Comparison and profiling scripts

| Script | What it measures | Output |
|---|---|---|
| `team_tools_sweep.py` | The disorder sweep (W × configs → fraction topological) through **every engine on the same seeds**: PythTB serial, PythTB parallel, batched solve + PythTB Wilson (`pyW`), batched pipeline with `wilson.py` Newton-Schulz (`NS`, i.e. `pipeline.run_batch` unchanged), the same with the SVD unitary part (`SVD`). Each in double (`-d`) and single (`-s`); CUDA variants are added automatically on a GPU machine. | `results/team_tools_sweep.csv` |
| `profile_solve_model.py` | Line profile of PythTB's `WFArray.solve_model`; variants of the diagonalisation (eigenvalues only, 1 BLAS thread, SciPy occupied-only, the Hermiticity check); torch CPU `eigh` vs thread count. | `results/profile_solve_model.txt` |
| `profile_wilson.py` | Stage times of `wilson.py` vs PythTB's `berry_phase` on the same states; Newton-Schulz residual per step; what stopping early saves; torch thread count. | `results/profile_wilson.txt` |
| `tf_metal_vs_numpy.py` | PythTB's own `use_tensorflow=True` path on a Mac with Apple's Metal plugin: which ops Metal can run, `solve_model` NumPy vs TF, Metal float32 matmul, TF CPU thread scaling. | `results/tf_metal_vs_numpy.txt` |
| `../scripts/sample_cpu_sim.py` | First end-to-end sanity simulation with plain PythTB (clean Z2 checks, stage timings, small disorder sweep); `--tensorflow` swaps in TF `eigh`. **Uses Orth et al.'s disorder convention [−W, W]**, unlike the rest of our code. | `results/sample_cpu_*` |

`results/benchmark_comparison.html` is a one-page summary (charts + tables) of everything above.
Results and their analysis: [`../RESULTS.md`](../RESULTS.md), sections R7 onwards.

**Which number do we quote?** The GPU speed-up for the disorder sweep should be quoted
against `parallel` (honest) and may also be shown against `serial` (what users do
today). Always say which.

## How a comparison is made (detail)

These are the rules the scripts above follow. New benchmarks should follow them too.

**Same materials in every engine.** Seeds come from `cpu_baseline.config_seeds` (per W:
`config_seeds(seed + 1000*W, configs)`), and `batched.disorder_shifts` draws exactly the same
random numbers as `reference.disordered_kane_mele`, so every engine diagonalises identical
Hamiltonians. Every timing is reported next to a config-by-config check against PythTB serial:
Z2 identical, band gap within 1e-8 (double) or 1e-3 (single). A row that prints `NO` must not
be quoted.

**What is inside the clock.** Wall time (`time.perf_counter`) around the whole engine call:
building H, host↔device copies, solve, Wilson loop and Z2. On a GPU, `sync()` before reading the
clock. Outside the clock: process-pool start-up and worker warm-up, one warm-up call per engine
(imports, kernel loading, first allocation), and every reliability diagnostic.

**Diagnostics are never timed.** `reference.run_config(check=True)` (the default) runs an SVD over
every link to get `min_overlap`; that adds **~20% of wall time** (measured at 3×3, 6×6 and 8×8).
Timed PythTB runs therefore use `check=False`, and the `z2_reliable` flags come from a separate,
untimed `check=True` pass over the same seeds. Batched engines are timed with
`run_batch(check=False)`. ⚠ `cpu_baseline.py serial/parallel` still time `run_config` with the
default `check=True`, so the *wall times* in `cpu_serial.csv` / `cpu_parallel.csv` (R3) are ~20%
too slow (their per-stage sums are fine). Fix: pass `check=False` in `bench_serial` and
`_run_config_task`, and get the flags from an untimed pass.

**Repeats.** Sweeps: one timed pass per W, 9 values of W × 6 configs = 54 configs per supercell, so
noise averages out across W (per-W times varied by under 5%). Profiling scripts: best of 3 after
one warm-up call.

**Threads decide who wins on a CPU, so record them.**
- PythTB parallel: one process per core, 1 BLAS thread each (environment variables set before
  the workers start, because Accelerate reads them only at start-up).
- NumPy's batched `eigh` uses one core however many BLAS threads it is given (R7).
- PyTorch's CPU `eigh`, batched matmul (Newton-Schulz) and SVD take the same time with 8 threads
  or 1 on these batch shapes (R7, R10). So on a CPU our batched engine runs on one core.
- TensorFlow's CPU `eigh` does spread a batch over cores (R8).

**One benchmark at a time** on an otherwise idle machine: the sweep's parallel step uses every core.

**Mac specifics.** No CUDA; PyTorch MPS has no complex `eigh`; TensorFlow-Metal runs only float32
matmul on the GPU (R8). Numbers from a Mac are CPU numbers. TensorFlow needs a throwaway
environment pinned to `tensorflow==2.18.1` + `tensorflow-metal` 1.2.0 (newer TensorFlow fails to
load the plugin), which also pulls NumPy 2.0.2: compare TF only against NumPy timed in the same
environment, as `tf_metal_vs_numpy.py` does.

**Conventions that differ between scripts.**
- Disorder: our code draws from [−W/2, W/2]; `sample_cpu_sim.py` and Orth et al. use [−W, W]
  (W_ours = 2 × W_Orth, see `VALIDATION.md`).
- Mesh: team scripts use `scaled_nk` on a gamma-centred grid; `sample_cpu_sim.py` uses a fixed nk
  on k ∈ [0, 1). Both contain k = 0 and k = π, which Z2 needs.
- PythTB's TF path returns `energies` with a complex dtype; take `np.real` before comparing.

## Rules (from CLAUDE.md)

- Compare against PythTB 2.0's own vectorised code, not a hand-written slow loop.
- Warm up before timing (the first call pays import and allocation costs).
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

**Found while benchmarking:**
- `run_config(check=True)` inside the timed region inflates PythTB wall times by ~20% (see
  "How a comparison is made" above). Affects R3's wall times.
- On CPU, PyTorch ops on our batches are effectively single-threaded, so the batched engine
  loses to PythTB on all cores (R9). The batched design only pays off on a GPU.
- Newton-Schulz converges in 3–4 steps (double) / 2 steps (single) on our overlaps, but
  `wilson.py` always runs 12 (R11).

Earlier: a coarse mesh can give the wrong Z2 on big supercells (clean
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
