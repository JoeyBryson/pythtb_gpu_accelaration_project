# CLAUDE.md

Context for Claude Code on our SoTA Scientific Parallelism Hackathon project (21–22 Nov 2026, London).

## The project in one line

GPU-accelerate PythTB's topology pipeline so we can simulate thousands of disordered (imperfect) materials,
then use it to find the disorder-driven topological phase transition in the Kane-Mele model and check it
against published results.

**Pitch for non-physicists:** real materials are never perfect, and finding out whether a flawed material keeps
its useful electronic properties means simulating thousands of imperfect versions. We made that go from hours/days
to minutes on a GPU.

## Workstreams

- **GPU engine**: diagonalisation + Wilson loop on GPU, batching, memory/chunking, precision, benchmarking and
  speed-up numbers, speed-race visual
- **Models and analysis**: disordered model construction, test harness, averaging + error bars, phase diagram
  plot, finite size scaling
- **Physics validation**: paper parameters and expected results, "topological or not" check + sanity tests,
  validation against the papers, science framing

Don't invent physics claims in code comments or docs; flag anything uncertain as "to check" so the team can
confirm it.

## Repo

- Fork of PythTB 2.0: https://github.com/JoeyBryson/pythtb_gpu_accelaration_project (branch `main`)
- PythTB 2.0 requires Python >= 3.12
- Papers live in `GPU_experimentation/papers/`
- **Keep our new code separate from PythTB internals.** The GPU engine is a module that sits alongside PythTB
  (e.g. `gpu_engine.py`), not a rewrite of the library. Only touch PythTB's own files if unavoidable.
- Work done before the event (prototypes, test notebooks) must be clearly labelled (e.g. a `prep/` folder).
  SoTA allows pre-written code but judges need to see what was done on the day.

## The physics, minimally

- **Model:** Kane-Mele (graphene + spin-orbit coupling), `pythtb.models.kane_mele(esite, t, soc, rashba)`.
  Tutorial parameters: `kane_mele(1.0, 1.0, 0.3, 0.25)` (topological phase). Spinful.
  Tutorial: `docs/source/tutorials/kane_mele.ipynb`
- **Disorder:** Anderson disorder = random on-site energy per site, drawn uniformly from a range set by the
  disorder strength W. Needs supercells (`model.make_supercell([[n,0],[0,n]])`).
- **Pipeline (two stages):**
  1. *Solve*: build H(k) at every k-point and diagonalise (`WFArray.solve_model`, ~70% of CPU time)
  2. *Wilson loop*: along each k-line, overlap matrices between neighbouring k-points -> unitary part
     (PythTB uses SVD: `V @ Wh`) -> multiply along the line -> eigenvalue phases = Wannier centres
     (`WFArray.berry_phase` / `wilson_loop` / `_wilson_loop`, ~30%)
- **Topological or not:** for Kane-Mele it's a Z2 invariant, read from whether the Wannier centres wind across
  the cell. The exact algorithm for deciding this robustly is still to be agreed.
- **Validation targets:**
  - Orth, Sekera, Bruder, Schmidt (2016), *The topological Anderson insulator phase in the Kane-Mele model*,
    https://arxiv.org/abs/1512.03233. Uses **conductance via Kwant** (edges), not Wilson loops.
    We compare *where* the topological region is (bulk-boundary correspondence), not the method.
  - https://arxiv.org/abs/2301.02612: Z2 invariant for disordered Kane-Mele, including the TAI transition
- **The physics isn't new; our method and speed are.** As far as we can tell, batching many disorder
  configurations on GPU with a Newton-Schulz Wilson loop is new. Say "as far as we can tell", never "first".

## Key findings so far (Tesla T4, Kaggle; Kane-Mele supercells, 31x31 k-mesh)

CPU profile (PythTB 2.0):

| supercell | matrix | solve s | wilson s |
|---|---|---|---|
| 4x4 | 64 | 1.76 | 0.61 |
| 6x6 | 144 | 9.27 | 4.13 |
| 8x8 | 256 | 31.5 | 13.5 |

- PythTB's existing `use_tensorflow=True` (diagonalisation only) is slower than CPU until ~6x6, ~1.5x at 8x8.
  `berry_flux` also has a TF option. **The main Wilson loop (`berry_phase`/`wilson_loop`) has no GPU path.**
- Diagonalisation at 8x8 (961 matrices of 256x256): NumPy 24.6 s; CuPy c128 44.5 s (slower!), c64 4.5 s;
  **PyTorch c128 13.3 s (~1.9x), c64 2.5 s (~10x)**. CuPy's batched eigh falls off for bigger matrices; use torch.
- Wilson loop at 8x8 (batched over all k-lines): CPU 8.7 s; **GPU SVD 15.2 s (slower than CPU!)**;
  **GPU Newton-Schulz 1.33 s (~6.5x) in c128, 0.15 s (~58x) in c64**.
- Rough end-to-end estimate at 8x8 (not yet measured as one run): ~2.5–3x in double, ~10x in single.
- Precision: single-precision eigenvalue error ~4e-4 at 8x8; Wannier centres in single precision differ by
  ~1e-7 (checked up to 4x4). Probably fine, but needs confirming on the physics side.
- T4s are very slow at float64 (~1/32 of float32). Data-centre GPUs (A100/H100) are much better. Re-benchmark
  on whatever GPUs the event provides before deciding precision strategy.

## Technical decisions

- **Batch everything.** No Python loops over k-points, k-lines or disorder configurations. Stack into arrays
  with leading batch dimensions; `@`, `eigh`, etc. all accept stacks. A Python loop around GPU calls is often
  slower than the CPU.
- **Diagonalisation:** `torch.linalg.eigh` on a batched stack. Keep data on the GPU between steps.
- **Unitary part of overlaps:** Newton-Schulz instead of SVD:
  `X <- 0.5 * X @ (3I - X^H X)`, ~8 iterations. Converges fast because overlap singular values are ~0.99+.
  Validated: Wannier centres match PythTB's SVD version to ~1e-15 (CPU, double).
  **Caveat:** near gap closings / coarse meshes singular values drop and NS converges slowly. Add a convergence
  check (`||X^H X - I||`) and fall back to SVD if it doesn't converge. Phase transitions are exactly where
  this can happen, so don't skip it.
- **Product along each line:** tree reduction (pairwise products, log2(K) rounds) instead of sequential.
  Matmul is associative so order is preserved; handle odd lengths by carrying the last matrix.
- **Precision:** support both `complex128` and `complex64` via a flag. Validate on final physical outputs
  (Wannier centres / invariant), not intermediate matrices.
- **Memory:** estimate `elements x bytes` (16 B for complex128) with 2–4x workspace. Chunk batches to fit
  (T4 = 16 GB). Free CuPy's pool with `cp.get_default_memory_pool().free_all_blocks()` when needed.
- **Bigger supercells need fewer k-points** (the supercell Brillouin zone is smaller). Scale the k-mesh down
  with supercell size; the right mesh per size is still to be confirmed.
- **Device-agnostic code where it helps:** `xp = cp.get_array_module(x)` (CuPy) or a torch/numpy switch, so the
  same function runs on CPU for testing.

## The Wilson loop details that are easy to get wrong

- PythTB's `_wilson_loop` reshapes spinful wavefunctions: `(..., norb, 2)` -> flattened orbitals. Use
  `flatten_spin_axis=True` when pulling states.
- Closing the loop correctly at the Brillouin zone boundary needs the right phase (PythTB handles this in
  `wilson_loop` before calling `_wilson_loop`). When reimplementing, reproduce PythTB's handling exactly and
  test against it; the simple "open line" version used in early benchmarks skips this.
- Eigenvectors and SVD factors are only unique up to phase/sign. Compare unique quantities (eigenvalues,
  `V @ Wh`, projectors, Wannier centres), never raw eigenvectors.
- Wannier centres are periodic in [0, 1): compare with `d = min(|a-b|, 1-|a-b|)`.

## Testing rules

- Always keep the slow, obviously-correct reference (normal PythTB on CPU) and compare every optimised version
  against it on the same inputs.
- Tolerances: ~1e-8 for complex128, ~1e-4 to 1e-5 for complex64. Write the tolerance down with the test.
- Sanity checks for disorder: weak disorder should stay topological, very strong disorder should become trivial.

## Benchmarking rules

- Use `cupyx.profiler.benchmark` or explicit `torch.cuda.synchronize()` / `cp.cuda.Device().synchronize()`
  before stopping timers. GPU calls are async; unsynced timings are wrong.
- Warm up first (first calls include compilation/allocation).
- Compare against the **best** CPU version (batched NumPy, as PythTB already batches eigh), not a naive loop.
- Include host-to-device copies in end-to-end numbers. Always state the hardware.

## Milestones

1. **Clean model on GPU** (Sat morning): Kane-Mele end-to-end on GPU. Done when Wannier centres match PythTB
   and we have a measured speed-up.
2. **One disordered system** (Sat afternoon): random on-site energies in a supercell -> topological / not answer.
   Done when weak disorder stays topological and strong disorder kills it.
3. **Batched disorder** (Sat evening): many configs at once on GPU. Done when averaged results are stable and
   CPU vs GPU speed-up is measured.
4. **Full sweep + validation** (Sun morning): sweep disorder strength, find the transition, compare with the
   papers. Done when we have the phase diagram plot and CPU vs GPU time for it.
5. **Pitch** (Sun afternoon): code freeze ~15:00. Live demo with recorded backup.

Stretch: 2–3 system sizes showing the transition sharpening (finite size scaling), benchmark vs BerryEasy,
Hamiltonian construction on GPU.

## Prior art to be aware of (the "why ours?" question)

- **BerryEasy**: CuPy GPU package for nested/spin-resolved Wilson loops, built on PyBinding (not PythTB),
  aimed at disordered supercells. Closest competitor; benchmark against it if time allows.
- **TBPLaS 2.0**: tight-binding package with GPU support, Berry curvature and Chern numbers.
- **WannierBerri**: fast Berry curvature for real materials; CPU multiprocessing in versions we checked.
- **Newton-Schulz** is standard in ML (Muon optimiser, Polar Express); using it in Wilson loops appears uncommon.

## Style

- Code will be read by physicists: clear names, short docstrings stating array shapes, e.g.
  `# u: (lines, points, bands, orbitals) complex`.
- Small, testable functions. Commit and push working versions often.
- Ask before large refactors of PythTB itself.
