# Project Proposal (revised)

## Title
GPU-accelerated disorder-averaged topology of the Kane-Mele model: reproducing the topological Anderson insulator (TAI)

## Scientific problem
Orth et al. (2016, arXiv:1512.03233) show that static Anderson on-site disorder can turn a trivial Kane-Mele insulator into a topological one (the TAI). The transition is found by averaging over many disorder realizations of large systems, which is expensive.

We reproduce the TAI transition using PythTB's Kane-Mele model with Anderson disorder, `H_dis = W * sum_i eps_i c_i^dag c_i`, `eps_i ~ U[-1, 1]`. We then accelerate the calculation on GPUs.

## Scope and assumptions
- Disorder is static Anderson disorder, as in the paper. Thermal fluctuations are out of scope. They could be a stretch goal (a random-onsite or random-bond model).
- We cannot fully reproduce the paper's results. Its observable is the two-terminal conductance G(W, E_F), computed with Kwant on a 93a x 150a sample with metallic leads. PythTB has no leads or scattering-matrix code, so we do not compute G, the 2G_0 edge-state plateau, or the conductance maps (Figs. 2-3).
- What we can do is detect and locate the disorder-induced transition from a trivial to a topological insulator. We compute a topological invariant instead (spin Chern number for lambda_R = 0, via twisted boundary conditions and Wilson loops) at a Fermi level inside the mobility gap. From this we estimate the critical disorder strength Wc as a function of lambda_nu, with statistical error bars from disorder averaging, and compare it with the paper's Fig. 5 and the Born approximation (Eqs. 2-3).
- This is a qualitative-to-semi-quantitative check. It is not a like-for-like quantitative comparison, because our supercells are much smaller than the paper's, so the transition is broadened by finite-size effects, and the invariant and the conductance are different observables. Agreement in the trend and approximate location of Wc is the success criterion, not exact values.
- Parameters follow the paper: lambda_SO = 0.3t, lambda_R = 0, lambda_nu = 1.45t, 1.65t, 1.85t. We first check that the PythTB `kane_mele` parameters map onto the paper's conventions.
- Supercells are smaller (L ~ 6-30) than the paper's ~28k sites. We expect finite-size broadening and will use finite-size scaling.

## Validation targets
1. The clean-limit phase boundary at lambda_nu = 3*sqrt(3)*lambda_SO.
2. For lambda_nu = 1.65t, a trivial phase at W = 0 and a topological phase at W ~ t, averaged over realizations.
3. Wc vs lambda_nu (Fig. 5) compared with the Born approximation (Eqs. 2-3).
4. GPU results compared with the CPU PythTB reference: eigenvalues, Wannier centres and invariants.

## Parallelisation strategy
- Batch over disorder realizations, W values and twist angles (k-mesh) at once, as the independent dimension.
- Batched Hamiltonian construction and eigh on the GPU, using torch (faster than cupy in our T4 tests).
- Batched Wilson loops, using Newton-Schulz polar decomposition in place of SVD, and a pairwise product tree.
- Exploit S_z conservation (lambda_R = 0) by splitting into two spin blocks, which halves the matrix size.
- Precision: FP64 and FP32, with FP32 errors checked against FP64 near the transitions.

## Benchmarks
- Baselines: unmodified PythTB (numpy, single-thread and multi-core, and `use_tensorflow`) and a CPU multiprocessing version over realizations.
- Metrics: time per realization, realizations per hour, speedup against system size, and precision vs accuracy.
- Preliminary results (Kaggle T4, 8x8 supercell, 31x31 mesh): eigh is about 1.9x faster in FP64 and about 10x in FP32. The Wilson loop is about 6.5x in FP64 and about 58x in FP32. The estimated end-to-end speedup is about 2.5-3x in FP64 and about 10x in FP32.

## Scientific value
- More disorder realizations give tighter error bars on Wc.
- Larger systems allow finite-size scaling.
- A broader scan of (lambda_nu, W) is feasible, such as a phase diagram resembling Fig. 4.
- Well-explained negative results are included, for example where FP32 fails or where the T4's weak FP64 limits the gains.

## Stretch goals
- Add lambda_R > 0 (Z2 invariant or Bott index).
- Add a thermal-fluctuation model (random onsite or bond variations).
- Use Chebyshev-expansion projectors (matmul-only) as a diagonalisation-free alternative.

## Risks
- Finite-size effects blur the transition.
- Newton-Schulz convergence near gap closings.
- Hamiltonian construction and host-to-device copies may become the bottleneck.
- The hackathon GPU may differ from the T4, which changes the FP64 trade-offs.
