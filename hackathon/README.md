# Hackathon workspace

Everything in this folder was written **at the SoTA Scientific Parallelism Hackathon (21–22 Nov 2026)**,
except `prep/`, which holds work done before the event. PythTB itself lives in the repo root and is
installed from there (editable), so we don't modify it.

Project context, physics, decisions and milestones: [`../CLAUDE.md`](../CLAUDE.md).
Competition brief and judging criteria: [`../GPU_experimentation/competition_requirements.md`](../GPU_experimentation/competition_requirements.md).

## Setup (2 minutes)

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/JoeyBryson/pythtb_gpu_accelaration_project.git
cd pythtb_gpu_accelaration_project/hackathon
uv sync                                 # Python 3.12 venv in .venv, installs PythTB from ..
uv run pytest -q                        # should pass (GPU test skips on laptops)
uv run python scripts/check_env.py      # versions, GPU, quick eigh timing
```

**On a GPU machine**, the same thing in one go: `bash hackathon/scripts/setup_gpu_box.sh`.
Optional CuPy (CUDA 12 only): `uv sync --extra cupy`.

Jupyter: `uv run jupyter lab`, or in VS Code pick the kernel at `hackathon/.venv`.
Adding a package: `uv add <pkg>` (updates `pyproject.toml` + `uv.lock`; commit both).

Laptops (incl. Apple Silicon) run on CPU. We skip Apple's MPS backend because it has no
complex128/complex `eigh`. That's fine for correctness work, but **benchmark numbers only come from the GPU box**.

## Layout

| Path | What | Workstream |
|---|---|---|
| `gpu_engine/backend.py` | device + precision (`"double"`/`"single"`), GPU-synced `timed()` | GPU engine |
| `gpu_engine/reference.py` | slow, obviously-correct PythTB CPU baseline; compare everything to it | all |
| `gpu_engine/` (new modules) | batched `eigh`, Newton-Schulz Wilson loop, chunking | GPU engine |
| `analysis/` | disordered models, averaging, phase diagram, finite-size scaling | Models and analysis |
| `benchmarks/` | timing scripts (state the hardware in every output) | GPU engine |
| `notebooks/` | exploration + plots for the pitch | anyone |
| `results/` | small CSV/PNG outputs (large arrays are git-ignored) | anyone |
| `tests/` | pytest; compare to `reference.py` with tolerances written down | all |
| `kaggle/`, `modal_gpu/` | run on a cloud GPU from your laptop: free Kaggle T4, or any Modal GPU (`uvx modal run modal_gpu/app.py --gpu A100`) | GPU engine |
| `prep/` | pre-event work (also see `../GPU_experimentation/test_scripts/`) | — |

## Quick example

```python
from gpu_engine import get_backend, timed
from gpu_engine.reference import clean_kane_mele, solve, wannier_centres

model = clean_kane_mele(supercell=2)           # tutorial params, topological phase
with timed("cpu solve"):
    wfa = solve(model, nk=31)
wc = wannier_centres(wfa, n_occ=model.norb)    # half filling: spinful, so 2*norb states, norb occupied
be = get_backend("single")                     # CUDA if present, else CPU
print(be.describe())
```

## Team rules (short version of CLAUDE.md)

- Batch everything; no Python loops over k-points / lines / disorder configs.
- Every optimised function gets a test against `reference.py`. Tolerances: 1e-8 (double), 1e-4 (single).
- Compare unique quantities only (eigenvalues, projectors, Wannier centres), never raw eigenvectors.
- Time with `timed(...)` (it calls `torch.cuda.synchronize()`), warm up first, include host↔device copies.
- Small commits, pull often: `git pull` before you start, push working versions.
