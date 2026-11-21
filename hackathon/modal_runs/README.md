# Running sweeps on many Modal GPUs

Separate from `kaggle/`, which is unchanged. The science code is the same: each point runs
`analysis/sweep.py::run_point` with the spin-up Chern number C↑ (`gpu_engine/spin_chern.py`),
so the CSV rows match Kaggle and laptop runs. Points are independent, so Modal runs them on many
GPUs at once and bills per second.

## One-time setup (about 5 min)

1. **Sign up:** <https://modal.com/signup> (GitHub login works). Add a card under Settings → Billing
   if you'll go past the free credit, and **set a spending limit** there before big runs.
2. **Client:** `pipx install modal` (already done on Joey's machine).
3. **Log in this machine:** `modal setup`. This opens the browser; approve it. It stores a token in
   `~/.modal.toml`. **Never commit that file.**
4. **Check:** `modal profile current` prints your workspace.

## Run (from `hackathon/`)

| Command | What | Rough cost |
|---|---|---|
| `modal run modal_runs/app.py --preset test` | C↑ tests against the CPU version, plus timing at L = 8/12/16, on one A100 | < $1 |
| `modal run modal_runs/app.py --preset timing` | Seconds per sample at L = 8, 12, 16, 24, 32 (W = 2, 50 samples) | < $1 |
| `modal run modal_runs/app.py --preset pilot` | λν = 1.65, W 0..9, 100 samples, L = 8 and 12: both transitions | ≈ $1 |
| `modal run modal_runs/app.py --preset wc1 --max-gpus 10` | **Core goal.** Finite-size scaling of Wc₁: L = 8, 12, 16, 24, W 1..3, 1000 samples | ≈ $13 |
| `modal run modal_runs/app.py --preset wc2 --max-gpus 10` | The same for Wc₂ (W 6..8) | ≈ $13 |
| `modal run modal_runs/app.py --preset fig5 --max-gpus 10` | Wc₁ at 7 more λν, L = 8/12/16, 500 samples: an equivalent of Orth Fig. 5 | ≈ $7 |
| `modal run modal_runs/app.py --preset big --max-gpus 10` | L = 32 at both transitions, 300 samples | ≈ $25 |

Costs are **estimates** until `timing` has run. Update `EST_S` in `app.py` from its output.

- **Dry run first:** add `--dry-run` to list the points and print the cost estimate without
  starting any GPU.
- **GPU choice:** `--small-gpu L4` is used for L ≤ 12 and `--big-gpu A100` for larger sizes
  (try `H100` for L ≥ 24).
- **Concurrency:** `--max-gpus N`. Your plan caps it (Starter: about 10 GPUs, to check in the
  dashboard).
- **Results:** `results/modal/<preset>.csv`, appended as each point finishes.
- **Resume:** run the same command again and finished points are skipped.
- **Stop:** Ctrl-C stops the local side. Check the Modal dashboard that the containers have
  stopped too (`modal app stop pythtb-tai-sweeps`).
- **Code changes:** the image is rebuilt automatically when the code changes. The first build
  takes a few minutes (torch download); later runs reuse the cache.

## Not here yet

- **CPU baselines on the same hardware** (PythTB as-is, and C↑ in NumPy on all cores). Run them
  on a Modal CPU container, or on the GPU host, before quoting speed-ups.
- Merging `results/modal/*.csv` with the analysis (crossings, collapse, figures).
