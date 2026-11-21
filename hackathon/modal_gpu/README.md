# Running on a Modal GPU

Written at the hackathon (21 Nov 2026). One command uploads our code, runs it on a Modal cloud
GPU, streams the log to your terminal and downloads the results. It does the same job as
[`../kaggle/`](../kaggle/README.md), with three differences: you pick the GPU (T4 up to H100/B200),
there's no queue, and you pay per second (Kaggle is free).

## One-time setup

1. **uv** installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`). The Modal CLI runs
   through `uvx`, so there's nothing else to install.
2. **Modal account + token.** Go to modal.com > Settings > API tokens, then run
   `uvx modal token set --token-id ak-... --token-secret as-... --profile=<you>`.
   The token is saved in `~/.modal.toml`. **Never put it in the repo, a commit or a notebook.**
3. **Payment method.** Modal only runs GPU functions once the workspace has a card on file
   (Settings > Billing). Without one you get `Please add a payment method to use T4 GPU functions`.
   `--gpu cpu` runs without a card.
4. Check it works (CPU, no card needed, ~1 min): `uvx modal run modal_gpu/app.py --preset check --gpu cpu`

## Run it

From `hackathon/`:

```bash
uvx modal run modal_gpu/app.py                                  # preset "speed" on a T4
uvx modal run modal_gpu/app.py --preset check --gpu H100        # versions, GPU, eigh timing
uvx modal run modal_gpu/app.py --preset sweep_pilot --gpu A100
uvx modal run modal_gpu/app.py --gpu L4 --cmd "scripts/ensemble_sweep.py --L 4 6 --samples 500 --tag l4"
uvx modal run modal_gpu/app.py --cmd="-m pytest -q; scripts/check_env.py"   # several commands
uvx modal run modal_gpu/app.py --list-presets                   # show presets, run nothing
```

| Flag | Default | Meaning |
|---|---|---|
| `--preset` | `speed` | a preset from `PRESETS` in `kaggle/run.py` (same presets as Kaggle), or `check` / `tests` |
| `--cmd` | | run these instead of a preset: `python <cmd>` from `hackathon/`, several separated by `;`. Use `--cmd="..."` when it starts with `-` |
| `--gpu` | `T4` | `T4`, `L4`, `A10G`, `L40S`, `A100`, `A100-80GB`, `H100`, `H200`, `B200`, or `cpu` |
| `--cpu` | `4` | CPU cores reserved (relevant for PythTB's all-cores CPU baseline) |
| `--timeout-h` | `4` | hours before Modal stops the run |

**Pick the GPU with the precision in mind.** T4 and L4 are very slow in float64 (CLAUDE.md);
A100/H100 are much better. Run `--preset check` once on a new GPU type: it prints the float32
vs float64 `eigh` timings.

## Where the results go

- `results/modal/<time>-<preset>-<gpu>/`: every file the jobs wrote to `results/` (CSV, JSON,
  PNG, NPZ), plus `modal-run.log` with the GPU name and every command's output and exit code.
- The same files are on the Modal volume `pythtb-gpu-results` under `runs/<time>-.../`, saved
  while the run goes (every minute). **If you stop a run (Ctrl-C) or it times out**, get what it
  had so far with
  `uvx modal volume get pythtb-gpu-results runs/<folder> results/modal/`
  (`uvx modal volume ls pythtb-gpu-results runs` lists the folders).
- Add numbers to `RESULTS.md` with the GPU (from the log), the command and the folder name.
  CPU baselines on Modal run on Modal's CPUs: never compare them with GPU numbers from another
  machine.

## How it works

`app.py` builds a container (Python 3.12, torch, numpy, matplotlib, ...; built once, then
cached), adds `pythtb/` and `hackathon/` from your laptop (not `.venv`, `results/`,
`joeys_og_folder/`, notebooks), and runs each job as `python <job>` from `hackathon/`. Your
**local, uncommitted** code is what runs, so commit afterwards what produced the numbers.
A failing job is reported and the next one still runs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Please add a payment method to use ... GPU functions` | add a card under Modal > Settings > Billing, or use `--gpu cpu` |
| `Token missing` / `401` | run `uvx modal token set ...` again; check `uvx modal profile current` |
| log says `WARNING: torch can't use this GPU` | the GPU driver is older than torch's CUDA build: `TORCH_INDEX=https://download.pytorch.org/whl/cu128 uvx modal run ...` (rebuilds the image once) |
| a job shows `exit 1` | its output is in the log just above that line |
| a run hit the timeout | raise `--timeout-h` (max 24) and fetch the partial results from the volume |
