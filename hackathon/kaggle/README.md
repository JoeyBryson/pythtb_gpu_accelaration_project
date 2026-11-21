# Running on a Kaggle GPU

One command packs our code, runs it on a Kaggle T4 with your account, waits, and downloads
the results to your laptop. Nothing to click in a notebook.

## One-time setup (~5 minutes)

1. **Kaggle account, phone-verified.** Kaggle only allows GPU and internet access to
   verified accounts (Settings > Phone verification).
2. **API token.** Kaggle > Settings > API > *Create New Token*. Save the token string in a
   file only you can read. **Never put it in the repo, a commit, a notebook or chat.**
   ```bash
   mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
   printf '%s' 'PASTE_TOKEN_HERE' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```
   (If Kaggle gave you a `kaggle.json` file instead, put it at `~/.kaggle/kaggle.json`, `chmod 600`.)
3. **uv** installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`). The script runs the
   Kaggle CLI through `uvx`, so there's nothing else to install.
4. Check it works: `uvx --from 'kaggle>=1.8' --with six kaggle kernels list --mine`

## Run it

```bash
git pull
cd hackathon
KAGGLE_USER=<your-kaggle-username> bash kaggle/gpu_run.sh
```

What happens:
1. Packs the repo's code (not `.venv`, not `results/`, not `joeys_og_folder/`).
2. Uploads it as a **private** dataset `<you>/pythtb-gpu-hackathon-code` (created on the
   first run, new version after that). Only your account can see it.
3. Starts the private script `<you>/pythtb-gpu-run` on a T4 and waits for it to finish.
4. Downloads the log and result CSVs to `results/kaggle/<date-time>/`.

**What it runs** is the `JOBS` list at the top of `kaggle/run.py`. Right now: the test suite,
then the final speed benchmark (our GPU code in single precision vs PythTB on all CPU cores,
8×8 and 12×12 with 40 configurations, 16×16 with 16). Expect **~15–20 minutes** once it
starts, almost all of it PythTB's baseline.

## While it runs

- **Watch the live log** at `https://www.kaggle.com/code/<you>/pythtb-gpu-run`. Lines look like
  `[19:41:00 +129s] PythTB parallel 12x12: 4/40 configs, 34s elapsed, ~304s left`.
- **Stuck in "Queued" for more than ~5 minutes?** Kaggle's free GPUs are shared. Re-submitting
  usually gets it going: run the command again, then **cancel the older queued version** on
  the Kaggle page (otherwise it may run later and use quota on a duplicate).
- **Cancel** a run on the Kaggle page (the CLI can't). The script notices, downloads whatever
  there is, and stops.
- GPU quota is ~30 h/week per account; these runs use minutes.

## Afterwards

- Results are in `results/kaggle/<date-time>/out/*.csv` plus `pythtb-gpu-run.log`.
- Add them to `RESULTS.md` with the machine, the command (the JOBS line) and the folder name,
  following the existing entries. Never compare numbers from different machines.
- Logs (`*.log`) are git-ignored; force-add a log only if `RESULTS.md` cites it
  (`git add -f results/kaggle/<date-time>/pythtb-gpu-run.log`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `Unauthorized` | token missing or wrong: check `~/.kaggle/access_token` |
| run shows `ERROR` | open the log on the Kaggle page; the failing command is printed with `exit N` |
| GPU not used / `cuda` skipped | account not phone-verified, or the run fell back to CPU |
| `datasets status` never says `ready` | Kaggle is slow to process uploads sometimes; wait, or Ctrl-C and re-run |
