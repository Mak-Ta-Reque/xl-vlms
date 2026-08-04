# Project instructions

## Disk: always use /media/NVME_8TB/abka03/, never /home/abka03/

The root filesystem (`/`, which `/home` lives on) is at **97% used, ~56GB free**.
`/media/NVME_8TB` has **5TB free**. `/home/abka03` itself is small (~7GB) and
mostly other tools' working directories (`.vscode-server`, `.claude`, `.npm`) —
not something to grow further.

**Every file this project creates or downloads must live under
`/media/NVME_8TB/abka03/`** — datasets, model weights, caches, outputs, scratch
files, conda environments and package caches. Never let a script default to
writing under `/home/abka03/`.

Already correctly configured, don't change without a specific reason:
- `HF_HOME=/media/NVME_8TB/abka03/hfcache` (`.env`) — HF model/dataset cache.
- `conda config --show envs_dirs` lists `/media/NVME_8TB/abka03/conda` first —
  `conda create -n <name>` (no `-p`) lands there.
- `conda config --show pkgs_dirs` → `/media/NVME_8TB/abka03/conda_pkgs`.
- `NLTK_DATA`, `PIP_CACHE_DIR`, `TORCH_HOME`, `MPLCONFIGDIR` (`.env`) — all
  redirected to `/media/NVME_8TB/abka03/{nltk_data,pip_cache,torch_home,mpl_config}`.
- `outputs/`, `data/` (this repo) already live under
  `/media/NVME_8TB/abka03/Projects/xl-vlms/`, which is itself on the NVME mount.

Deliberately left alone (fixed, small, one-time footprint — not worth the risk
of relocating): the conda **base** install itself
(`/home/abka03/miniforge3`, ~370MB) and its shell activation hooks. Moving this
would require reinstalling conda and editing shell rc files — out of scope
unless explicitly requested.

When adding new tooling that downloads/caches anything (a new model, a new
Python package with its own cache dir, a new dataset), check where it defaults
to writing and redirect it under `/media/NVME_8TB/abka03/` — via `.env` if the
tool respects an env var, otherwise via an explicit `--cache-dir`/output-path
argument. Don't assume `~` or the default cache location is safe here.

Note: `$CLAUDE_JOB_DIR/tmp` (background-job scratch space, e.g.
`/home/abka03/.claude/jobs/<id>/tmp`) is managed by the Claude Code harness
itself, is auto-cleaned, and is not part of this exclusion — it's fine to use
for temporary scripts/smoke-test drivers as instructed elsewhere in the system
prompt.
