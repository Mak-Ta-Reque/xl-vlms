# XL-VLMS Production-Readiness Improvement Plan

Audited 2026-07-09 on branch `logitlens`. The canonical entry point is
**`scripts/run_full_pipeline.sh`** — this plan is organized around that pipeline's
critical path first, everything else second. Items marked **[FIXED]** were resolved
together with this document; the rest is prioritized future work.

## The critical path

`scripts/run_full_pipeline.sh` orchestrates, in order:

| Step | Sub-script | Python entry point |
|------|-----------|--------------------|
| 1. Dataset inference → concept map | `run_dataset_inference.sh` | `inference/dataset_inference.py`, `concept_image_mapping.py` |
| 2. Crop images | `run_crop_images.sh` | `preprocessing/random_crops.py` |
| 3. Generate concept features | `run_feature_gen_cgdl.sh` | `src/save_features.py` |
| 4. Decompose features | `run_feature_decompose_cgdl.sh` | `src/analyse_features.py`, `src/combine_concepts.py` |
| 5. VLM explainer | `run_vlm_explainer_no_gt.sh` | `inference/vlm_explainer.py` |
| 6. Concept deletion eval | `run_concept_deletion_eval_token.sh` | `eval/concept_deletion_eval.py` |
| 7–9. Plots | — | `scripts/plot_concept_deletion_eval_token.py`, `scripts/plot_eval_summary_across_methods.py` |

Anything not in this table (`api/`, `demo/`, `glimpse/`, `inference/vlm_explainer_multibatch.py`,
the root-level `clip_bert_*` scripts, backup files) is off the critical path and handled in §4–§6.

---

## §1 — P0: Bugs and hazards ON the critical path

### 1.1 The orchestrator itself (`scripts/run_full_pipeline.sh`)

- **Machine-specific defaults from a previous machine** — all broken on this box unless
  `.env` overrides every one of them:
  - `INPUT_DIR=/mnt/sdz/abka03_data/data/imagenet_5_class/train` (line 68)
  - `OUTPUT_DIR=/mnt/sdz/abka03_data/outputs/imagenet_5_gpu1` (line 69)
  - `HF_HOME=/mnt/sda/abka03-data/hf_cache` (line 71)
  - `CROP_INPUT_ROOT=/mnt/abka03/xlvlm_data/imagenet_1000` (line 81)
  - `IMAGE_ROOT=/mnt/abka03/xlvlm_data/imagenet10class/val_grids` (line 91)
  - `DEVICE=cuda:1` (line 76) — assumes a second GPU exists.
  Fix: default to repo-relative paths (`$ROOT_DIR/data`, `$ROOT_DIR/outputs/run_$TIMESTAMP`
  as the usage text already claims) and fail fast with a clear message when a required
  path doesn't exist.
- **`eval $@` in `run_step()` (line 46)** — word-splitting/quoting hazard; any argument
  with spaces or shell metacharacters (e.g. the default `PROMPT`) is re-parsed by the
  shell. Replace with `"$@"` execution or `bash -c "$1"` with a single quoted string.
- **Stale-output skip logic**: every step is skipped if *any* matching file exists
  (e.g. step 3: "found features under `$FEATURES_DIR/features`" on the first `*.pth`).
  A partial/failed previous run silently poisons all downstream steps. Fix: write a
  `.done` sentinel per step (or check counts), and add a `--force` flag.
- **Drift markers**: line 248 carries a commented-out alternate concept filename
  (`redefine_activations_text_grounding_..._gl_regrounded.pth`) that other code
  (`api/main.py:79`) still expects — the pipeline's output naming and its consumers
  disagree. Standardize the artifact naming in one place.
- **Three competing orchestrators exist**: `scripts/run_full_pipeline.sh` (canonical),
  `scripts/run_full_pipeline_without_coroping.sh` (documented in README, name typo),
  and `scripts/run_full_pipeline.py` (Python port of the *without_coroping* variant).
  They will drift. Keep the canonical one, delete or clearly deprecate the other two,
  and update the README (§7).

### 1.2 Step 3 — `src/save_features.py`

- **[FIXED]** `NameError` on the default path: `toi_str` (line 654) was only assigned when
  `token_of_interest` was set with `prompt_template` in `cgdl`/`yn`; now always defined.
- **[FIXED]** Generation-only postprocessing (slicing `out[b, input_lens[b]:]` +
  `batch_decode`) ran even in teacher-forcing mode where `out` is logits → garbage
  predictions. Now gated on `args.generation_mode`.
- **[FIXED]** The `json_crop_map` branch overwrote the user's `--save_filename` with a
  hardcoded `qwen2_..._train_{key}` name; now derives from the user-provided base.
- **Open:** unused `num_image_tokens_per_sample` block (lines ~590-597) can
  `AttributeError` on processors without `image_token_id` — delete it.
- **Open:** behavior-changing env vars with unguarded `int()` casts and no `--help`
  surface: `MASK_BLUR_RADIUS`, `INPAINTING_METHOD`, `CROP_MODE`, `PATCH_SIZE`,
  `MASK_CONTEXT_PIXELS`, plus typo fallback `POSITIVE_NEGATVE_SEGMENT` (lines 155-294).
  Promote to validated CLI args; the orchestrator already passes most of these values.
- **Open:** dead `_debug_save = False` branches (~lines 208, 446-493, 544-569) — remove.

### 1.3 Step 4 — `src/analyse_features.py`, `src/combine_concepts.py`, decomposition

- **[FIXED]** `analysis/feature_decomposition.py:290` imported `src.helpers.sae_`, which
  fails under the repo's `src`-on-path convention → `decomposition_method="sae"` was
  dead-on-arrival. Now imports `helpers.sae_` like its sibling.
- **Open:** `helpers/sae.py` vs `helpers/sae_.py` differ only in comments + a
  `sys.path.insert` hack, yet both are load-bearing (`sae`→`sae_`, `sae2`→`sae`).
  Unify into one module.
- **Open:** `src/combine_features.py` writes `combined_features.pth` into the same
  directory it globs (re-runs double-count) and assumes identical keys across `.pth`
  files. Verify whether `run_feature_decompose_cgdl.sh` hits this; fix regardless.

### 1.4 Shared plumbing — `src/helpers/`

- **[FIXED]** `arguments.py`: `--captioning_metrics` used `type=List[str]`
  (`TypeError` the moment the flag is passed) → `nargs="+"`; `--save_analysis` used
  `type=bool` (could never be turned off) → new `str2bool` converter.
- **Open:** `setup_hooks` (`helpers/utils.py:936-950`) silently attaches no hook when
  `--hook_names`/`--modules_to_hook` lengths disagree — the run "succeeds" with no
  features. Make it a hard error.
- **Open:** `extract_token_of_interest_states_all` (`helpers/utils.py:455-463`) mixes a
  list index with a tensor-valued slice stop (invalid indexing) and returns the opposite
  mask convention from its sibling at line 361. Needs a fix + unit test.
- **Open:** hardcoded magic token `special_ids.add(106)` ("qween" EOS) in shared pooling
  logic (`helpers/utils.py:613`) — derive from the tokenizer.
- **Open:** `models/image_text_model.py:113-114` does `.to(model.device).to(model.dtype)`
  on the whole `BatchEncoding` — breaks with `device_map="auto"` and risks casting
  `input_ids` to float. Cast only floating-point tensors.

### 1.5 Steps 1, 2, 5, 6 — remaining on-path files

- `inference/dataset_inference.py`: contains an HF `login(token=...)` call (line ~262) —
  make sure it only reads `HF_TOKEN` from the environment and never a literal.
- `preprocessing/random_crops.py`, `eval/concept_deletion_eval.py`: the `/mnt/abka03/...`
  paths here are only in docstrings/usage examples (not live defaults) — update the
  examples, low priority. `concept_deletion_eval.py` leans on broad `except Exception:`
  fallbacks (lines 54-85) that can mask real failures — narrow them.
- `inference/vlm_explainer.py` is the on-path explainer; `vlm_explainer_multibatch.py`
  (74% identical, and with a hardcoded `--out_json` default to `/mnt/abka03/...` at line
  758) is an off-path fork — consolidate (§5).

## §2 — P1: Reproducibility (the pipeline can't run from a fresh clone)

- **`requirements.txt` is unusable**: torch, transformers, matplotlib, nltk, spacy,
  bert-score, clip and most core deps are commented out (lines 1-23). Rebuild with
  pinned, mutually consistent versions. Add the missing undeclared deps: `sam3`,
  `lang_sam`, `python-dotenv`, `qwen_vl_utils`, `language_evaluation`, `timm`, `pandas`,
  `pytest`. Note the active env for this project is conda `xlvlms`
  (`/media/NVME_8TB/abka03/conda/xlvlms`) — snapshot it as the starting point.
- **`sam3` undeclared everywhere** (imported 21× in `src/sam3_utils.py`,
  `src/save_features.py`) and not installed by the Dockerfile.
- **Version skew**: Dockerfile base CUDA 12.5.1 vs cu126 wheels vs README cu124/cu126 vs
  commented pin torch 2.8+cu126; `numpy>=1.20,<2` (active) vs `numpy==2.3.2` (commented).
  Pick one story; make README/Dockerfile/requirements agree.
- **No packaging**: add `pyproject.toml`, make `src/` installable (`pip install -e .`),
  delete the per-file `sys.path.insert` hacks.
- **Docker**: `.dockerignore` now excludes `.env` **[FIXED]**; still open — the image
  hand-lists deps instead of using requirements, and its entrypoint/OUTPUT_DIR are
  hardcoded to an ad-hoc run.

## §3 — P2: Logit lens completion (current branch work)

`src/helpers/logit_lens.py` exists but is **not yet wired into the pipeline** — nothing
imports it. Done so far:

- **[FIXED]** duplicated `valid_token_count` clamping block (was pasted twice).
- **[FIXED]** `"mean"` aggregation returned a mean probability with an argmax position;
  branches collapsed and semantics documented (`best_position`/`top_tokens` always refer
  to a single position).
- **[FIXED]** warns when the final norm module can't be resolved instead of silently
  producing garbage logits.
- **[FIXED]** the `save_hidden_states_logit_lens` hook (`helpers/utils.py:741`) was
  byte-identical to the mean hook — it now keeps per-position hidden states, which is
  what logit lens needs.

Remaining: call `score_hidden_state_with_logit_lens` from the analysis stage on the
saved per-position states; expose `position_aggregation`/layer selection as CLI args;
add an end-to-end test with a small checkpoint (`tests/test_logit_lens.py` covers units
only).

## §4 — P2: API/demo hardening (off critical path, local-only today)

Fixed now because they were cheap and severe:

- **[FIXED]** `eval()` on strings from `.pth` files → `ast.literal_eval`
  (`api/pth_to_json_converter.py:148,168`).
- **[FIXED]** path traversal via raw `file.filename` in upload save (`api/main.py:1098`).
- **[FIXED]** `uvicorn reload=True` in the entrypoint (`api/main.py:1245`).
- **[FIXED]** `.env` excluded from Docker images via `.dockerignore`.
- **Done by user:** the live HF token that was in `.env` — rotate on huggingface.co.

Required before any network deployment:

- `torch.load(..., weights_only=False)` on served `.pth` files (`api/main.py:243,620,927`,
  `api/save_data.py:211`, `api/pth_to_json_converter.py:90`, `demo/classify_api.py:160`)
  — unpickling is code execution; migrate to `weights_only=True`-compatible artifacts.
- CORS `allow_origins=["*"]` + `allow_credentials=True`, no auth, binds `0.0.0.0`
  (`api/main.py:63-69,1243`, `demo/classify_api.py:105-107,865`).
- Unbounded in-memory uploads, no size/content-type validation (`api/main.py:669-731,
  1030-1101`).
- Blocking 1-hour `subprocess.run` of the pipeline inside an async handler with an
  unlocked `pipeline_status` global (`api/main.py:1136`) — move to a background task.
- Bare `except:` clauses (`api/main.py:895,903,913,920,1024`).
- `api/main.py:79` defaults to an ad-hoc run artifact (`outputs/screen_run/...`).

## §5 — P3: Consolidation (duplication debt)

- Orchestrators: keep `run_full_pipeline.sh`, remove/deprecate
  `run_full_pipeline_without_coroping.sh` and `scripts/run_full_pipeline.py` (§1.1).
- The five BERT/CLIP scoring scripts (`clip_bert_score_explanation.py` vs `..._json.py`
  are 94% identical; plus `clip_bert_scores.py`, `bert_score_eval_json.py`,
  `eval/clip_bert_score_eval.py`) ≈ 2,700 lines with the same helpers reimplemented
  3-5×. Merge into one `eval/` module.
- `inference/vlm_explainer.py` vs `vlm_explainer_multibatch.py` (74% identical): extract
  shared model-loading/seeding/input-prep code; keep one entry point with a batch flag.
- Unify `helpers/sae.py` / `helpers/sae_.py` (§1.3).

## §6 — P3: Repository cleanup

- Delete dead files: `src/analyse_features_backup.py`, `src/save_features_backup.py`,
  `src/helpers/utils_backup.py`, `src/models/gemma3_backup.py` (shadows the live
  `Gemma3nVL`), `preprocessing/crops_to_json_old.py`,
  `glimpse/chat_gpt_implentaion_of_glimplse.py`, the empty `src/main.py` stub.
- Fix `.gitignore`: `.png`/`.pth`/`.notebooks` entries lack `*` and match nothing;
  duplicate `token.txt`; `temp/` never ignored.
- Untrack clutter: vendored `git-lfs-3.5.1/` (13 MB incl. 11 MB binary), `temp/`,
  root images (`output.png` 8.3 MB, `grid_4_*.jpg`, `n0*.JPEG`), `.docx` research note,
  `## Chat Customization Diagnostics.md`, `plan-*.prompt.md`,
  `api/test_result_custom_prompt.json`, `api/TKG.png`; strip outputs from committed
  notebooks (several 2-6 MB).
- **History rewrite (approved, needs coordination)**: `.git` is ~1.1 GB. After the
  working-tree cleanup lands: `git filter-repo --strip-blobs-bigger-than 1M` (with an
  allowlist for needed assets), force-push, collaborators re-clone.
- `.gitmodules`: `model-kg-visualization` uses an SSH URL to a private repo — switch to
  HTTPS or make optional.

## §7 — Tests, CI, docs

- Add `pytest` + root `conftest.py` (make `src/` importable), drop per-file `sys.path`
  shims; move the GPU benchmark scripts (`tests/test_sam3_batch.py`,
  `tests/test_langsam_batch.py`) to `scripts/benchmarks/`. Note: `pytest` is not
  installed in the `xlvlms` conda env; the pytest-style test files currently collect
  0 tests under `unittest`.
- **Broken test (pre-existing):** `tests/test_crops_to_json_unit.py` imports
  `calculate_iou` from `preprocessing/crops_to_json.py`, which no longer defines it —
  the test drifted from the module. Restore the function or update the test.
- ~50 of ~55 `src/` modules have zero coverage. Priority order = the critical path:
  `helpers/utils.py` (hooks, token extraction), `analysis/feature_decomposition.py`
  (each method on tiny synthetic matrices), `helpers/arguments.py` (round-trip every
  flag), `save_features.py` helpers, `combine_concepts.py`, then model wrappers with
  mocked processors.
- Add a **pipeline smoke test**: run `scripts/run_full_pipeline.sh` end-to-end on the
  committed dummy data (`data/`) with a small model, asserting each step's artifacts
  exist — this is the single highest-value test for the actual entry point.
- CI: GitHub Actions with lint (ruff) + CPU-only unit tests on push/PR.
- README: document `scripts/run_full_pipeline.sh` as THE entry point (it currently
  documents `run_full_pipeline_without_coroping.sh`), the `.env` contract (which
  variables the orchestrator actually reads), and one install story. Refresh
  `api/README.md` (stale cross-machine references).

---

## Suggested execution order (future sessions)

1. **§1.1 orchestrator hardening + §2 reproducibility** — safe defaults, `eval` fix,
   sentinel-based skip logic, rebuilt requirements, packaging. This makes every later
   change verifiable via the pipeline smoke test.
2. **§1 remaining on-path bugs** with unit tests as they're fixed.
3. **§7 pipeline smoke test + CI** to lock it in.
4. **§6 repo cleanup + history rewrite** (one coordinated change).
5. **§5 consolidation** — safe once tests exist.
6. **§3 logit lens wiring** (feature work on this branch) and **§4 API hardening**.
7. Docs last, describing the end state.
