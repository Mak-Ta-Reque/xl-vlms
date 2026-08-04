# XL-VLMS Production-Readiness Improvement Plan

Audited 2026-07-09 on branch `logitlens`. The canonical entry point is
**`scripts/run_full_pipeline.py`** — the Python orchestrator implementing the
JSON-coordinate crop flow (crops stored as bbox/RLE coordinates in
`inference/crops.json`; `save_features.py --dataset_name json_crop_map` crops
on the fly; no crop PNGs are written). This plan is organized around that
pipeline first, everything else second. Items marked **[FIXED]** were resolved
together with this document; the rest is prioritized future work.

## The critical path

`scripts/run_full_pipeline.py` runs, in order (steps 2–7 as subprocesses for
GPU-memory isolation). Renumbered 2026-07-10: layer selection is now its own
step 3, everything after shifts by one (old step N ≥ 3 = new step N+1):

| Step | What | Python entry point |
|------|------|--------------------|
| 1 | Dataset inference → concept map (+ vocab/top-N selection) | `inference/dataset_inference.py`, `concept_image_mapping.py`, `preprocessing/select_top_concepts.py` |
| 2 | Build crops JSON (coordinates, not files) | `preprocessing/crops_to_json.py` |
| 3 | Per-tag layer selection via logit lens (optional, `LOGIT_LENS_LAYER_SELECTION=1`) → `logitlens/selected_layers.json` | `src/select_layers.py` |
| 4 | Generate features from crops JSON (hooks each tag's selected layer when step 3 ran, else `LAYER_PATH`) | `src/save_features.py --dataset_name json_crop_map` |
| 5 | Decompose + combine + reground (per method) | `src/analyse_features.py`, `src/combine_concepts.py` |
| 6 | VLM explainer (per method) | `inference/vlm_explainer_multibatch.py` |
| 7 | Concept deletion/insertion + BERT/CLIP + AUC eval | `eval/concept_deletion_eval.py`, `eval/clip_bert_score_eval.py`, `eval/concept_curve_auc_eval.py` |
| 8 | Plots | `scripts/plot_concept_deletion_eval_token.py`, `plot_eval_summary_across_methods.py` |

**Legacy orchestrators (deprecate):** `scripts/run_full_pipeline.sh` (PNG-crop
flow — hardened and validated end-to-end 2026-07-09, but a different pipeline)
and `scripts/run_full_pipeline_without_coroping.sh` (shell ancestor of the .py).
Both should be removed once the .py orchestrator has full parity and a smoke test.

---

## The method — scientific summary

What the pipeline implements, stated precisely (evidence in the deep-audit
section below):

1. **Concept-candidate discovery.** A VLM captions each dataset image with a
   fine-grained comma-separated object list (step 1); inverted into a
   concept-tag → images map, optionally vocab-filtered and truncated to the
   top-N tags by support (`preprocessing/select_top_concepts.py`).
2. **Concept-region localization.** For each tag, images are segmented
   (GroundingDINO+SAM2 via LangSAM, or SAM3 point-grid auto masks) and each
   region is stored as bbox + RLE mask coordinates in `crops.json` (step 2).
   A detector-free alternative is also available: sliding-window cropping
   (`CROP_MODE=sliding_window`), which tiles each image with square windows
   at a configurable stride (`SLIDING_WINDOW_STRIDE_RATIO`), drops
   near-duplicate windows via CLIP similarity, and stores bbox-only crops
   (no masks) — see `SLIDING_WINDOW_CROPS_GUIDE.md`.
3. **Per-tag layer selection (optional, logit lens).** With
   `LOGIT_LENS_LAYER_SELECTION=1`, a dedicated pipeline step
   (`src/select_layers.py`, step 3, right after crops.json) picks the
   extraction layer per tag: it samples `LOGIT_LENS_NUM_PATCHES` of the
   tag's own crops.json regions (preprocessed exactly like extraction),
   runs them through the VLM, and sweeps the decoder layers
   (`LOGIT_LENS_LAYERS`, default all) with the logit lens. Each layer is
   scored by the **relative probability p(tag token) / p(top-1 token)** —
   `LOGIT_LENS_MODE=patch` scores the visual-token positions of one forward
   pass, `text` scores the answer-producing positions of a generation —
   max over positions, mean over regions; argmax wins. Results land in
   `<output>/logitlens/<tag>/` (per-layer scores JSON + bar plot) and
   `<output>/logitlens/selected_layers.json` (the tag → layer map consumed
   by step 4). Disabled (default), extraction uses the static `LAYER_PATH`.
4. **Representation extraction.** For each (tag, region) sample, the region is
   isolated by inpainting/blurring the background (`INPAINTING_METHOD`,
   `MASK_BLUR_RADIUS`, `MASK_CONTEXT_PIXELS`), and the VLM is prompted with a
   binary probe — *"Classify the image as either [tag] or No [tag]…"* — in
   generation mode. The representation is the **mean over the generated answer
   tokens** of the hidden state at the tag's selected layer (from step 3) or
   at the static `LAYER_PATH` (`model.language_model.norm` by default): one
   D-dim vector per region. The layer used is recorded as `selected_layer` in
   the feature `.pth` (step 4).
5. **Concept dictionary learning.** Per tag, the N_regions × D matrix is
   factored by Semi-NMF (sklearn `DictionaryLearning`, `positive_code=True`,
   K = `DECOMP_COMPONENTS` = 2, α = 20) into K sub-concept directions +
   loadings. Text grounding = logit-lens through the final RMSNorm + real
   `lm_head` (top dictionary-filtered vocabulary tokens per direction; the
   norm is applied only to mid-layer features — see `needs_final_norm`);
   image grounding = the top-activating regions
   (`NUM_MOST_ACTIVATING_SAMPLES`, -1 = all, ordered by activation).
   A **positive/negative split** (see the dedicated section below) routes
   each direction into the positive concept bank
   (`combined_concept_<method>_cr<ratio>_*`) or the negative bank
   (`combined_negative_concept_<method>_cr<ratio>_*`) based on the VLM's
   per-region verdicts and `CLEAN_EXAMPLE_RATIO`. Each tag is decomposed at
   its own recorded layer, and `selected_layer` +
   `direction_positive_ratio` are carried per concept into both banks
   (step 5).
6. **Explanation.** At inference, each generated token's residual-stream vector
   at the same layer is matched to concept directions by cosine similarity;
   per-token top-N concepts with their text/image grounding form the
   explanation (step 6). NOTE: the explainer still uses the single static
   `LAYER_PATH` — with per-tag layers enabled, use each concept's
   `selected_layer` from the bank (open gap, see §3).
7. **Faithfulness evaluation.** Concept-coordinate deletion/insertion curves
   through `norm→lm_head` (target-token probability vs fraction of coordinates
   zeroed/restored), AUC summaries, and BERTScore/CLIPScore of concept
   grounding vs ground-truth words — the latter with a random-concept control
   (step 7-8).

This is a **regional-probing concept-dictionary method**: concepts are learned
from the model's own answer-conditioned representations of isolated visual
evidence, grounded through the model's own vocabulary head, and evaluated by
representational ablation. The planned extension (plan-conceptGraph.prompt.md)
is an **append-only ConceptBank** (SQLite) where new concepts are added by
assign-or-cluster against frozen existing directions instead of re-factoring.

### Positive/negative concept selection — algorithm and limitations

Implemented 2026-07-10 (`compute_direction_positive_ratios` in
`src/analysis/multimodal_grounding.py`, consumed by `src/combine_concepts.py`;
pinned by `tests/test_direction_positive_ratio.py`). Replaces the old purity
filter that computed the positive ratio over each direction's grounding
prediction list — that ratio became meaningless with
`NUM_MOST_ACTIVATING_SAMPLES=-1`, because every direction then carried the
same (mostly negative) full sample set.

**Algorithm.** For one tag with N regions and K = `DECOMP_COMPONENTS`
directions:

1. Semi-NMF gives loadings `A` (N × K, `A ≥ 0`): `A[i, j]` = how much
   direction j explains region i's representation. Rows of `A` are in
   original sample order, aligned with the per-region VLM verdicts
   (`model_predictions`: "Person" vs "No person" from the step-4 probe).
2. **Hard assignment**: `assignment[i] = argmax_j A[i, j]` — each region is
   owned by exactly one direction, partitioning the N regions into K
   disjoint groups `S_j`.
3. **Per-direction ratio**:
   `ratio_j = #{i ∈ S_j : VLM verdict positive} / |S_j|` — e.g. 20 assigned
   regions, all positive → 1.0; 15 of 20 → 0.75. `|S_j| = 0` (a direction no
   sample prefers) or unalignable predictions → `ratio_j = None`.
4. **Split**: `ratio_j ≥ CLEAN_EXAMPLE_RATIO` → positive bank;
   below-threshold or `None` → negative bank (kept, not discarded). The
   threshold is embedded in the bank file names (`…_cr0.5_raw.pth`) and each
   concept stores its `direction_positive_ratio`, so different thresholds
   can be produced and evaluated side by side. Step-5 logs print
   `Direction j: assigned=…, positive_ratio=…` per direction for auditing.

**Why assignment-based**: with sliding-window crops ~97% of a tag's regions
are negatives ("No [tag]"). A ratio over all samples (or any fixed top-k
containing mostly negatives) dilutes the few positives to ~0 and rejects
everything. Argmax assignment scores each direction only on the regions it
actually explains best, so a direction that captures the few true positives
scores high regardless of how many negatives exist globally.

**Limitations.**

1. **Absolute threshold, not relative.** The rule can reject ALL directions
   of a tag: if Semi-NMF lays its directions so the few positives share a
   direction with many visually-similar negatives (e.g. 23 positives + 80
   negatives assigned → 0.22 < 0.5), the tag ends with no positive concept
   even though that direction is clearly the most tag-like one. Remedies, in
   order of strength: raise K (finer partition → purer directions); a
   relative rule (keep the per-tag argmax-ratio direction when above a
   softer floor); or label-split decomposition (factorize the VLM-positive
   regions separately — pure by construction, planned in §3).
2. **Hard assignment ignores margins.** A region loading 0.51/0.49 counts
   fully for one direction; borderline regions add assignment noise.
   *Mitigated 2026-07-10*: the soft activation-weighted ratio
   `weighted_j = Σ_{i pos} A[i,j] / Σ_i A[i,j]` is now always computed and
   stored per concept (`direction_weighted_ratio`, logged next to the
   assigned ratio), and `DIRECTION_RATIO_MODE=assigned|weighted|both`
   selects which metric the split thresholds (default: assigned).
   Real-data example of its diagnostic value: person's winning direction had
   assigned=1.0 (owns exactly the 6 positives) but weighted=0.40 (60% of its
   activation mass comes from weakly-loading negatives), while
   sky/tree/white scored 0.98/0.79/0.94 on both. All-zero activation rows
   (samples the factorization explains with no direction) are now always
   excluded from the assignment instead of argmax-defaulting to direction 0.
3. **The ratio trusts the VLM verdict.** `model_predictions` comes from the
   binary probe on inpainted/blurred regions; probe errors (false "No
   [tag]" on small/occluded regions, or false positives from background
   leakage) propagate directly into the split. The verdict parser
   (`is_negative_prediction`) is prefix-based ("no", "no_", "unk", …) and
   can misread free-form answers.
4. **Small assigned sets are high-variance.** `ratio_j` over |S_j| = 3
   regions moves in steps of 1/3. *Mitigated 2026-07-10*:
   `MIN_DIRECTION_ASSIGNED=<n>` (default 1 = off) requires a minimum number
   of assigned samples before a direction can enter the positive bank;
   `direction_num_assigned` is stored per concept in both banks for
   auditing.
5. **Factorization quality bounds selection quality.** The split only picks
   among the K directions SNMF learned from a heavily negative-dominated
   matrix; it cannot create a clean positive direction that the
   factorization never separated (same root cause as limitation 1).

---

## Deep audit 2026-07-09 — prioritized improvements

Three parallel deep-dives (segmentation stack; core method; explanation+eval+
logging) over the canonical pipeline. Measured smoke-run profile (90 images,
10 tags, Qwen2.5-VL-7B, 1×RTX 4090; **old 7-step numbering** — add 1 to steps
≥ 3 for the current 8-step pipeline): step 2 = 4m42s, **step 3 = 27m55s
(~80%)**, step 4 = 65s, step 5 = 18s, step 6 = 60s; **7 full model loads**
per run (scales as 5 + 2·TOP_N; +1 when logit-lens layer selection is on).

### MUST — validity and correctness of the method

1. **Wire in (or remove) the normalized/regrounded concepts.** Step 4 computes
   `combined_concept_snmf_gl.pth` (graph-Laplacian smoothing + L2 norm) and
   `..._gl_regrounded.pth` (re-grounded words), but step 5 and step 6 load
   `combined_concept_snmf_raw.pth` (`run_full_pipeline.py:692,742`) — **the
   normalization and regrounding never influence any reported result**. Decide
   which artifact is THE concept bank and wire it through explainer + eval
   (also: `--load_matched_features` in the reground call is inert).
2. **Fix the explainer normalization axis.**
   `vlm_explainer_multibatch.py:552,559`: `F.normalize(acts, dim=0)` normalizes
   each feature coordinate **across tokens** (and is applied twice). Cosine
   similarity is invariant to per-vector scaling but not per-coordinate
   scaling — reported similarities are distorted. Use `dim=-1` once, or drop
   the pre-normalization entirely (cosine already normalizes).
3. **Make the deletion/insertion eval do what it claims.**
   `eval/concept_deletion_eval.py:288-324` computes a gradient then discards it
   (`concept_mutiply=True` → ranking by `|v|` only); the docstring/module
   header claim `|grad·vec|` ordering. Either implement gradient-based ordering
   or delete the dead backward pass and rename. Also document that
   `--num_points` is a **percentage** (70 = first 70% of coordinates at fixed
   64-sample resolution), not a point count.
4. **Add a null model to the faithfulness curves.** Deletion/insertion + AUC
   have no random-direction / random-concept baseline (the BERT/CLIP table has
   one; reuse that pattern). Without it the curves cannot support a
   faithfulness claim.
5. **Unify the concept-identity key.** Three names coexist for the tag
   (`concept` in features, `concept_name` in save_features items,
   `image_concept_names` in combined output; grounding reads `concept`) — this
   already produced `None` concept names in explanations. One key, one schema,
   asserted end-to-end. Prerequisite for the ConceptBank.
6. **Seed the decomposition.** `DictionaryLearning` is constructed without
   `random_state` (`feature_decomposition.py:229-243`) — runs are not
   reproducible even with global seeds set. Pass the pipeline seed.
7. **Run manifest.** Write `logs/run_manifest.json` per run: resolved config,
   git SHA, per-step durations, SHA256 of key artifacts. Cheap (config is one
   object) and the single highest-value reproducibility addition.
8. **Decide the bbox-vs-mask contract.** `_normalize_bbox_to_patch_size`
   (`crops_to_json.py:955-980`) overwrites every bbox with a fixed
   `patch_size²` square around the centroid while the RLE keeps the true
   extent — JSON bbox and mask deliberately disagree. Confirm intended;
   document it in the crops.json schema either way.

### MAY — performance (ordered by measured impact)

9. **Step 3 (80% of wall time): re-enable the KV cache.** Generation runs with
   `use_cache=False` (`save_features.py:171-176,629`) — quadratic decode cost.
   The mean-over-generated-tokens hook fires per decode step regardless of KV
   caching; verify hook output equivalence on a few samples, then enable.
   Expected: step 3 several-fold faster. (If some hook variant truly needs
   full-sequence recompute, gate per hook.)
10. **Step 6: one model load instead of 2·TOP_N.** One
    `ConceptDeletionEvaluator` can serve all ranks × {insert, delete}
    (`run_full_pipeline.py:748-800`). Saves ~2.5-7.5 min/method. Additionally
    batch the 64 sequential batch-1 `norm→lm_head` forwards into one stacked
    `(64,D)` forward and vectorize across tokens (1-2 orders of magnitude on
    eval compute), and stop recomputing `_grad_and_order` per token when the
    order depends only on `vec`.
11. **Step 4: stop loading the full VLM twice for `lm_head`.** Both the
    decompose-grounding and reground subprocesses load the 7B model just to
    clone `lm_head` + tokenizer (`analyse_features.py:53-81`). Persist the
    cloned `lm_head` weights once (a few hundred MB) and load only that.
12. **Step 2: cache tag-independent work per image.** The per-tag outer loop
    re-loads, re-resizes, and re-segments an image once per tag it appears
    under; auto/semantic masks and CLIP embeddings are tag-independent
    (`crops_to_json.py:1301,1356,1558`). Deduplicate the image set, compute
    shared results once, fan out to tags. Up to N× (N = avg tags/image).
13. **Step 2 (SAM3 path): batch the point grid.** `predict_auto_masks_sam3`
    runs 256 sequential single-point forwards per image with `empty_cache()`
    after each (`sam3_utils.py:1073-1169`) and no autocast — batch the points,
    autocast bf16, hoist `empty_cache` out of inner loops (also in
    `crops_to_json.py:1367,1402,1566`).
14. **Step 5: pass `--batch_size`.** Full batching support exists
    (`_collate_inputs`, left-padding) but the orchestrator never passes it →
    batch=1 (`run_full_pipeline.py:707-722` vs configured `BATCH_SIZE`).
15. **Logging scheme** (single coherent upgrade): **[FIXED]** subprocess output
    now streams live (`Popen` + line relay + `python -u` +
    `PYTHONUNBUFFERED=1`, replacing the buffered `subprocess.run`). Still open:
    per-step logfiles `logs/stepN_<name>.log`; standardize `eval/*` and
    `inference/*` on `logging` (no bare `print`); route tqdm away from
    logfiles; per-step timing table at the end. (Manifest is MUST #7.)
16. **Precision**: fp16/bf16 for CLIP dedup embeddings and LangSAM (currently
    fp32); TF32 already on for SAM3 prompted path.
17. **Parallelize per-tag `DictionaryLearning`** (CPU, single-threaded,
    max_iter=12000, one call per tag) across a process pool.

### MAY — scientific strengthening

18. **Representation choice study.** The pooled mean over 1-2 answer tokens of
    a binary probe is a weak, prompt-dependent signal, and the final norm layer
    is maximally entangled with the yes/no decision. The logit-lens
    infrastructure (per-position hook) already exists — compare: mean-pooled
    vs last-prompt-token vs per-position representations, and a small
    mid/late-layer sweep, on the existing deletion/insertion + BERT/CLIP
    metrics. This is also a prerequisite for a stable, prompt-agnostic
    ConceptBank vector definition.
    **18b. Region-conditioning ablation (same study).** The current
    isolate-and-inpaint input (tight crop + background removal) maximizes
    causal purity but destroys context and yields few vision tokens on small
    regions. Compare against context-preserving alternatives, judged by the
    same AUC + BERT/CLIP metrics: (a) coordinate prompting — full untouched
    image, region passed as box coordinates in the prompt (Qwen2.5-VL is
    grounding-trained; zero pixel edits); (b) visual marking — mask outline /
    ellipse overlay on the full image (Set-of-Mark, red-circle prompting);
    (c) soft background attenuation — full image with dimmed/desaturated
    background (new method in `_apply_background_masking`, no crop);
    (d) two-image prompt — full image + crop together; (e) mask-pooled vision
    tokens — pool vision-token hidden states inside the mask, prompt-free
    (cleanest ConceptBank extraction, but changes the probing paradigm).
    (a) and (c) are ~1 day each, touching only sample prep in
    `save_features.py`.
19. **Rank selection instead of fixed K=2.** Every tag is forced into exactly
    2 sub-concepts. Use reconstruction-error/stability-based selection (or the
    ConceptBank's assign-vs-cluster threshold) per tag.
20. **Held-out evaluation.** Decomposition, purity filtering, and grounding
    exemplar selection all use the same crops — in-sample and optimistic.
    Split regions per tag (e.g. 80/20) and report grounding/faithfulness on
    held-out regions.
21. **De-circularize the purity filter.** `CLEAN_EXAMPLE_RATIO` validates
    concepts with the same model's own predictions. Add an external check
    (CLIP zero-shot agreement on the top-activating crops is already computed
    in eval — reuse it) or at minimum report both filtered and unfiltered
    results.
22. **ConceptBank readiness** (from plan-conceptGraph.prompt.md): freeze-and-
    append updater (assign if `max cos ≥ τ_assign`, cluster novel residuals,
    dedup at `τ_dup`) replacing per-tag re-factoring in append mode; L2-
    normalize banked vectors (today only the unused `_gl.pth` is normalized);
    fixed extraction spec (layer + pooling + prompt) stored with each vector;
    per-tag `.pth` layout already gives natural append granularity.

### OPTIONAL

23. Multi-GPU sharding of the deduplicated image list in step 2 (subprocess
    isolation already exists); a resident-model server process serving steps
    3/5/6 to eliminate all reloads (trades isolation for speed).
24. Async/threaded RLE encoding + JSON writes overlapped with GPU work;
    vectorize the O(k²) Python NMS in `predict_auto_masks_sam3`; re-enable the
    SAM3 model singleton (`crops_to_json.py:180-182`).
25. Eval polish: clip ±1σ bands at 0 (probabilities), plumb `--curve_points`,
    rename `num_points`→`coverage_pct`, unify the three CSV schemas in
    `eval/<method>/` with a schema version, wire the existing multi-run
    aggregation scripts into step 7 for cross-seed error bars.
26. Surface silently-swallowed failures: CSV/JSON writes wrapped in
    `try/except: pass` in the eval (`run_with_args`) must at least log; a
    missing rank CSV currently just vanishes from AUC/plots.
27. LangSAM `topn` parameter is accepted but unused
    (`langsam_utils.py:319-388`); dead `model["minimum_keep"]` reassignment
    per call — tidy.

---

## §1 — P0: The canonical orchestrator (`scripts/run_full_pipeline.py`)

### Fixed now
- **[FIXED]** `DEBUG_SAVE_VLM_INPUTS` defaulted to **"1"**: every plain run
  forced Step 3 to re-run and **cascade-deleted decomposition, explanations,
  eval, and plots**. Debug is now opt-in (default "0").
- **[FIXED]** Auto-`pip install python-dotenv` at import time (a network/env
  side effect in a pipeline run) — replaced with a clear error message.
- **[FIXED]** Concept selection (`NUM_CONCEPT` + `CONCEPTS_VOCAB` vocab
  filtering) was an internal function — extracted to
  `preprocessing/select_top_concepts.py` (standalone CLI + importable
  function); the orchestrator now delegates to it. Note: these two env vars are
  consumed **only** by this orchestrator — the shell variants ignore them, and
  `run_feature_decompose_dl.sh` reads the confusingly-similar `NUM_CONCEPTS`.

### Open — orchestrator correctness/robustness
- **Stale-output skip logic**: steps are skipped if artifacts exist (e.g. Step 3:
  any `features/*.pth`). A crashed step leaves partial output that skips the
  re-run — the same hazard fixed with `.done` markers in the legacy shell
  script. Adopt per-step completion sentinels here (write a marker file after
  each step; skip requires marker AND artifacts; `--force` to override).
- **Cascade deletion is aggressive**: `_delete_downstream_outputs` erases all
  downstream results whenever an earlier step re-computes. Combined with a bad
  skip decision this destroys hours of computation. Gate it behind a
  confirmation flag or move deleted outputs to a trash dir under `$OUTPUT_DIR`.
- **Step 1 runs in-process** (monkey-patching `sys.argv` to call
  `dataset_inference.main()` / `concept_image_mapping.main()`): no GPU-memory
  isolation for the model load, and a failing step exits via `SystemExit`
  mid-orchestrator. Steps 2–6 already use subprocesses — do the same for step 1
  and drop the `sys.argv` patching (same for step 7 plot imports).
- **Subprocess output is buffered**: `_run_python_subprocess` collects stdout
  and only relays it after the process exits — a multi-hour Step 3 shows
  nothing until it finishes. Stream line-by-line (`Popen` + `readline`).
- **Step 6 always re-runs** the BERT/CLIP table and AUC table even when all
  eval CSVs were skipped — add output-existence checks.
- **No `--force` / dry-run**; `--only-step`/`--skip-to-step` exist but ignore
  artifact freshness. Unify resume semantics with the sentinel scheme.

## §2 — P0: Bugs on the .py critical path (shared modules)

Fixed during this effort (most found by actual pipeline failures):

- **[FIXED]** `src/save_features.py`: guaranteed `NameError` on `toi_str` in the
  default path; generation-only decoding ran in teacher-forcing mode; user's
  `--save_filename` silently overwritten in the `json_crop_map` branch; dead
  `num_image_tokens_per_sample` block (`AttributeError` risk) removed.
- **[FIXED]** `src/analysis/feature_decomposition.py`: `from src.helpers.sae_`
  broke `decomposition_method="sae"` under the repo's `src`-on-path convention.
- **[FIXED]** `src/analysis/multimodal_grounding.py`: IndexError when metadata
  lists are empty/short — all per-sample picks bounds-checked (`_safe_pick`).
  **Root cause open:** save side writes key `concept_name`, grounding reads
  `concept` → concept names come out `None`. Align the key names end-to-end.
- **[FIXED]** `src/helpers/arguments.py`: `type=List[str]` (TypeError when used)
  and `type=bool` (could never be disabled) → `nargs="+"` / `str2bool`.
- **[FIXED]** `src/helpers/utils.py`: invalid tensor indexing in
  `extract_states_before_special_tokens` (rewritten as masked mean, verified
  numerically); `setup_hooks` now errors on hook/module length mismatch instead
  of silently attaching nothing.
- **[FIXED]** `src/langsam_utils.py`: one bad image hitting the upstream SAM2
  batch assertion killed the whole detection step — per-image failures are now
  skipped with a warning.
- **[FIXED]** `inference/dataset_inference.py`: swallowed fatal errors and
  exited 0 — now exits non-zero.
- **[FIXED]** `inference/vlm_explainer.py`: `--batch_size` defaulted from the
  global `BATCH_SIZE` env while rejecting >1 at runtime → dedicated
  `EXPL_BATCH_SIZE` (default 1). (The .py pipeline uses the multibatch variant;
  fix kept for the legacy path.)
- **[FIXED]** `src/combine_features.py`: no longer folds its own output back in
  on re-runs.
- **[FIXED]** Legacy shell orchestrator + 4 sub-scripts hardened (env
  precedence vs `.env`, `eval $@` removed, portable defaults, `.done` markers,
  `$PYTHON_BIN` routing) — validated end-to-end before the entry-point
  correction; useful until deprecation.

Open:
- `helpers/sae.py` vs `helpers/sae_.py` duplicate module (both load-bearing:
  `sae`→`sae_`, `sae2`→`sae`) — unify.
- Magic token `special_ids.add(106)` in shared pooling (`helpers/utils.py:613`)
  — derive from tokenizer.
- `models/image_text_model.py:113` casts the whole `BatchEncoding` to
  `model.dtype` — breaks under `device_map="auto"`, risks casting `input_ids`.
- Undocumented env knobs with unguarded `int()` casts in `save_features.py`
  (`MASK_BLUR_RADIUS`, `INPAINTING_METHOD`, `CROP_MODE`, `PATCH_SIZE`,
  `MASK_CONTEXT_PIXELS`, typo fallback `POSITIVE_NEGATVE_SEGMENT`) — promote to
  validated CLI args fed by the orchestrator config.
- Broad `except Exception:` fallbacks in `eval/concept_deletion_eval.py` and
  bare `except:` in `api/main.py` — narrow and log.
- `data/train1` is a broken symlink (`-> dtd/dtd/images`) — remove/repoint.

## §3 — P1: Missing features / capability gaps (future plan)

- **Label-split decomposition** (stronger alternative to the threshold-based
  positive/negative split; see "Positive/negative concept selection —
  algorithm and limitations" above). Split each tag's feature matrix by the
  VLM's per-region verdict BEFORE factorizing: Semi-NMF on the positive
  regions → positive concepts (pure by construction), Semi-NMF on the
  negative regions → negative/background concepts. Fixes limitation 1
  (a tag losing all directions because SNMF mixed its few positives with
  visually-similar negatives) at the root. Needs a guard for tags with
  fewer positives than K (skip or K=1) and reuses the existing
  positive/negative bank plumbing.
- **`sam3` is not installed** (imported 21× in `src/sam3_utils.py`,
  `src/save_features.py`) — so the user's preferred
  `CROP_MODE=semanticsegments_sam3` cannot run; `langsam` is the working
  fallback today. Future: install Meta's SAM3 package + checkpoints into the
  `xlvlms` env, declare it in requirements, and add a CI-checkable import guard
  with a clear error message.
- **HIGH PRIORITY — Per-tag layer selection via logit lens, with the layer
  recorded end-to-end.** **IMPLEMENTED 2026-07-10** (unit-tested, pending a
  full GPU pipeline run) — now a **dedicated pipeline step 3** between
  crops.json and feature extraction (the pipeline is renumbered to 8 steps:
  1 inference, 2 crops, 3 select layers, 4 features, 5 decompose,
  6 explainer, 7 eval, 8 plots). `src/select_layers.py` (subprocess, own
  model load) calls `src/helpers/layer_selection.py` per tag: it sweeps
  decoder layers on the tag's own crops.json regions (uniform sample,
  preprocessed exactly like extraction; `LOGIT_LENS_LAYER_SELECTION=1`,
  `LOGIT_LENS_MODE=patch|text`, `LOGIT_LENS_LAYERS`,
  `LOGIT_LENS_NUM_PATCHES`), scores the **relative** probability
  p(tag)/p(top-1) via the extended `score_hidden_state_with_logit_lens`,
  writes per-tag intermediates to `<output>/logitlens/<tag>/` and the
  tag → layer map to `<output>/logitlens/selected_layers.json`. Step 4
  (`save_features.py`) reads that map (or sweeps inline when run standalone
  without it), hooks the tag's layer, and the layer flows on as before:
  feature `.pth` key `selected_layer` → `analyse_features.py` (per-file
  `module_to_decompose` override) → grounding dict → `combine_concepts.py`
  (per-concept `selected_layer` list). Skip/rerun semantics: step 3 is
  skipped when `selected_layers.json` exists or the flag is 0; a rerun
  cascade-deletes features and everything downstream. Remaining gap:
  steps 6/7 (explainer + eval) still use the single static `LAYER_PATH`.
  Original verification notes:
  Today one static layer (`LAYER_PATH`, default `model.language_model.norm`)
  is used for every tag — passed uniformly as `--modules_to_hook` for
  extraction and `--module_to_decompose` for decomposition
  (`run_full_pipeline.py:133,584,632`). No code selects a layer per concept.
  The building blocks already exist but are **dead code with zero callers** in
  `src/helpers/logit_lens.py`: `score_hidden_state_with_logit_lens` (line 216;
  returns `concept_token_probability` per hidden state),
  `write_layer_selection_debug` (line 345; already emits `selected_layer` +
  per-layer `layer_candidates` JSON/histogram), `resolve_layer_modules`
  (expands layer-range specs like `layers.[0-27]` to module names), and the
  `save_hidden_states_logit_lens` hook (per-position states,
  `helpers/utils.py:742`). Plan:
  1. **Layer sweep per tag**: hook all candidate decoder layers on a few
     samples of the tag, score each layer's hidden states with
     `score_hidden_state_with_logit_lens`, and pick the layer with the highest
     probability of producing the tag token itself (argmax of
     `concept_token_probability`); dump the sweep via
     `write_layer_selection_debug`.
  2. **Extract at that layer**: run step-3 feature extraction with the
     selected layer as `modules_to_hook` for that tag.
  3. **Record it in the feature file**: the feature `.pth` currently stores
     `hidden_states` keyed by module name only (implicit); add an explicit
     separate key, e.g. `selected_layer` (+ the sweep scores), saved alongside
     the other keys in `save_hidden_states_to_file` (`helpers/utils.py:665`).
  4. **Record it in the decomposition output**: the grounding dict saved as
     `decompose_activations_*.pth` (`multimodal_grounding.py:225-247`) has
     **no layer key at all** today; add the same `selected_layer` key there
     and carry it through `combine_features.py` into the concept bank, so
     every extracted concept is traceable to the layer it came from.
- **Logit lens is not wired in** (`src/helpers/logit_lens.py` is
  implemented+unit-tested, module-level bugs fixed, and the
  `save_hidden_states_logit_lens` hook now keeps per-position states — but the
  analysis stage never calls `score_hidden_state_with_logit_lens`). Future:
  add a `logit_lens` analysis mode in `src/analyse_features.py` that consumes
  the per-position hidden states, expose `position_aggregation`/layer selection
  as CLI args, and add an end-to-end test with a small checkpoint.
- **Representation extraction: in-encoder patch masking instead of pixel-space
  region isolation.** Today a region is isolated in pixel space
  (crop + inpaint/blur background) before it is fed to the VLM. Future: pass
  the **full image** through the vision encoder and, inside the image encoder
  head, multiply by zero the patch embeddings whose patches fall outside the
  selected region (e.g. outside the sliding-window crop bbox), forcefully
  blocking all non-region patches. This keeps the encoder's native resolution
  and positional context, avoids inpainting artifacts, and pairs naturally
  with `CROP_MODE=sliding_window` since window bboxes map cleanly onto the
  patch grid. Needs: a bbox→patch-index mapping for the encoder's patch size,
  a forward hook (or wrapper) on the vision tower that zeros the masked patch
  embeddings, and an ablation comparing it against the current
  inpainting/blurring isolation (`INPAINTING_METHOD`, `MASK_BLUR_RADIUS`).
- **Concept names in explanations are `None`** until the `concept_name` vs
  `concept` metadata key drift is aligned (see §2) — after alignment, add a
  regression test asserting non-null concept names in `vlm_explanations.json`.
- **Explainer variants**: `vlm_explainer.py` vs `vlm_explainer_multibatch.py`
  are 74% identical; only multibatch is on the canonical path. Future: extract
  shared model-loading/seeding/input-prep into one module, keep a single entry
  point with a batch flag, delete the other.
- **BERT/CLIP scoring**: 5 near-duplicate scripts (~2,700 lines, two files 94%
  identical) — consolidate into one `eval/` module with a thin CLI; only
  `eval/clip_bert_score_eval.py` is on the canonical path.

## §4 — P1: Clean, reusable architecture (target design)

The pipeline should become an installable package with declarative steps:

1. **Packaging**: add `pyproject.toml`; make `src/` an installable package
   (working name `xlvlms`) with `pip install -e .`; delete every
   `sys.path.insert` (orchestrator, tests, scripts). Entry point:
   `python -m xlvlms.pipeline` (console script `xlvlms-pipeline`), replacing
   `scripts/run_full_pipeline.py` (keep a thin wrapper for compatibility).
2. **Config**: one typed config object (dataclass; `PipelineConfig` is a good
   start) with a single documented precedence chain CLI > env > `.env` >
   defaults, validation at startup (fail fast on unknown `CROP_MODE`, missing
   dirs, sam3-required-but-absent), and `--print-config` for reproducibility.
   Kill duplicate/near-miss knobs (`NUM_CONCEPT` vs `NUM_CONCEPTS`,
   `POSITIVE_NEGATVE_SEGMENT` typo fallback).
3. **Step framework**: each step declares inputs, outputs, and a completion
   sentinel; the runner decides skip/rerun from sentinels + artifact presence,
   supports `--force`, `--only-step`, `--from-step`, and streams subprocess
   output live. Cascade deletion becomes "move to trash dir + log", not rm.
4. **Single source of truth for shared logic**: concept selection
   (`preprocessing/select_top_concepts.py` — done), model loading/seeding
   (explainer consolidation), SAE module unification, artifact naming (the
   `combined_concept_{method}_raw.pth` vs `..._gl_regrounded.pth` drift between
   pipeline and `api/main.py:79`).
5. **Reproducibility**: rebuild `requirements.txt` (torch/transformers/etc. are
   currently commented out; `sam3`, `lang_sam`, `python-dotenv`, `inflect`,
   `qwen_vl_utils`, `timm`, `pandas`, `pytest` undeclared); align the
   torch/CUDA story across README/Dockerfile/requirements (currently three
   different stories); Dockerfile should install from requirements and not
   leak `.env` (dockerignore fixed) nor hardcode an ad-hoc run as entrypoint.
6. **Tests & CI**: pytest + root `conftest.py`; pipeline smoke test running
   `run_full_pipeline.py` on `data/val` with `CROP_MODE=random`,
   `NUM_CONCEPT=10`, tiny budgets, asserting each step's artifacts; fix the two
   pre-existing broken tests that drifted from `preprocessing/crops_to_json.py`
   (`test_crops_to_json_unit.py` imports missing `calculate_iou`;
   `test_semantic_sam3_postprocess.py` expects 4 masks, gets 2); move the GPU
   benchmark scripts out of `tests/`; GitHub Actions with lint + CPU tests
   (none exists today).
7. **Docs**: README currently documents the deprecated
   `run_full_pipeline_without_coroping.sh` — rewrite around
   `run_full_pipeline.py`, the `.env` contract, and one install story.

## §5 — P2: Security hardening (api/ + demo/, local-only today)

- **[FIXED]** `eval()` on `.pth`-sourced strings → `ast.literal_eval`; upload
  path traversal (`Path(file.filename).name`); `uvicorn reload=True`; `.env`
  excluded from Docker images. **Done by user:** HF token in `.env` rotated.
- Open (required before any network deployment): `torch.load(weights_only=False)`
  on served `.pth` files (unpickling RCE — move to safetensors or
  `weights_only=True`); CORS `*` + `allow_credentials=True` + no auth + binds
  `0.0.0.0`; unbounded in-memory uploads; blocking 1-hour `subprocess.run`
  inside an async handler with unlocked global state; ad-hoc
  `outputs/screen_run/...` artifact as default concept source.

## §6 — P3: Repository cleanup

- Delete dead files: `src/*_backup.py`, `src/helpers/utils_backup.py`,
  `src/models/gemma3_backup.py` (shadows the live `Gemma3nVL`),
  `preprocessing/crops_to_json_old.py`, `glimpse/chat_gpt_implentaion_of_glimplse.py`,
  the empty `src/main.py` stub, and the legacy orchestrators once parity lands.
- Fix `.gitignore` (`.png`/`.pth` entries lack `*` and match nothing; `temp/`
  never ignored) and untrack clutter: vendored `git-lfs-3.5.1/` (11 MB binary),
  `temp/`, root images (`output.png` 8.3 MB, grids, JPEGs), `.docx`,
  `## Chat Customization Diagnostics.md`, `plan-*.prompt.md`,
  `api/test_result_custom_prompt.json`, `api/TKG.png`, notebook outputs.
- **History rewrite (approved, needs coordination)**: `.git` is ~1.1 GB;
  after working-tree cleanup, `git filter-repo --strip-blobs-bigger-than 1M`
  (with an asset allowlist), force-push, collaborators re-clone.
- `.gitmodules` SSH URL for `model-kg-visualization` → HTTPS or optional.

---

## Validation status

- **Legacy shell pipeline** (`run_full_pipeline.sh`, PNG flow): validated
  end-to-end 2026-07-09 on `data/val` (90 images, Qwen2.5-VL-7B, snmf,
  detector=langsam), exit 0, all artifacts produced.
- **Canonical Python pipeline** (`run_full_pipeline.py`, JSON flow): validated
  end-to-end 2026-07-09 on `data/val` (90 images, Qwen2.5-VL-7B, snmf,
  `CROP_MODE=langsam`, `NUM_CONCEPT=10`), exit 0. Vocab filter reduced 1,343
  concept candidates to top 10; crops.json 3.2 MB of coordinates with **zero
  crop PNGs written**; 10 feature .pth; 2.9 MB explanations JSON; deletion/
  insertion CSVs + BERT/CLIP + AUC tables + plots all produced. One dependency
  installed along the way: `pycocotools` (needed for `RLE=1`) was missing from
  the `xlvlms` env despite being listed in requirements.
  **Caveat found post-run:** the vocab filter was silently skipped in that run —
  `.env` had `CONCEPTS_VOCAB="${ROOT_DIR:-$PWD}/..."`, which bash expands but
  python-dotenv resolves to the literal `$PWD/...` (it does not expand bare
  `$VAR` inside a `${VAR:-default}` default). The "top 10" was therefore by
  image count only. Fixed by using a plain absolute path in `.env`.
  **Convention:** `.env` is consumed by BOTH `source` (shell) and
  python-dotenv — only use syntax both understand: plain values or simple
  `${VAR}` references, no shell default-expansion, no `$(...)`. With the fix,
  the vocab filter works (verified: 1,343 concepts → 1 on `data/val`, since
  `src/assets/vocab.txt` is a 6-entry vehicle vocabulary from the COCO
  experiments).

## Suggested execution order (future sessions)

1. **Deep-audit MUST items 1-8** — they gate the validity of every result the
   pipeline produces (dead gl/reground wiring, explainer normalization axis,
   eval ordering claim + null baseline, key unification, decomposition seed,
   run manifest, bbox contract decision).
2. **MAY performance 9-11** (KV cache, single-process eval, lm_head persistence)
   — the three biggest measured wins; then 12-14 (segmentation caching, SAM3
   grid batching, explainer batching) and the logging scheme (15).
3. §4.1-4.3 packaging + config + step framework; §4.6 smoke test + CI.
4. **MAY scientific 18-22** — representation/layer study, rank selection,
   held-out eval, purity de-circularization, ConceptBank updater.
5. §3 capability gaps (sam3 install, logit-lens analysis wiring); §6 cleanup +
   history rewrite; consolidation.
6. §5 API hardening; docs last.
