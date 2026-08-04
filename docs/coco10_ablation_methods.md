# coco10 ablation: method differences and run instructions

Three independent axes: `PROMPT_TEMPLATE` (`cgdl`/`non_contrastive`/`null`, §3),
`CROP_MODE` (`none`/`sliding_window`/`langsam`, §2), and `DECOMP_STRATEGY`
(`per_tag`/`pooled`, §4) — any combination is valid. The **default** matrix (one
strategy per template, matching each template's original design) is 3×3×2 seeds =
**18 configs**; the **full** matrix (every template at both strategies) is
3×3×2×2 seeds = **36 configs**. Canonical entry point for a single config is
`scripts/run_full_pipeline.py`; `scripts/run_ablation.py` orchestrates the whole
matrix via `--decomp-strategies {default,all}`.

## 1. Preparing the dataset

The ablation needs `data/coco10/{train_all,val,val_masked,val_grids}` to exist
before any config can run. Two scripts build it, in order, from the real COCO
train2017/val2017 pools at
`/media/NVME_8TB/abka03/Projects/vlm_space/local-evidence-vlm/data/raw/`
(`train2017/`, `val2017/`, `annotations/instances_{train,val}2017.json`).

```bash
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms

# Step 1: pull the 10 categories (apple, banana, bird, cake, cat, cup, dog,
# donut, knife, orange) from real COCO train2017 (-> train_all/) and
# val2017 (-> val/), plus manifest.json recording per-category image/annotation
# ids for the next script.
python preprocessing/build_coco10_dataset.py \
  --train-cap 300 \
  --test-cap 50 \
  --seed 42
#   --train-cap 20 --test-cap 5   # smaller pilot subset, for a fast smoke test

# Step 2: decode each val image's ground-truth COCO segmentation mask for its
# category, crop tight + padding -> val_masked/<category>/, then assemble
# val_masked/ into 2x2 grids -> val_grids/ (same layout run_full_pipeline.py's
# explainer/eval steps expect as IMAGE_ROOT).
python preprocessing/build_coco10_masked_grids.py \
  --context-pixels 16 \
  --num-grids 50 \
  --grid-n 4 \
  --image-size 384 \
  --seed 42
#   --num-grids 10   # pilot
```

Both scripts are idempotent — re-running clears and rebuilds their own output
subdirectories (`train_all`/`val`, and `val_masked`/`val_grids` respectively)
rather than accumulating stale files from a prior build. `src/assets/coco10_vocab.txt`
(the 10 category names, one per line — used to filter/rank candidate concepts in
step 1 of the main pipeline) is a static, already-committed asset; it doesn't need
to be regenerated.

Output layout after both steps:

```
data/coco10/
  train_all/<file>.jpg          flat pool, all categories' train images (INPUT_DIR)
  val/<category>/<file>.jpg     per-category unmasked val images
  val_masked/<category>/<file>.jpg   ground-truth-mask-cropped val images
  val_grids/*.jpg                2x2 grids of val_masked images (IMAGE_ROOT)
  manifest.json                  per-category train/test image+annotation ids
```

## 2. What differs by `CROP_MODE` (input processing)

`CROP_MODE` controls how each training image is turned into the crop(s) fed to the
VLM during feature extraction (steps 1-4). It has nothing to do with prompt wording.

| | `none` | `sliding_window` | `langsam` |
|---|---|---|---|
| What gets cropped | Nothing — whole image, resized to `IMAGE_SIZE_WIDTH` | Geometric `PATCH_SIZE × PATCH_SIZE` tiles, stride = `SLIDING_WINDOW_STRIDE_RATIO` × window width | Real object masks/boxes from LangSAM (Grounding-DINO + SAM), text-prompted per tag |
| Detector used | none (`step_2_build_crops_json` short-circuits to `_build_whole_image_crops_json`, no subprocess) | none — pure geometry + CLIP scoring (tiles are ranked by CLIP similarity to the tag phrase, top-scoring kept, then deduped) | Real segmentation model (`preprocessing/crops_to_json.py --detector langsam`) |
| bbox geometry | Full image | Fixed `PATCH_SIZE×PATCH_SIZE` square (160×160 in this ablation) — intentional, by design | Tight, variable-size box from the real detected mask, clamped to image bounds (not forced into a fixed square) |
| Relative cost | Cheapest — no cropping/detection step at all | Medium — CLIP-scores many candidate tiles per image | Slowest — runs a real detector per image |
| Config values used | `PATCH_SIZE=160` (irrelevant here) | `PATCH_SIZE=160`, `SLIDING_WINDOW_STRIDE_RATIO=0.3` | `PATCH_SIZE=160` (used only for the RLE/bbox filter, not to constrain box shape) |

`MASKS_PER_IMAGE=500`, `CONCEPT_MASKS_PER_IMAGE=2000` apply to all three (upper
bounds, rarely hit).

## 3. What differs by `PROMPT_TEMPLATE`

This axis changes prompt wording and hidden-state extraction only — decomposition
strategy, layer selection, and purity filtering are now the independent
`DECOMP_STRATEGY` axis (§4), not tied to a specific template. Prompt text below is
`TASK_PROMPTS[template]["ShortCaptioning"]` from `src/models/constants.py` — the
key `JSONDataset` (used by steps 3/4) actually reads.

| | `cgdl` | `non_contrastive` | `null` |
|---|---|---|---|
| Prompt sent to the VLM | `"Classify the image as either [concept] or No [concept] based on its content. Return only the predicted label."` — `[concept]` substituted with the tag name per image | `"What are the objects in the image?"` — same for every image, tag-agnostic, open-ended | `""` — literally empty instruction |
| `MAX_NEW_TOKENS` | 50 (default) | 50 (default) | 1 (forced by `run_ablation.py`; a single decoder forward pass) |
| `HOOK_NAMES` (step 4) | `save_hidden_states_for_token_of_interest` — extract the hidden state at the position where the tag word (or "No `[concept]`") is generated; falls back to position 0 if it never appears, with a found/not-found mask saved alongside | Same as `cgdl` — the tag word can appear anywhere in the open caption, so the same extraction applies | `save_hidden_states_mean` — a single forward pass has nothing to disambiguate, mean-of-one-token is the same as the token itself |
| Token-of-interest sample filtering (`keep_only_token_of_interest`, `src/analysis/__init__.py`) | **Mandatory (always True)** — samples where the tag word was never found are dropped, not pooled in as if they were real matches | Mandatory, same reason | N/A — `null`'s files carry no `token_of_interest_mask` key (mean hook), so this is a no-op regardless of value |

Both `cgdl` and `non_contrastive` used to differ here (mean-pool vs.
token-of-interest, and non_contrastive's filtering used to be disabled) — as of
this revision they're treated identically: any template whose response can run
several tokens gets the same precise per-tag-word extraction and the same
mandatory drop of not-found samples, since mean-pooling always risks diluting the
signal with filler/punctuation or biasing toward whatever token the response
happens to end on, regardless of which template produced that response.

## 4. What differs by `DECOMP_STRATEGY`

Independent of `PROMPT_TEMPLATE` and `CROP_MODE` — any template/crop combination
can use either strategy via the `DECOMP_STRATEGY` env var (`scripts/run_full_pipeline.py`,
`PipelineConfig.decomp_strategy`). Defaults when not set explicitly: `cgdl` →
`per_tag`, `non_contrastive`/`null` → `pooled` (each template's original design;
set `DECOMP_STRATEGY` explicitly to override, e.g. `cgdl`+`pooled` or
`non_contrastive`+`per_tag`).

| | `per_tag` | `pooled` |
|---|---|---|
| Step 5 decomposition | One SNMF call **per tag** (`DECOMP_COMPONENTS=2` concepts/tag), then merged + purity-filtered by `src/combine_concepts.py` | Every tag's features concatenated **once** (`src/combine_features.py`) into a single matrix, one SNMF call for the whole dataset (`DECOMP_COMPONENTS_GLOBAL=20` concepts total) |
| Per-tag logit-lens layer selection (step 3, `LOGIT_LENS_LAYER_SELECTION`) | Honored (if set in `.env`) — each tag can hook a different decoder layer, since each tag is decomposed independently, so mismatched layers across tags don't conflict | **Force-disabled** regardless of `.env` — every tag must use the same static `LAYER_PATH` (`model.language_model.norm`), since pooled features need one consistent module key across all tags |
| `CLEAN_EXAMPLE_RATIO` purity filter | Applied as configured (0.2 by default) — meaningful, since each tag has its own positive/negative split to filter on, regardless of which template produced it | **Forced to 0.0** — pooled decomposition has no per-tag split to check a purity ratio against (every sample is pooled together unfiltered); every SNMF direction with ≥1 assigned sample is accepted instead |

This is why, e.g., `cgdl`+`pooled` and `non_contrastive`+`per_tag` are both valid,
newly-supported combinations: the rules above apply the same way regardless of
which template chose that strategy.

**Zero-sample tags in `per_tag` mode**: with the token-of-interest hook mandatory
(§3) and a template like `non_contrastive` whose match rate is well under 100%, a
tag can end up with 0 samples surviving the filter for a given run (e.g. the model
never happened to say "cup" for any of that tag's crops) — more likely the smaller
`IMAGE_BUDGET` is. `src/analysis/__init__.py::analyse_features()` detects this and
skips decomposition for just that tag with a warning log
(`Skipping decomposition for '...': 0 samples survived token-of-interest
filtering`), rather than crashing the whole config — found and fixed via smoke
testing the new `non_contrastive`+`per_tag` combination (a 0-sample tag with the
old code crashed `decompose_activations()`'s stats print on an empty matrix).
`pooled` mode isn't affected the same way, since all tags' samples are combined
into one matrix before decomposition, so one empty tag doesn't zero out the whole
matrix.

**Validation status**: all 4 combinations newly enabled by this axis
(`cgdl`+`pooled`, `non_contrastive`+`per_tag`, `null`+`per_tag`, `null`+`none`)
have been smoke-tested end-to-end (small `IMAGE_BUDGET`, one crop mode each) and
confirmed working, including the fix above — not just implemented in theory. A
full-scale run of the expanded matrix (§5) hadn't been launched as of this
revision.

## 5. Full settings grid

Shared across every config in the ablation (`scripts/run_ablation.py`'s
`ENV_OVERRIDES`, unless noted as forced/overridden above):

```
DATASET           coco10 (INPUT_DIR=data/coco10/train_all, IMAGE_ROOT=data/coco10/val_grids)
DECOMP_METHODS    snmf
IMAGE_BUDGET      -1        (all training images per category)
EXPL_MAX_IMAGES   -1        (all val_grids images)
CLEAN_EXAMPLE_RATIO  0.2    (forced to 0.0 whenever DECOMP_STRATEGY=pooled — see §4)
CONCEPTS_VOCAB    src/assets/coco10_vocab.txt
EXPL_PROMPT_MODE  mcq
EXPL_CHOICES      apple,banana,bird,cake,cat,cup,dog,donut,knife,orange
VLM_MODEL         google/gemma-3n-E4B-it
PATCH_SIZE        160
BATCH_SIZE        16
DL_ALPHA          23
SAE_TARGET_SPARSITY  0.99
SEED              42 and 43 (two runs per config)
```

The configs actually run, via `scripts/run_ablation.py --decomp-strategies {default,all}`:

```
default (18 configs — one strategy per template, each template's own default):
{cgdl, non_contrastive, null} × {none, sliding_window, langsam} × {42, 43} = 18

all (36 configs — every template at both strategies):
{cgdl, non_contrastive, null} × {none, sliding_window, langsam} × {per_tag, pooled} × {42, 43} = 36
```

`null` runs with all 3 crop modes (including `none`) for a uniform matrix — earlier
versions of this ablation only ran `null` with `langsam`/`sliding_window`.
Directory names only get a `_<decomp_strategy>` suffix when the strategy is NOT
that template's default (e.g. `sliding_window_cgdl_seed42` stays unchanged for the
default `per_tag` run; the new pooled variant is
`sliding_window_cgdl_pooled_seed42`) — so the 18 pre-existing default-strategy
directories keep their original names and are recognized as already complete.

## 6. Running a single method

Each config is one invocation of `scripts/run_full_pipeline.py`, configured entirely
through environment variables (loaded from `.env`, then overridden per-config). To
run one config manually:

```bash
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms

export DECOMP_METHODS=snmf
export IMAGE_BUDGET=-1
export EXPL_MAX_IMAGES=-1
export CLEAN_EXAMPLE_RATIO=0.2      # forced to 0.0 automatically whenever DECOMP_STRATEGY resolves to pooled
export INPUT_DIR=data/coco10/train_all
export IMAGE_ROOT=data/coco10/val_grids
export CONCEPTS_VOCAB=src/assets/coco10_vocab.txt
export NUM_CONCEPT=-1
export EXPL_PROMPT_MODE=mcq
export EXPL_CHOICES="apple,banana,bird,cake,cat,cup,dog,donut,knife,orange"

export PROMPT_TEMPLATE=cgdl          # or non_contrastive, or null
export CROP_MODE=sliding_window      # or none, or langsam
export DECOMP_STRATEGY=per_tag       # or pooled -- omit to use this template's default (see §4)
export SEED=42
export OUTPUT_DIR=outputs/ablation_coco10/sliding_window_cgdl_seed42

# only needed for PROMPT_TEMPLATE=null:
# export MAX_NEW_TOKENS=1

python scripts/run_full_pipeline.py --output-dir "$OUTPUT_DIR" --decomp snmf
```

`HOOK_NAMES`, per-tag layer selection, and the purity filter do **not** need to be
set manually — `run_full_pipeline.py` picks the right value automatically based on
`PROMPT_TEMPLATE` (§3) and `DECOMP_STRATEGY` (§4). Do not set `HOOK_NAMES` or
`CLEAN_EXAMPLE_RATIO` in `.env` yourself; an explicit value there overrides the
automatic default for every config.

Each step of the pipeline skip-checks its own output (crops.json, features/,
concept/, etc.), so re-running the same `OUTPUT_DIR` after a partial failure
resumes instead of redoing completed steps — delete the specific stale
subdirectory (usually `concept/`) if a step needs to be forced to rerun with a
code fix.

## 7. Running all methods together

```bash
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms

python scripts/run_ablation.py \
  --dataset coco10 \
  --image-budget -1 \
  --expl-max-images -1 \
  --clean-example-ratio 0.2 \
  --devices cuda:0,cuda:1,cuda:2 \
  --decomp-strategies default   # or 'all' for the full 36-config matrix (§5)
```

This builds the 18-config matrix (or 36 with `--decomp-strategies all`), shares the
step-1 VLM captioning pass across all configs (identical regardless of
`CROP_MODE`/`PROMPT_TEMPLATE`/`DECOMP_STRATEGY`, cached under
`outputs/ablation_coco10/_shared_inference_budgetneg1/`), and runs 3-way in parallel
across the listed devices (a queue hands out devices as they free up — you don't
need exactly 3 GPUs, fewer works too, just less parallel). Each config's own log is
at `outputs/ablation_coco10/<config_name>/ablation_run.log` (see §5 for the naming
convention);
the driver's own stdout has the running OK/FAILED summary.

`--dry-run` prints the config plan (18 or 36 depending on `--decomp-strategies`)
without launching anything — useful to sanity check before a real (multi-hour)
launch.

To retarget just a subset of configs (e.g. re-running only the ones that failed),
there's no CLI flag for that — either edit `PROMPT_TEMPLATES`/`CROP_MODES`/`SEEDS`/
`DECOMP_STRATEGIES` at the top of `scripts/run_ablation.py` temporarily, or write a
small standalone script that mirrors `run_one()`/`ENV_OVERRIDES` from that file for
just the configs you need (this is what was done mid-session to retry failed
`non_contrastive` configs without re-running the ones that had already passed).

## 8. Aggregating results

```bash
python scripts/ablation_report.py --dataset coco10 --decomp-strategies default   # or 'all'
```

Must be passed the same `--decomp-strategies` value used to run the ablation.
Reads every `outputs/ablation_coco10/<config>/` directory, averages metrics across
both seeds per (prompt_template, crop_mode, decomp_strategy) combination, and
writes `outputs/ablation_coco10/ablation_report.csv` (18 or 36 rows) — printed to
stdout as a table too.

## 9. Evaluation metrics — what each one measures and how it's implemented

Every metric below is read directly from files each config's own pipeline run
already produces (step 6 explainer, step 7 faithfulness eval) or computed fresh
from the concept bank by `ablation_report.py` itself. None of them require a
separate eval run — they're all byproducts of the normal 8-step pipeline.

### 9.1 Grounding: BERT score / CLIP score

**Source**: `eval/clip_bert_score_eval.py`, invoked as step 7's last stage; output
table `eval/snmf/clip_bert_topk_table.csv` (one row per rank `k=1..5`).

**Question answered**: when the model names a concept for an image (via the step-6
explainer's top-k concept predictions per generated token), does that concept's own
*label* (BERT) and *grounding images* (CLIP) actually match what a human/independent
model would say about that same image? This tests whether concepts are
semantically real, independent of the faithfulness question (metrics in §9.2) of
whether they causally drive the model's output.

**BERT score** — for each image, take the model's actual generated caption word(s)
(`model_output`, tokenized and lightly aligned back to sub-word pieces via
`build_gt_token_map`) as the ground truth. For the top-`k` concepts predicted for
that token, compare each concept's clean vocab label (`concept_name`, e.g.
`"apple"` — not the noisy logit-lens `text_grounding` readout) against the ground
truth word using `bert-score`'s F1 (contextual embedding similarity, `lang="en"`).
Averaged per-image, then across images → `bert_mean` per rank `k`.

**CLIP score** — for the same top-`k` concepts, crop each concept's *grounding
images* (`image_grounding_paths` + `image_grounding_bboxes` from the concept bank —
the images that most activate that concept during decomposition) to their bbox, run
CLIP ViT-B/32 image encoding, encode the ground-truth word as text, and score
cosine similarity (scaled by `w=2.5`, clipped at 0): `clip_score = 2.5 · max(0, cos(image_embed, text_embed))`.
Averaged the same way as BERT.

**Random baseline** (`random_bert_score`/`random_clip_score`) — identical
computation, but the top-`k` concepts are replaced with `k` uniformly-random
concepts sampled from the *same config's own concept bank* (`_build_random_predictions`,
seeded, no replacement needed to exceed the concept count). This is the
"what score would you get from a concept bank that carries no real information for
this specific image" floor — real scores should sit meaningfully above it.

### 9.2 Faithfulness: insertion / deletion AUC

**Source**: `eval/concept_deletion_eval.py` (produces the raw probability
curves) → `eval/concept_curve_auc_eval.py` (integrates each curve into one AUC
number); output table `eval/snmf/concept_curve_auc_token_table.csv`.

**Question answered**: does the concept vector assigned to a token *causally* drive
the model's confidence in the word it actually generated? This is orthogonal to
grounding (§9.1) — a concept could have a perfectly sensible label/image but zero
actual causal influence on the model's output, or vice versa.

**Mechanics** (`ConceptDeletionEvaluator`): the concept vector lives in a mid-layer
hidden-state space. `LMHeadSubModel` reprojects it through `[final_norm] → lm_head`
(logit-lens style) to get vocabulary logits, then reads the softmax probability of
the token the model actually generated at that position.

- **Deletion** (`evaluate_token`): start from the full concept vector, progressively
  zero out its coordinates in some order, and track how the target token's
  probability falls as more of the vector is deleted.
- **Insertion** (`evaluate_token_insertion`): the inverse — start from an all-zero
  vector and progressively insert coordinates back in the same order, tracking how
  probability rises.
- **Coordinate order** (`--order_mode`, default **`value`**): coordinates are
  sorted by descending `|vec|` magnitude — the concept vector's own largest
  components go first. (`gradient` — rank by `|∂logit/∂vec · vec|` — and `random` —
  a permutation, the chance-level baseline below — are also implemented but not the
  default; `value` was chosen deliberately over `gradient` since gradient-based
  ranking can reweight importance far from the vector's own actual magnitude
  structure.)
- Both directions are computed **per rank** (rank 1/2/3 = the concept the
  explainer ranked highest/2nd/3rd-most-similar for that token) and in **token
  mode** — one measurement per generated token across the whole val set, not
  collapsed to one-per-image — with `NUM_POINTS=70` samples along the curve.

**AUC (relative)** (`_compute_auc_relative` in `concept_curve_auc_eval.py`): the
raw softmax probability sits in a tiny, scale-dominated band (~1/vocab_size, e.g.
~4e-6) that isn't meaningfully comparable across configs/ranks on its own. The
curve is min-max rescaled to its **own** `[0, 1]` range — `(y - y_min) / (y_max -
y_min)` — then integrated via trapezoidal rule and divided by the x-span. This
answers "where does this curve sit, on average, between its own floor and
ceiling" — scale-invariant, comparable across configs. This is the
`addition_auc_*`/`deletion_auc_*` family in `ablation_report.csv`.

**Random baseline** (`addition_auc_*_random`/`deletion_auc_*_random`) — the exact
same relative-AUC computation, but with `--order_mode random` (a random coordinate
permutation instead of value-magnitude order). The gap between the real and random
AUC is what the *value-magnitude ordering itself* buys over chance — both curves
still integrate the same causal deletion/insertion mechanism, only the coordinate
order differs.

### 9.3 Concept-bank quality: sparsity, overlap, instability

These three don't need any VLM inference at all — they're computed directly from
the learned concept dictionary (`concept/snmf/combined_concept_snmf_*_raw.pth`,
the `concepts` tensor, shape `[n_concepts, hidden_dim]`) by `ablation_report.py`
itself, so they're identical in spirit regardless of crop mode or prompt template.

**Sparsity** (`sparsity_higher_better`, `hoyer_selectivity`) — the Hoyer sparsity
measure per concept atom: `(√n - L1/L2) / (√n - 1)`, clamped to `[0, 1]` and
averaged across atoms. 1.0 = a single dominant coordinate (maximally selective);
0.0 = mass spread evenly across all `n` dimensions. Higher means each concept
"points at" a more concentrated, interpretable direction rather than a diffuse
blend.

**Overlap** (`overlap_lower_better`, `concept_overlap`) — mean pairwise cosine
similarity between all *different* concept atoms (small values below `1e-3`
zeroed first to suppress noise), off-diagonal only. Lower means the discovered
concepts are more mutually distinct — the dictionary isn't wasting capacity on
near-duplicate directions.

**Instability** (`instability_lower_better`, `matched_cosine_similarity`) —
compares the seed-42 and seed-43 concept banks for the *same* config: L2-normalize
both, compute the full cross-similarity matrix, solve optimal one-to-one atom
matching via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`,
maximizing total matched similarity), then report `1 − mean(matched similarity)`.
Lower means the same 20 (or 2-per-tag) concepts get rediscovered almost identically
across two random seeds — a proxy for whether the decomposition is finding a real
structure in the data versus fitting noise that happens to differ every run.

### 9.4 Where each number in the results table comes from

| Column | Computed in | Needs a real VLM to compute? |
|---|---|---|
| `bert_score`, `clip_score`, `*_rank{1,2,3}` | `eval/clip_bert_score_eval.py` (step 6 output) | Yes — needs concept-image crops + captions |
| `random_bert_score`, `random_clip_score` | same file, random-concept substitution | Yes, same inputs |
| `addition_auc_*`, `deletion_auc_*` | `eval/concept_deletion_eval.py` + `concept_curve_auc_eval.py` (step 7) | Yes — needs the model's own logit-lens head |
| `addition_auc_*_random`, `deletion_auc_*_random` | same, `--order_mode random` pass | Yes, same model |
| `sparsity_higher_better`, `sparsity_frac_near_zero_1pct`, `overlap_lower_better` | `ablation_report.py`, direct from `concept/snmf/*_raw.pth` | No — pure linear algebra on the saved concept vectors |
| `instability_lower_better` | `ablation_report.py`, compares seed42 vs. seed43 banks | No — same, just needs both seeds' banks present |

## 10. Running individual evaluation scripts

Steps 6-7 of `run_full_pipeline.py` already run all of this automatically per
config. The commands below are for re-running (or debugging) one evaluation
script standalone — e.g. after tweaking a metric's implementation, without
redoing the whole pipeline. All paths below assume a config directory like
`outputs/ablation_coco10/sliding_window_cgdl_seed42/` — substitute your own
`OUTPUT_DIR`, `method` (decomposition method, default `snmf`), and `layer_path`.

```bash
conda activate xlvlms
cd /media/NVME_8TB/abka03/Projects/xl-vlms

CFG=outputs/ablation_coco10/sliding_window_cgdl_seed42
METHOD=snmf
CONCEPT_PATH="$CFG/concept/$METHOD/combined_concept_${METHOD}_cr0.2_raw.pth"   # cr0.0 whenever DECOMP_STRATEGY=pooled (see §4)
EXPLANATIONS="$CFG/explanations/$METHOD/vlm_explanations.json"
EVAL_OUT="$CFG/eval/$METHOD"
```

**10.1 Faithfulness curves — `eval/concept_deletion_eval.py`** (one call per
rank × {insertion, deletion} × {real, random} — the pipeline runs `4 × TOP_N`
of these; `TOP_N` defaults to 3, so 12 calls per config):

```bash
# Real (value-magnitude order), rank 1, insertion:
python eval/concept_deletion_eval.py \
  --results_json "$EXPLANATIONS" \
  --concept_path "$CONCEPT_PATH" \
  --model_name google/gemma-3n-E4B-it \
  --layer_path model.language_model.norm \
  --mode token \
  --num_points 70 \
  --out_dir "$EVAL_OUT" \
  --device cuda:0 \
  --rank 1 \
  --insertion
# writes c_insertion_token_rank1.{png,csv,json}

# Same, but deletion instead of insertion: drop --insertion
# Same, but the random-order chance-level baseline: add --order_mode random
#   (writes the same filenames with a _random suffix)
# Repeat for --rank 2 and --rank 3 (TOP_N=3 in this ablation)
```

Useful extra flags when running standalone: `--mode sequence` (one
measurement per image via `top_concepts_over_sequence` instead of per
generated token), `--order_mode gradient` (the non-default `|grad·vec|`
ranking), `--grad_top_zero_frac 0.15` (skip the top 15% of the ordering before
starting the sweep — a smoothing knob), `--curve_points 64` (resolution along
the sweep).

**10.2 Grounding scores — `eval/clip_bert_score_eval.py`** (one call covers all
ranks `1..max_k` and both real + random-concept baseline in a single run):

```bash
python eval/clip_bert_score_eval.py \
  --json_path "$EXPLANATIONS" \
  --concept_path "$CONCEPT_PATH" \
  --max_k 3 \
  --seed 42 \
  --out_dir "$EVAL_OUT" \
  --output_prefix clip_bert_topk
# writes clip_bert_topk_table.{csv,json}
```

**10.3 AUC summary table — `eval/concept_curve_auc_eval.py`** (reads the curve
JSONs §10.1 already wrote in `$EVAL_OUT`, integrates each into one number per
rank — run this *after* §10.1, not standalone):

```bash
python eval/concept_curve_auc_eval.py \
  --out_dir "$EVAL_OUT" \
  --top_n 3 \
  --mode token \
  --output_prefix concept_curve_auc_token
# writes concept_curve_auc_token_table.{csv,json} -- this is what
# ablation_report.py's addition_auc_*/deletion_auc_* columns read
```

## 11. Running all evaluations together

For one config, step 7 alone (skipping steps 1-6 if their outputs already
exist) can be re-triggered by just re-running the full pipeline command from
§6 — every step skip-checks its own output, so if `features/`, `concept/`,
and `explanations/` are already there, only step 7 (and 8, plots) actually
runs:

```bash
python scripts/run_full_pipeline.py --output-dir "$CFG" --decomp snmf
```

To force step 7 specifically to redo (e.g. after changing an eval script),
delete its output first so the skip-check doesn't short-circuit it:

```bash
rm -rf "$CFG/eval"
python scripts/run_full_pipeline.py --output-dir "$CFG" --decomp snmf
```

To run every evaluation (§10.1-10.3) for **all configs** at once, this is
just step 7 of `run_ablation.py`'s full sweep (§7) — there's no separate
"evals only" entry point, since evals depend on each config's own concept bank
and explanations already existing:

```bash
python scripts/run_ablation.py \
  --dataset coco10 --image-budget -1 --expl-max-images -1 \
  --clean-example-ratio 0.2 --devices cuda:0,cuda:1,cuda:2 \
  --decomp-strategies default   # or 'all'
```

Then aggregate everything into one table (§8):

```bash
python scripts/ablation_report.py --dataset coco10 --decomp-strategies default   # or 'all'
```
