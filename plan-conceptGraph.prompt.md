# Sophisticated `.env` + Explainable Inference + Append-Only ConceptBank (SQLite)

This document is an implementation-ready spec that **reuses the existing pipeline code** while adding:
- a unified `.env` configuration surface,
- an “on-the-fly” inference CLI for explainable inference,
- **append-only concept bank storage** in **SQLite via SQLAlchemy** so **new concepts can be added without recomputing old ones**,
- dry-run/offline smoke tests using `data/` (no GPU, no model downloads),
- optional multi-GPU sharding for throughput-heavy steps.

> **Key constraint:** new concepts are learned **only from features + decomposition** (not from text concept discovery).  
> Step 1 can remain for other tasks, but the ConceptBank is populated from Steps 3–4.

---

## 1) Current pipeline wiring (unchanged entrypoints)

### Key scripts
- Orchestrator: `scripts/run_full_pipeline.py`
- Step 1 (optional): `inference/dataset_inference.py`
- Step 2: `preprocessing/crops_to_json.py`
- Step 3: `src/save_features.py`
- Step 4: `src/analyse_features.py`
- Step 5: `inference/vlm_explainer_multibatch.py`
- Step 6: `eval/concept_deletion_eval.py`
- Optional plotting: `scripts/plot_concept_deletion_eval_token.py`
- Grid creation: `preprocessing/create_grids.py`

### Existing outputs (keep compatible)
- Crops JSON: `OUTPUT_DIR/inference/crops.json`
- Features: `OUTPUT_DIR/features/*.pth`
- Decomposition snapshot: `OUTPUT_DIR/concept/<method>/combined_concept_<method>_raw.pth`
- Explanations JSON: `OUTPUT_DIR/explanations/<method>/vlm_explanations.json`

---

## 2) New capabilities (high-level)

### 2.1 Unified `.env`
All configuration is centralized in `.env` and consumed by:
- `scripts/run_full_pipeline.py`
- new `inference/explain_inference.py` wrapper
- ConceptBank update routines

### 2.2 Explainable inference CLI
A new CLI runs Step 5 explainer **independently** on:
- a single image (`--image`), or
- a directory/list (`--image_root`, `--images_txt`)
and writes **the same** `vlm_explanations.json` schema.

### 2.3 Append-only ConceptBank (SQLite + SQLAlchemy)
A long-lived store of concept vectors:
- Concepts are **vector directions** learned from **feature extraction + decomposition**
- New concepts can be **appended** without modifying existing concepts
- Stored and indexed via SQLite (SQLAlchemy). Vectors are stored either:
  - in external files (recommended default), referenced by SQLite, or
  - as SQLite BLOBs (optional for small banks)

### 2.4 Dry-run + offline tests
Add `DRY_RUN=1` and `LOCAL_FILES_ONLY=1` to ensure:
- no model downloads,
- no GPU required,
- schema-correct JSON outputs produced.

### 2.5 Multi-GPU sharding (optional)
For Step 3/5 throughput (and optionally Step 1), enable multi-process sharding:
- one process per GPU,
- split image lists across workers,
- merge outputs deterministically.

---

## 3) Design principles & decisions

### 3.1 Config precedence (single rule)
**CLI args > env vars > defaults**

### 3.2 Backward compatibility mapping
- `HUGGINGFACE_HUB_TOKEN` → `HF_TOKEN` if `HF_TOKEN` unset  
- `OBJECT_DETECTION=1` → `DETECTOR=langsam` if `DETECTOR` unset  
- `OBJECT_DETECTION=0` → `DETECTOR=rand` if `DETECTOR` unset  
- `PROMPT` → `CONCEPT_DISCOVERY_PROMPT` if unset  
- `VLM_MODEL` fallback:
  - Step 1 uses `CONCEPT_DISCOVERY_MODEL or VLM_MODEL`
  - Steps 3–6 use `CONCEPT_EXTRACTION_VLM or VLM_MODEL`

### 3.3 Append-only concept creation (critical)
Global matrix factorization methods can change existing concepts when new data is added.
To satisfy **“add new concepts without recalculating previous concepts”**, ConceptBank updates use an **append-only updater**:
- Assign new features to existing concepts if similarity ≥ `ASSIGN_THRESHOLD`
- Otherwise create **new concept vectors** from novel residuals/clusters
- Deduplicate against existing concepts (`DUPLICATE_THRESHOLD`)
- Persist only new concepts (existing concepts remain immutable)

> You may still keep Step 4 “snapshot” decomposition (`combined_concept_*_raw.pth`) for research, but the scalable bank update must be append-only.

---

## 4) Unified `.env` variables (Checklist)

### 4.1 Auth / HF cache
- `HF_TOKEN` (optional)
- `HUGGINGFACE_HUB_TOKEN` (optional alias)
- `HF_HOME` (optional)

### 4.2 Models (two-model setup)
- `CONCEPT_DISCOVERY_MODEL` (Step 1 only; optional for ConceptBank)
- `CONCEPT_EXTRACTION_VLM` (Steps 3–6; **required for ConceptBank**)
- `VLM_MODEL` (legacy fallback)

### 4.3 Device / reproducibility
- `DEVICE_ID` (legacy single GPU selection)
- `CUDA_VISIBLE_DEVICES` (optional; if set externally, do not override)
- `SEED`
- `BATCH_SIZE`

### 4.4 Data & output
- `INPUT_DIR`
- `OUTPUT_DIR`
- `IMAGE_ROOT` (optional; default derived from validation mode)

### 4.5 Validation mode
- `VALIDATION_MODE=single|grid`
- `GRID_N` (default `4`)
- `GRID_NUM_GRIDS` (default `40`)
- `GRID_IMAGE_SIZE` (default `512`)

### 4.6 Step 2 detector
- `DETECTOR=rand|langsam|sam2|sam3`  
  - `sam2/sam3` must raise a clear NotImplementedError until implemented.

### 4.7 Step 3 features
- `LAYER_PATH`
- `DATASET_SIZE`

### 4.8 Step 4 decomposition / bank update
- `DECOMP_METHODS` (e.g. `snmf`)
- `NUM_CONCEPTS`
- `CONCEPT_UPDATE_MODE=snapshot|append` (default `append`)

### 4.9 Explainer
- `TOP_N`
- `EXPL_PROMPT_MODE=unsupervised|binary|mcq`
- `EXPL_LABEL`
- `EXPL_CHOICES`
- `EXPL_PROMPT` (explicit override)
- `EXPL_MCQ_JSON` (optional)
- `SAVE_ONLY_GENERATED_TOKENS` (optional)

### 4.10 ConceptBank (SQLite/SQLAlchemy)
- `CONCEPT_BANK_DB` (default: `{OUTPUT_DIR}/concept_bank/concept_bank.sqlite`)
- `CONCEPT_BANK_STORAGE=file|blob` (default `file`)
- `CONCEPT_VECTOR_DIR` (default: `{OUTPUT_DIR}/concept_bank/vectors/`)
- `BANK_TOKEN_TYPE=vision|language|pooled` (default `pooled` or as supported)
- `ASSIGN_THRESHOLD` (default e.g. `0.5`)
- `DUPLICATE_THRESHOLD` (default e.g. `0.9`)
- `MAX_NEW_CONCEPTS_PER_RUN` (optional)

### 4.11 Multi-GPU sharding
- `GPU_MODE=single|shard` (default `single`)
- `GPU_IDS=0,1,...` (optional)
- `NUM_WORKERS` (default: len(GPU_IDS))
- `SHARD_STRATEGY=contiguous|round_robin`

### 4.12 Dry-run/offline
- `DRY_RUN=0|1` (default `0`)
- `LOCAL_FILES_ONLY=0|1` (default `0`)

---

## 5) Required code changes (reuse-first)

### 5.1 Centralize config parsing
- Create `src/config/pipeline_config.py` and move `PipelineConfig` there.
- `scripts/run_full_pipeline.py` imports it.
- `inference/explain_inference.py` imports it.

### 5.2 Extend `preprocessing/crops_to_json.py`
- Add `--detector`
- Map legacy `--object_detection` to `detector`
- Guard `sam2/sam3` with NotImplementedError

### 5.3 Extend `inference/vlm_explainer_multibatch.py`
- Add `--prompt` (override)
- Add `--mcq_json`
- Expose a reusable function `explain_images(...)`
- Add ConceptBank loading support:
  - `--bank_db` + optional `--bank_id`/bank selector
  - fallback: legacy `.pth` concept vectors path

### 5.4 Add `inference/explain_inference.py` (new)
- Thin wrapper that:
  - loads config
  - resolves image list
  - calls `explain_images()`
  - writes schema-compatible JSON

### 5.5 Add ConceptBank module (new): `src/concept_bank/`
**SQLAlchemy models + load/store + updater**
- `models.py`: ORM tables
- `session.py`: engine/session factory
- `store.py`: create bank, insert concepts/vectors
- `load.py`: load vectors into torch/numpy (with storage mode `file|blob`)
- `updater.py`: append-only concept update algorithm
- Optional `index.py`: FAISS index for large banks (future)

### 5.6 Integrate ConceptBank into Step 4 (`src/analyse_features.py`)
When `CONCEPT_UPDATE_MODE=append`:
- Load existing bank for `(model, layer_path, token_type)`
- Use features (from Step 3 outputs) and append-only updater to add new concepts
- Persist only new concepts into SQLite

When `CONCEPT_UPDATE_MODE=snapshot`:
- Keep existing behavior (write `combined_concept_*_raw.pth`)
- Optionally also store snapshot concepts into SQLite under `method="snapshot"` (non-appendable)

### 5.7 Multi-GPU sharding utility (new): `src/utils/sharding.py`
- Split image lists
- Spawn worker processes with per-worker `CUDA_VISIBLE_DEVICES`
- Merge JSON outputs deterministically by `image_path`

### 5.8 Dry-run behavior
Each entrypoint should support `DRY_RUN=1`:
- no model loading
- produce schema-correct dummy outputs

---

## 6) ConceptBank schema (SQLite / SQLAlchemy)

### 6.1 `banks`
- `bank_id` (PK)
- `model_name`
- `layer_path`
- `token_type`
- `feature_dim`
- `created_at`
- `storage_mode` (`file|blob`)
- `vector_dir` / `vectors_path` (optional)
- Unique constraint on `(model_name, layer_path, token_type, feature_dim)`

### 6.2 `concepts`
- `concept_id` (PK, UUID)
- `bank_id` (FK)
- `idx` (int, stable within bank; append assigns next idx)
- `method` (`append_update|snapshot`)
- `status` (`active|deprecated`)
- `meta_json` (optional; stats, thresholds, support)

### 6.3 `concept_vectors`
- `concept_id` (FK/PK)
- Storage:
  - `path` if `file`, OR `blob` if `blob`
- `dtype`, `dim`, `norm`, `checksum`

### 6.4 Optional: `concept_examples`
- `concept_id`
- `image_path`
- `score`

---

## 7) Append-only concept update algorithm (spec)

Inputs:
- `X_new`: new feature vectors (N×D), normalized
- `C_old`: existing concept vectors (K×D), normalized

Steps:
1. Compute similarity `S = X_new @ C_old.T`
2. Assign if `max(S_i) >= ASSIGN_THRESHOLD`
3. Novel pool = vectors with `max(S_i) < ASSIGN_THRESHOLD`
4. Build new concept vectors from novel pool:
   - option A: greedy residual clustering (no sklearn)
   - option B: MiniBatchKMeans (sklearn)
5. Deduplicate candidates against `C_old`:
   - if nearest similarity ≥ `DUPLICATE_THRESHOLD`, drop candidate
6. Persist new concepts/vectors to DB; old concepts remain unchanged

Required invariant:
- Existing concept vectors are **immutable** once stored.

---

## 8) Tests (no GPU, no downloads)

Create `tests/` and add:

### 8.1 Config parsing
- verifies legacy mappings
- verifies derived paths and validation mode behavior

### 8.2 Schema smoke tests (dry-run)
- `crops_to_json --dry_run` produces schema-correct crops JSON
- `vlm_explainer_multibatch --dry_run` produces schema-correct explanations JSON
- `inference/explain_inference.py --dry_run` works on `data/`

### 8.3 ConceptBank tests
- create bank with dummy vectors
- run append update; confirm:
  - old vectors unchanged (byte/array equality)
  - only new concepts inserted
  - idempotence: re-running with same data inserts 0 new concepts

---

## 9) Recommended CLI additions

### 9.1 Update bank (append-only)
Add a small entrypoint (optional but useful):
- `scripts/update_concept_bank.py`
  - runs Step 3 on a concept source directory
  - runs append update into SQLite

### 9.2 Explain inference
- `inference/explain_inference.py` (required)

---

## 10) Definition of Done

- `.env` drives all config; legacy keys still work
- `DETECTOR=rand|langsam` works; `sam2/sam3` guarded
- Explainer supports `--prompt` and `--mcq_json`
- `inference/explain_inference.py` writes the same `vlm_explanations.json` schema
- `DRY_RUN=1` runs on CPU with no downloads
- ConceptBank stored in SQLite + SQLAlchemy; append update adds new concepts without touching existing ones
- Tests pass offline

---

## 11) Example `.env` snippets

### Minimal dry-run
```env
INPUT_DIR=data
OUTPUT_DIR=outputs/run1
DRY_RUN=1
DETECTOR=rand
VLM_MODEL=dummy
VALIDATION_MODE=single
TOP_N=5
EXPL_PROMPT_MODE=unsupervised
CONCEPT_BANK_DB=outputs/run1/concept_bank/concept_bank.sqlite
CONCEPT_UPDATE_MODE=append
CONCEPT_BANK_STORAGE=file
```

### Full run (single GPU)
```env
HF_TOKEN=...
HF_HOME=/scratch/hf
INPUT_DIR=/datasets/myset
OUTPUT_DIR=outputs/run_big
CONCEPT_EXTRACTION_VLM=your-vlm
LAYER_PATH=model.language_model.norm
DETECTOR=langsam
VALIDATION_MODE=grid
GRID_N=4
GRID_NUM_GRIDS=40
GRID_IMAGE_SIZE=512

CONCEPT_BANK_DB=outputs/run_big/concept_bank/concept_bank.sqlite
CONCEPT_BANK_STORAGE=file
CONCEPT_UPDATE_MODE=append
ASSIGN_THRESHOLD=0.55
DUPLICATE_THRESHOLD=0.92
```

### Sharded multi-GPU explain
```env
GPU_MODE=shard
GPU_IDS=0,1
NUM_WORKERS=2
SHARD_STRATEGY=contiguous
```

---

## 12) Notes on Step 1 (concept discovery)
Step 1 is **not required** for ConceptBank creation under this spec.
It can remain for other analyses, but “new concepts” must come from Steps 3–4 + append update.
