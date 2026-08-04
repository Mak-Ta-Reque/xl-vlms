#!/usr/bin/env python3
"""
XL-VLMs Pipeline - Native Python Implementation

This script is a Python conversion of run_full_pipeline_without_coroping.sh
for easier debugging and development in VS Code.

Steps:
1) Dataset inference -> concepts map
2) Build crops JSON from concept→image mapping
3) (Optional, LOGIT_LENS_LAYER_SELECTION=1) Per-tag layer selection via
   logit lens -> logitlens/selected_layers.json
4) Generate features from crops JSON (on-the-fly cropping, hooks the
   per-tag selected layer when step 3 ran)
5) Decompose features (one or more methods)
6) Run VLM explainer per method
7) Concept deletion eval per method
8) (Optional) Plots per method + summary

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --input-dir /path/to/data --output-dir /path/to/output
"""

import os
import sys
import argparse
import subprocess
import shutil
import logging
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

try:
    from dotenv import load_dotenv
except ImportError as err:
    raise SystemExit(
        "python-dotenv is required to load the .env configuration: "
        "pip install python-dotenv"
    ) from err

from src.helpers.utils import resolve_layer_path


# =============================================================================
# Configuration
# =============================================================================

class PipelineConfig:
    """Configuration class that loads from .env file and CLI arguments."""
    
    def __init__(self):
        # Load .env file
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        self.root_dir = ROOT_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data & outputs
        # input_dir: Contains training images for object detection (step 2) and feature generation (step 3)
        self.input_dir = self._get_path("INPUT_DIR", ROOT_DIR / "data")
        self.output_dir = self._get_path("OUTPUT_DIR", ROOT_DIR / f"outputs/run_{self.timestamp}")
        self.hf_home = self._get_path("HF_HOME", Path.home() / ".cache/huggingface")
        # image_root: Contains validation images for VLM explainer (step 5) - separate from input_dir
        self.image_root = self._get_path("IMAGE_ROOT", self.input_dir / "grids")
        
        # Model/runtime
        self.vlm_model = self._get_str("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.batch_size = self._get_int("BATCH_SIZE", 48)
        self.seed = self._get_int("SEED", 42)
        # Device selection: DEVICE ("cuda:1", "cuda:0,2", "auto", "cpu");
        # DEVICE_ID=<n> is honored as a fallback alias when DEVICE is unset.
        self.device = self._get_str("DEVICE", "").strip()
        if not self.device:
            device_id = self._get_str("DEVICE_ID", "").strip()
            self.device = f"cuda:{device_id}" if device_id else "cuda:0"
        
        # Crops JSON generation
        self.masks_per_image = self._get_int("MASKS_PER_IMAGE", 5)
        self.concept_masks_per_image = self._get_int("CONCEPT_MASKS_PER_IMAGE", 1)
        self.patch_size = self._get_int("PATCH_SIZE", 200)
        self.min_images_per_tag = self._get_int("MIN_IMAGES_PER_TAG", 10)
        self.max_images_per_tag = self._get_int("MAX_IMAGES_PER_TAG", 128)
        self.crop_mode = self._get_str("CROP_MODE", "random").strip().lower()
        if self.crop_mode in {"random", "sliding_window"}:
            self.object_detector = "none"
        elif self.crop_mode in {"sam3", "langsam", "semanticsegments_sam3"}:
            # semanticsegments_sam3 uses SAM3 auto masks in step 2, but the
            # downstream feature saver decides whether to treat them as fg-only
            # (POSITIVE_NEGATIVE_SEGMENT=0) or to keep fg/bg pairing.
            self.object_detector = self.crop_mode
            if self.crop_mode == "semanticsegments_sam3":
                self.object_detector = "sam3"
        elif self.crop_mode == "none":
            # No real cropping: step 2 emits one whole-image bbox per (tag,
            # image) pair instead of running a detector, so steps 3-8 stay
            # completely unchanged (they only ever consume crops.json).
            self.object_detector = "none"
        else:
            raise ValueError(
                "Invalid CROP_MODE. Use one of: random, sliding_window, langsam, sam3, semanticsegments_sam3, none"
            )
        self.detection_batch_size = self._get_int("DETECTION_BATCH_SIZE", 2)
        self.mask_context_pixels = self._get_int("MASK_CONTEXT_PIXELS", 0)
        # Positive/Negative binary segmentation (1=fg/bg only, 0=multi-mask)
        self.positive_negative_segment = int(
            os.environ.get(
                "POSITIVE_NEGATIVE_SEGMENT",
                os.environ.get("POSITIVE_NEGATVE_SEGMENT", "0"),
            )
        )
        # Inference prompt and image preprocessing
        self.prompt = self._get_str(
            "PROMPT",
            "Identify every visible object, item, concept, and pattern in the image at the most fine-grained level. Output only single words in a strict comma-separated list, no sentences or explanations."
        )
        self.prompt_template = self._get_str("PROMPT_TEMPLATE", "cgdl")
        # Decomposition strategy is an independent axis from PROMPT_TEMPLATE --
        # any template can use either:
        #   per_tag: one SNMF call per tag (DECOMP_COMPONENTS concepts/tag),
        #            merged + CLEAN_EXAMPLE_RATIO-purity-filtered by
        #            src/combine_concepts.py. Meaningful whenever each tag's
        #            features carry a real per-tag positive/negative signal
        #            to filter on.
        #   pooled:  every tag's features concatenated once (src/combine_features.py)
        #            into a single matrix, one SNMF call for the whole dataset
        #            (DECOMP_COMPONENTS_GLOBAL concepts total), no purity filter
        #            (every direction with >=1 assigned sample is accepted).
        # Defaults preserve each template's original behavior when
        # DECOMP_STRATEGY isn't set explicitly: cgdl -> per_tag,
        # non_contrastive/null -> pooled. Set DECOMP_STRATEGY=per_tag/pooled
        # to override for any template (e.g. cgdl+pooled, non_contrastive+per_tag).
        self.decomp_strategy = self._get_str(
            "DECOMP_STRATEGY",
            "per_tag" if self.prompt_template == "cgdl" else "pooled",
        ).strip().lower()
        if self.decomp_strategy not in ("per_tag", "pooled"):
            raise ValueError(f"DECOMP_STRATEGY must be 'per_tag' or 'pooled', got {self.decomp_strategy!r}")
        # Use IMAGE_SIZE_WIDTH as the reference; height is derived per-image via aspect ratio.
        self.image_size_width = self._get_int("IMAGE_SIZE_WIDTH", 512)
        # Keep for backward compatibility / debugging; not used as a fixed resize target.
        self.image_size_height = self._get_int("IMAGE_SIZE_HEIGHT", 512)
        self.image_size = (self.image_size_width, self.image_size_height)
        self.image_budget = self._get_int("IMAGE_BUDGET", 300)# reduce for test , use 200 -> for a better run
        self.segmentation_confidence = self._get_float("SEGMENTATION_CONFIDENCE", 0.5)
        
        # Decomposition methods
        self.decomp_methods = self._get_str("DECOMP_METHODS", "snmf").split(",")
        self.decomp_components = self._get_int("DECOMP_COMPONENTS", 2)
        # Concept count for the non-contrastive/null global decomposition path
        # (step 5): these prompt templates don't produce a per-tag contrastive
        # fg/bg split, so all tags' features are pooled first and decomposed
        # once, budget-matched to roughly decomp_components * num_tags.
        self.decomp_components_global = self._get_int("DECOMP_COMPONENTS_GLOBAL", 20)
        # Sparsity regularization: higher = sparser decomposition.
        # DL_ALPHA feeds SNMF/dictionary-learning methods (snmf, nndl); it is
        # sklearn DictionaryLearning's L1 `alpha` on the sparse code (default 20).
        # SAE_SPARSITY_LAMBDA feeds sae/sae2; it is the L1 penalty weight on the
        # SAE hidden code (default 0.0005). Try e.g. 1.0 for a much sparser result.
        self.dl_alpha = self._get_float("DL_ALPHA", 20.0)
        self.sae_sparsity_lambda = self._get_float("SAE_SPARSITY_LAMBDA", 0.0005)
        # Hard top-k activation for sae/sae2: guarantees this exact zero-fraction
        # in the code (e.g. 0.99), unlike SAE_SPARSITY_LAMBDA which plateaus well
        # short of high targets. Empty/unset disables it (soft L1 penalty only).
        _sae_target_sparsity = self._get_str("SAE_TARGET_SPARSITY", "").strip()
        self.sae_target_sparsity = float(_sae_target_sparsity) if _sae_target_sparsity else None
        self.num_concept = self._get_int("NUM_CONCEPT", -1)  # -1 = use all tags, else top N tags by crop count
        # Optional concept vocab filter. Unset/empty (e.g. commented out in
        # .env) => None => NUM_CONCEPT picks the top-N most frequent tags
        # without any vocab filtering.
        _vocab = self._get_str("CONCEPTS_VOCAB", "").strip()
        self.concepts_vocab = Path(_vocab) if _vocab else None
        self.dataset_size = self._get_int("BAG_SIZE", 100)
        self.delete_intermediate_files = self._get_int("DELETE_INTERMEDIATE_FILES", 0) == 1
        
        # Explainer/Eval
        self.layer_path = resolve_layer_path(
            self._get_str("LAYER_PATH", "model.language_model.norm")
        )
        # Any template whose response can run several tokens has the same
        # mean-pooling problem: averaging over all generated tokens dilutes
        # the signal with filler/punctuation and biases toward whatever token
        # the response happens to end on, rather than the token that actually
        # names the concept. save_hidden_states_for_token_of_interest instead
        # extracts the hidden state at the tag word itself wherever it occurs
        # in the response (falls back gracefully, with a found/not-found
        # mask, when the tag word never appears -- see
        # extract_token_of_interest_states). cgdl's forced "X or No X" answer
        # and non_contrastive's open caption both have a real tag word to find
        # this way; null's empty prompt + MAX_NEW_TOKENS=1 has no multi-token
        # response to disambiguate in the first place, so it stays on mean
        # (equivalent to the single token either way).
        _default_hook_names = (
            "save_hidden_states_mean"
            if self.prompt_template == "null"
            else "save_hidden_states_for_token_of_interest"
        )
        self.hook_names = self._get_str("HOOK_NAMES", _default_hook_names)
        # Per-tag layer selection via logit lens (step 3). When enabled, each
        # tag's extraction layer is picked by a logit-lens sweep and LAYER_PATH
        # is only the fallback; values flow to subprocesses via the environment.
        self.logit_lens_layer_selection = self._get_int("LOGIT_LENS_LAYER_SELECTION", 0)
        self.logit_lens_mode = self._get_str("LOGIT_LENS_MODE", "patch")
        self.logit_lens_layers = self._get_str("LOGIT_LENS_LAYERS", "auto")
        self.logit_lens_num_patches = self._get_int("LOGIT_LENS_NUM_PATCHES", 8)
        

        self.top_n = self._get_int("TOP_N", 5)
        self.num_most_activating_samples = self._get_int("NUM_MOST_ACTIVATING_SAMPLES", 10)
        # Tokens generated during feature extraction (steps 3/4) -- default 50
        # matches save_features.py/select_layers.py's own CLI default (both
        # previously received no --max_new_tokens flag at all). Set to 1 for
        # a single-forward-pass extraction (e.g. with PROMPT_TEMPLATE=null),
        # since --generation_mode with max_new_tokens=1 does exactly one
        # decoder forward pass without needing a separate teacher-forcing path.
        self.max_new_tokens = self._get_int("MAX_NEW_TOKENS", 50)
        self.num_points = self._get_int("NUM_POINTS", 70)
        self.expl_prompt_mode = self._get_str("EXPL_PROMPT_MODE", "unsupervised")
        self.expl_label = self._get_str("EXPL_LABEL", "")
        self.expl_choices = self._get_str("EXPL_CHOICES", "")
        # Set when IMAGE_ROOT points at single-object crops rather than
        # multi-cell grids -- adjusts the explainer's mcq/unsupervised prompt
        # wording so it doesn't ask about non-existent grid cells.
        self.single_object = os.environ.get("SINGLE_OBJECT", "0") == "1"
        # Rebuttal P_bin_shuf control: substitute a different, reproducibly
        # sampled concept into the [concept] prompt placeholder instead of
        # the crop's true tag (see JSONDataset.create_dataset()). Only
        # affects step 4's prompt text -- concept-bank bucketing is untouched.
        self.shuffle_concept_prompt = os.environ.get("SHUFFLE_CONCEPT_PROMPT", "0") == "1"
        self.shuffle_concept_vocab = self._get_str("SHUFFLE_CONCEPT_VOCAB", "")
        # Images the explainer takes from IMAGE_ROOT:
        # -1 = all, 0 = skip explanation (and eval/plots), N = first N.
        self.expl_max_images = self._get_int("EXPL_MAX_IMAGES", -1)
        # Positive/negative concept split threshold; embedded in the bank
        # file names (…_cr<ratio>_raw.pth) so runs with different ratios can
        # be evaluated side by side.
        self.clean_example_ratio = self._get_float("CLEAN_EXAMPLE_RATIO", 0.8)
        # Purity filtering is tied to DECOMP_STRATEGY, not PROMPT_TEMPLATE:
        # per_tag decomposition gives each tag's own positive/negative split a
        # real ratio to filter on (regardless of which template produced it),
        # while pooled decomposition has no such per-tag split to check --
        # every sample's hidden state is pooled together unfiltered (see
        # keep_only_token_of_interest in src/analysis/__init__.py), so
        # constraining concept selection to a CLEAN_EXAMPLE_RATIO threshold
        # meant for a per-tag split isn't meaningful there. Accept every SNMF
        # direction with >=1 assigned sample (MIN_DIRECTION_ASSIGNED) instead.
        # Written back to the environment so the combine_concepts.py
        # subprocess (which re-reads CLEAN_EXAMPLE_RATIO itself) and this
        # process's own ratio_tag-based file paths agree.
        if self.decomp_strategy == "pooled":
            self.clean_example_ratio = 0.0
            os.environ["CLEAN_EXAMPLE_RATIO"] = "0.0"
        self.ratio_tag = f"cr{self.clean_example_ratio:g}"
        
        # Plot
        self.plot_ymin = self._get_float("PLOT_YMIN", 6.55e-6)
        self.plot_ymax = self._get_float("PLOT_YMAX", 7.10e-6)
        
        # Derived paths
        self._setup_derived_paths()
    
    def _expand_path(self, value: str) -> str:
        """Expand {ROOT_DIR} placeholder in paths."""
        return value.replace("{ROOT_DIR}", str(self.root_dir))
    
    def _get_str(self, key: str, default: str) -> str:
        value = os.environ.get(key, default)
        return self._expand_path(value) if "{ROOT_DIR}" in value else value
    
    def _get_int(self, key: str, default: int) -> int:
        value = os.environ.get(key)
        if value is None or value == "":
            return default
        return int(value)
    
    def _get_float(self, key: str, default: float) -> float:
        value = os.environ.get(key)
        if value is None or value == "":
            return default
        return float(value)

    def _get_path(self, key: str, default: Path) -> Path:
        value = os.environ.get(key)
        if value is None or value == "":
            return default
        expanded = self._expand_path(value)
        return Path(expanded)
    
    def _setup_derived_paths(self):
        """Setup derived paths based on output directory."""
        self.concept_map_json = self.output_dir / "inference" / "concepts_to_images.json"
        self.active_concept_map_json = self.concept_map_json
        self.crops_json = self.output_dir / "inference" / "crops.json"
        self.objects_csv = self.output_dir / "inference" / "objects.csv"
        self.logitlens_dir = self.output_dir / "logitlens"
        self.selected_layers_json = self.logitlens_dir / "selected_layers.json"
        self.features_dir = self.output_dir
        self.decomp_dir = self.output_dir / "concept"
        self.explain_dir = self.output_dir / "explanations"
        self.eval_dir = self.output_dir / "eval"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
    
    def update_from_args(self, args: argparse.Namespace):
        """Update config from CLI arguments."""
        if args.input_dir:
            self.input_dir = Path(args.input_dir)
        if args.output_dir:
            self.output_dir = Path(args.output_dir)
            self._setup_derived_paths()
        if args.decomp:
            self.decomp_methods = args.decomp.split(",")
        if args.plot_ymin:
            self.plot_ymin = args.plot_ymin
        if args.plot_ymax:
            self.plot_ymax = args.plot_ymax


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(logs_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("xl-vlms-pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate output when setup_logging is called multiple times.
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(console_fmt)
    logger.addHandler(console)
    
    # File handler
    log_file = logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# GPU memory isolation helper
# =============================================================================

def _cleanup_gpu():
    """Best-effort in-process GPU memory cleanup."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _run_python_subprocess(script_args: List[str], env_overrides: dict = None,
                           logger: logging.Logger = None) -> None:
    """Run a Python module/script in a fresh subprocess for full GPU memory isolation.
    
    This prevents CUDA OOM when consecutive pipeline steps each need the full
    GPU memory (e.g. Step 3 generate features → Step 4 decompose features).
    Captures stdout/stderr and relays through the pipeline logger so that
    combine_concepts skip summaries (and other subprocess output) appear in logs.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)

    # -u: unbuffered child stdout, so output streams live instead of arriving
    # in one block when the step finishes.
    cmd = [sys.executable, "-u"] + script_args
    if logger:
        logger.debug(f"Subprocess: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, env=env, cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Relay subprocess output through the logger as it is produced
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if logger:
            logger.info(f"  [sub] {line}")
        else:
            print(line)

    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {returncode}): {' '.join(cmd)}"
        )


def _delete_downstream_outputs(config: PipelineConfig, step_num: int, logger: logging.Logger) -> None:
    """Delete output files of all downstream steps to force re-computation.
    
    When a step is re-computed, delete outputs of all subsequent steps so they
    will also re-compute to adapt to the changes.
    """
    downstream_deletions = []
    
    if step_num <= 1:
        # If step 1 reruns, delete everything after it
        downstream_deletions = [
            (config.crops_json, "Crops JSON"),
            (config.logitlens_dir, "Logit-lens layer selection"),
            (config.features_dir / "features", "Features"),
            (config.decomp_dir, "Decomposed concepts"),
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 2:
        # If step 2 reruns, delete outputs from step 3 onwards
        downstream_deletions = [
            (config.logitlens_dir, "Logit-lens layer selection"),
            (config.features_dir / "features", "Features"),
            (config.decomp_dir, "Decomposed concepts"),
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 3:
        # If step 3 (layer selection) reruns, delete outputs from step 4 onwards
        downstream_deletions = [
            (config.features_dir / "features", "Features"),
            (config.decomp_dir, "Decomposed concepts"),
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 4:
        # If step 4 (features) reruns, delete outputs from step 5 onwards
        downstream_deletions = [
            (config.decomp_dir, "Decomposed concepts"),
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 5:
        # If step 5 (decompose) reruns, delete outputs from step 6 onwards
        downstream_deletions = [
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 6:
        # If step 6 (explainer) reruns, delete outputs from step 7 onwards
        downstream_deletions = [
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 7:
        # If step 7 (eval) reruns, delete outputs from step 8 (plots)
        downstream_deletions = [
            (config.plots_dir, "Plots"),
        ]
    # Step 8 has no downstream dependencies
    
    if downstream_deletions:
        logger.info(f"Step {step_num} was re-computed; cascade deleting downstream outputs...")
        for path, label in downstream_deletions:
            if path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink()
                    logger.info(f"  Deleted: {label} ({path})")
                except Exception as e:
                    logger.warning(f"  Could not delete {label}: {e}")


def _ensure_top_concept_map(
    concept_map_json: Path,
    num_concept: int,
    logger: logging.Logger,
    concepts_vocab: Optional[Path] = None,
) -> Path:
    """Create and return a filtered concept map containing vocab-filtered top-N concepts.

    Delegates to the shared module so the selection logic has a single home.
    """
    from preprocessing.select_top_concepts import select_top_concepts

    result = select_top_concepts(concept_map_json, num_concept, concepts_vocab)
    if result != concept_map_json:
        logger.info(f"Using filtered concept map: {result}")
    return result


def _count_concepts_in_json(concept_map_json: Path) -> int:
    if not concept_map_json.exists():
        return 0
    with open(concept_map_json, "r", encoding="utf-8") as handle:
        concept_mapping = json.load(handle)
    if isinstance(concept_mapping, dict):
        return len(concept_mapping)
    return 0


# =============================================================================
# Pipeline Steps
# =============================================================================

def create_directories(config: PipelineConfig, logger: logging.Logger):
    """Create all necessary output directories."""
    dirs = [
        config.logs_dir,
        config.output_dir / "inference",
        config.output_dir / "features",
        config.output_dir / "concept",
        config.output_dir / "explanations",
        config.output_dir / "eval",
        config.output_dir / "plots",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directories under {config.output_dir}")


def step_1_dataset_inference(config: PipelineConfig, logger: logging.Logger):
    """Step 1: Dataset inference -> concept map."""

    config.objects_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if both CSV and concept map exist before skipping
    csv_exists = config.objects_csv.exists()
    concept_map_exists = config.concept_map_json.exists()

    if csv_exists and concept_map_exists:
        logger.info(f"Skip Step 1 (found {config.objects_csv} and {config.concept_map_json})")
        # Still resolve the filtered map so later steps log/count correctly
        config.active_concept_map_json = _ensure_top_concept_map(
            config.concept_map_json,
            config.num_concept,
            logger,
            concepts_vocab=config.concepts_vocab,
        )
        return
    
    # Step will re-compute; cascade delete downstream outputs
    _delete_downstream_outputs(config, 1, logger)
    
    if not csv_exists:
        # Import and run dataset_inference
        logger.info("START: Dataset Inference")
        
        sys.path.insert(0, str(ROOT_DIR / "inference"))
        from inference.dataset_inference import main as dataset_inference_main
        
        # Prepare arguments as if from command line
        inference_args = [
            "--dataset_path", str(config.input_dir),
            "--model_name", config.vlm_model,
            "--output_csv", str(config.objects_csv),
            "--prompt", config.prompt,
            "--batch_size", str(config.batch_size),
            "--image_size_width", str(config.image_size_width),
            "--image_budget", str(config.image_budget),
            "--device", config.device,
            "--trust_remote_code",
        ]
        
        # Parse args and run
        original_argv = sys.argv
        sys.argv = ["dataset_inference.py"] + inference_args
        try:
            dataset_inference_main()
        finally:
            sys.argv = original_argv
        
        logger.info("DONE: Dataset Inference")
    else:
        logger.info(f"Reusing existing CSV: {config.objects_csv}")
    
    # Build concept map
    if not concept_map_exists:
        logger.info("START: Build Concept Map")
        
        from concept_image_mapping import main as concept_mapping_main
        
        mapping_args = [
            "--input", str(config.objects_csv),
            "--output", str(config.concept_map_json),
        ]
        
        original_argv = sys.argv
        sys.argv = ["concept_image_mapping.py"] + mapping_args
        try:
            concept_mapping_main()
        finally:
            sys.argv = original_argv
        
        logger.info("DONE: Build Concept Map")
    else:
        logger.info(f"Reusing existing concept map: {config.concept_map_json}")

    config.active_concept_map_json = _ensure_top_concept_map(
        config.concept_map_json,
        config.num_concept,
        logger,
        concepts_vocab=config.concepts_vocab,
    )


def _build_whole_image_crops_json(config: PipelineConfig, logger: logging.Logger, concept_map_json: Path) -> None:
    """CROP_MODE=none variant of step 2: instead of running a detector,
    emit one crops.json entry per (tag, image) with bbox = the full image
    extent. Steps 3-8 need no changes since they only ever consume
    crops.json in this same schema (bbox-only records, RLE off, are already
    a supported variant — see RLE=0 in .env.example).
    """
    import random
    from PIL import Image
    from preprocessing.crops_to_json import load_mapping

    if config.seed is not None:
        random.seed(config.seed)

    mapping = load_mapping(str(concept_map_json))

    valid_tags = []
    for tag, rels in mapping.items():
        if not isinstance(rels, list) or len(rels) < config.min_images_per_tag:
            continue
        if config.max_images_per_tag > 0 and len(rels) > config.max_images_per_tag:
            rels = random.sample(rels, config.max_images_per_tag)
        valid_tags.append((tag, rels))

    if not valid_tags:
        raise RuntimeError(
            f"No concept tag survived filtering: all {len(mapping)} tags in "
            f"'{concept_map_json}' have fewer than min_images_per_tag="
            f"{config.min_images_per_tag} images."
        )

    total_images = sum(len(rels) for _, rels in valid_tags)
    logger.info(f"Building whole-image crops.json: {total_images} images across {len(valid_tags)} tags")

    result = {}
    skipped = 0
    for tag, rels in valid_tags:
        tag_entry = {}
        for rel in rels:
            image_path = config.input_dir / rel
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception as e:
                logger.warning(f"  Skipping unreadable image {image_path}: {e}")
                skipped += 1
                continue
            tag_entry[rel] = {
                "masks_rle": [{"bbox": [0, 0, width, height], "is_concept": True}]
            }
        if tag_entry:
            result[tag] = tag_entry

    if not result:
        raise RuntimeError("Whole-image crops.json build produced zero entries (all images unreadable?).")

    config.crops_json.parent.mkdir(parents=True, exist_ok=True)
    with open(config.crops_json, "w", encoding="utf-8") as f:
        json.dump(result, f)
    logger.info(f"Wrote whole-image crops.json: {config.crops_json} ({total_images - skipped} images, {skipped} skipped)")


def step_2_build_crops_json(config: PipelineConfig, logger: logging.Logger):
    """Step 2: Build crops JSON from concept→image map.

    Runs as a subprocess to isolate detector GPU memory (skipped for
    CROP_MODE=none, which builds crops.json in-process — no detector to load).
    """

    if config.crops_json.exists():
        # A crops.json without any concept entries (e.g. from a run where
        # every tag was filtered out) must not be reused — it would make
        # every downstream step fail with "0 concept splits".
        try:
            with open(config.crops_json, "r", encoding="utf-8") as handle:
                _crops = json.load(handle)
        except Exception:
            _crops = None
        if isinstance(_crops, dict) and len(_crops) > 0:
            logger.info(f"Skip Step 2 (found {config.crops_json})")
            return
        logger.warning(
            f"Existing {config.crops_json} is empty or unreadable — rebuilding it."
        )

    # Step will re-compute; cascade delete downstream outputs
    _delete_downstream_outputs(config, 2, logger)
    
    logger.info("START: Crops JSON")
    config.crops_json.parent.mkdir(parents=True, exist_ok=True)
    concept_map_json = _ensure_top_concept_map(
        config.concept_map_json,
        config.num_concept,
        logger,
        concepts_vocab=config.concepts_vocab,
    )

    if config.crop_mode == "none":
        _build_whole_image_crops_json(config, logger, concept_map_json)
        logger.info("DONE: Crops JSON (whole-image, no cropping)")
        return

    crops_args = [
        str(ROOT_DIR / "preprocessing" / "crops_to_json.py"),
        "--mapping_json", str(concept_map_json),
        "--image_root", str(config.input_dir),
        "--output_json", str(config.crops_json),
        "--detector", config.object_detector,
        "--masks_per_image", str(config.masks_per_image),
        "--concept_masks_per_image", str(config.concept_masks_per_image),
        "--min_images_per_tag", str(config.min_images_per_tag),
        "--max_images_per_tag", str(config.max_images_per_tag),
        "--patch_size", str(config.patch_size),
        "--batch_size", str(config.detection_batch_size),
        "--image_size_width", str(config.image_size_width),
        "--device", config.device,
        "--seed", str(config.seed),
        "--confidence_threshold", str(config.segmentation_confidence),
        "--positive_negative_segment", str(config.positive_negative_segment),
    ]

    _run_python_subprocess(crops_args, logger=logger)

    logger.info("DONE: Crops JSON")


def step_3_select_layers(config: PipelineConfig, logger: logging.Logger):
    """Step 3: Per-tag layer selection via logit lens (optional).

    Runs only when LOGIT_LENS_LAYER_SELECTION=1. For each concept tag in
    crops.json, sweeps decoder layers on sampled tag regions and writes the
    winning layer per tag to logitlens/selected_layers.json, which step 4
    uses to hook the right layer during feature extraction.

    Skipped whenever DECOMP_STRATEGY=pooled, for ANY prompt template: pooled
    decomposition concatenates every tag's features and decomposes once
    (step 5), which requires every tag's hidden states to live under the SAME
    module key. A per-tag layer would break that (and does: mixed module keys
    after pooling fail step 5's fixed --module_to_decompose). per_tag
    decomposition has no such constraint -- each tag is decomposed
    separately, so a different layer per tag is fine -- which is why this is
    keyed off decomp_strategy, not prompt_template: it's universal across
    templates for a given strategy, not special-cased per template.

    Runs as a subprocess to isolate VLM GPU memory.
    """
    if not config.logit_lens_layer_selection:
        logger.info("Skip Step 3 (LOGIT_LENS_LAYER_SELECTION=0)")
        return
    if config.decomp_strategy == "pooled":
        logger.info(f"Skip Step 3 (decomp_strategy=pooled uses a uniform layer, not per-tag selection; prompt_template={config.prompt_template})")
        return

    if config.selected_layers_json.exists():
        logger.info(f"Skip Select Layers (found {config.selected_layers_json})")
        return

    # Step will re-compute; cascade delete downstream outputs
    _delete_downstream_outputs(config, 3, logger)

    logger.info("START: Select Layers (logit lens)")

    select_args = [
        str(ROOT_DIR / "src" / "select_layers.py"),
        "--model_name", config.vlm_model,
        "--dataset_name", "json_crop_map",
        "--dataset_size", str(config.dataset_size),
        "--data_dir", str(config.input_dir),
        "--annotation_file", str(config.crops_json),
        "--split", "train",
        "--prompt_template", config.prompt_template,
        "--save_dir", str(config.features_dir),
        "--batch_size", "1",
        "--generation_mode",
        "--max_new_tokens", str(config.max_new_tokens),
    ]

    _run_python_subprocess(select_args, logger=logger)

    logger.info("DONE: Select Layers (logit lens)")


def step_4_generate_features(config: PipelineConfig, logger: logging.Logger):
    """Step 4: Generate features from crops JSON.

    Runs as a subprocess to isolate VLM GPU memory from Step 5. Works
    unchanged for CROP_MODE=none too, since crops.json already has one
    whole-image bbox per (tag, image) from step 2 in that mode.
    """

    features_path = config.features_dir / "features"
    debug_save_enabled = os.environ.get("DEBUG_SAVE_VLM_INPUTS", "0") == "1"
    selected_concept_count = _count_concepts_in_json(config.active_concept_map_json)
    existing_pth_count = len(list(features_path.glob("*.pth"))) if features_path.exists() else 0
    # Require the full expected tag count, not just "at least one file" --
    # a config interrupted mid-loop (e.g. a killed subprocess) can leave
    # exactly one tag's .pth file on disk, which the old any(...) check
    # treated as fully done, silently decomposing/evaluating on 1 of N tags
    # for the rest of the pipeline with no error raised. Confirmed this
    # happened in practice: a killed run left only one tag's feature file,
    # and every step downstream skip-checked past it to a "successful"
    # completion built on 10% of the intended data.
    features_complete = (
        features_path.exists()
        and existing_pth_count > 0
        and (selected_concept_count <= 0 or existing_pth_count >= selected_concept_count)
    )
    if features_complete:
        if debug_save_enabled:
            logger.info(
                f"Debug mode enabled (DEBUG_SAVE_VLM_INPUTS=1): forcing feature rerun even though features exist under {features_path}"
            )
            # Cascade delete downstream outputs
            _delete_downstream_outputs(config, 4, logger)
        else:
            logger.info(f"Skip Generate Features (found {existing_pth_count} feature files under {features_path})")
            return
    else:
        if existing_pth_count > 0:
            logger.warning(
                f"Found {existing_pth_count} feature file(s) under {features_path} but expected "
                f"{selected_concept_count} (from the filtered concept map) -- treating as incomplete "
                "(likely an interrupted prior run) and regenerating from scratch."
            )
        # Step will re-compute; cascade delete downstream outputs
        _delete_downstream_outputs(config, 4, logger)
        if existing_pth_count > 0:
            import shutil
            shutil.rmtree(features_path)

    logger.info("START: Generate Features")
    annotation_file = config.crops_json
    if selected_concept_count > 0:
        logger.info(f"Using top {selected_concept_count} concepts from the filtered concept map for feature generation")
    
    features_args = [
        str(ROOT_DIR / "src" / "save_features.py"),
        "--model_name", config.vlm_model,
        "--dataset_name", "json_crop_map",
        "--dataset_size", str(config.dataset_size),
        "--data_dir", str(config.input_dir),
        "--annotation_file", str(annotation_file),
        "--split", "train",
        "--hook_names", config.hook_names,
        "--modules_to_hook", config.layer_path,
        "--prompt_template", config.prompt_template,
        "--save_dir", str(config.features_dir),
        "--batch_size", str(config.batch_size),
        "--generation_mode",
        "--max_new_tokens", str(config.max_new_tokens),
        "--save_only_generated_tokens",
        "--exact_match_modules_to_hook",
    ]
    if config.shuffle_concept_prompt and config.shuffle_concept_vocab:
        features_args.append("--shuffle_concept_prompt")
        features_args.extend(["--shuffle_concept_vocab", config.shuffle_concept_vocab])

    # Force off regardless of the .env value: pooled decomposition (any
    # template) concatenates every tag's features and decomposes once against
    # one fixed module path, so save_features.py's per-tag layer-selection
    # fallback (which would hook a different layer per tag) must not run.
    step4_env_overrides = None
    if config.decomp_strategy == "pooled":
        step4_env_overrides = {"LOGIT_LENS_LAYER_SELECTION": "0"}

    _run_python_subprocess(features_args, env_overrides=step4_env_overrides, logger=logger)

    logger.info("DONE: Generate Features")


def step_5_decompose_features(config: PipelineConfig, logger: logging.Logger):
    """Step 5: Decompose features across methods.

    Each analyse_features call runs as a subprocess to isolate GPU memory,
    since loading the model's lm_head requires significant VRAM. Works
    unchanged for CROP_MODE=none too — features_dir/"features" still holds
    one .pth per tag (built from whole-image crops in step 4).

    DECOMP_STRATEGY selects which of the two happens, independent of
    PROMPT_TEMPLATE: per_tag decomposes each tag's features separately (below)
    and merges; pooled concatenates every tag's features into one matrix
    first and decomposes once, with each resulting atom used directly as a
    concept -- see combine_features.py.
    """
    use_per_tag_decomp = config.decomp_strategy == "per_tag"

    step_4_recomputed = False  # Track if any method was re-computed

    for method in config.decomp_methods:
        method = method.strip()
        method_dir = config.decomp_dir / method
        out_raw = method_dir / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        
        if out_raw.exists():
            logger.info(f"Skip Decompose ({method}) (found {out_raw})")
            continue
        
        # Mark that step 4 is being re-computed for this method
        step_4_recomputed = True
        
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyse features (subprocess — loads model for lm_head)
        base_analysis_name = "decompose_activations_text_grounding_image_grounding"
        intermediate_dir = method_dir / f"intermediate_{method}"
        
        logger.info(f"START: Decompose:{method} (batch)")

        features_dir = config.features_dir / "features"
        if use_per_tag_decomp:
            decompose_features_path = str(features_dir)
            decompose_num_concepts = config.decomp_components
        else:
            # Pool every tag's features into a single combined_features.pth
            # (once) so analyse_features.py takes its single-matrix path
            # instead of looping per tag.
            combined_features_path = features_dir / "combined_features.pth"
            if not combined_features_path.exists():
                _run_python_subprocess(
                    [str(ROOT_DIR / "src" / "combine_features.py"), str(features_dir)],
                    logger=logger,
                )
            decompose_features_path = str(combined_features_path)
            decompose_num_concepts = config.decomp_components_global

        analyse_args = [
            str(ROOT_DIR / "src" / "analyse_features.py"),
            "--model_name", config.vlm_model,
            "--analysis_name", f"{base_analysis_name}_{method}",
            "--features_path", decompose_features_path,
            "--module_to_decompose", config.layer_path,
            "--num_concepts", str(decompose_num_concepts),
            "--decomposition_method", method,
            "--num_most_activating_samples", str(config.num_most_activating_samples),
            "--save_dir", str(intermediate_dir),
            "--dl_alpha", str(config.dl_alpha),
            "--sae_sparsity_lambda", str(config.sae_sparsity_lambda),
        ]
        if config.sae_target_sparsity is not None:
            analyse_args += ["--sae_target_sparsity", str(config.sae_target_sparsity)]
        
        _run_python_subprocess(analyse_args, logger=logger)
        
        logger.info(f"DONE: Decompose:{method} (batch)")
        
        # Combine concepts (lightweight, no GPU — runs in-process)
        logger.info(f"START: Combine Concepts ({method})")
        
        combine_cli_args = [
            str(ROOT_DIR / "src" / "combine_concepts.py"),
            "--input_dir", str(intermediate_dir),
            "--output_path", str(method_dir / f"combined_concept_{method}.pth"),  # combine appends _cr<ratio>
            "--normalization", "gl",
        ]
        if config.delete_intermediate_files:
            combine_cli_args.append("--delete")
        
        _run_python_subprocess(combine_cli_args, logger=logger)

        logger.info(f"DONE: Combine Concepts ({method})")

        # Fail fast if the combined bank is empty — regrounding, explainer
        # and eval all need at least one concept.
        raw_bank_path = method_dir / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        try:
            import torch as _torch
            _bank = _torch.load(raw_bank_path, map_location="cpu")
            _n_concepts = int(_bank.get("concepts", _torch.empty(0)).shape[0]) if hasattr(
                _bank.get("concepts", None), "shape"
            ) else 0
            del _bank
        except Exception as exc:
            raise RuntimeError(f"Could not read combined concept bank {raw_bank_path}: {exc}")
        if _n_concepts == 0:
            raise RuntimeError(
                f"Combined concept bank is EMPTY ({raw_bank_path}). No concept "
                f"passed the CLEAN_EXAMPLE_RATIO={os.environ.get('CLEAN_EXAMPLE_RATIO', '0.8')} "
                "purity filter — the VLM answered 'No [tag]' for the top-activating "
                "regions of every direction. All filtered concepts were saved to the "
                f"negative bank (combined_negative_concept_{method}_{config.ratio_tag}_*.pth) for inspection. "
                "Check the 'Combine Concepts' log above for per-concept positive ratios; "
                "consider lowering CLEAN_EXAMPLE_RATIO, increasing regions per tag, or "
                "reviewing the crops (CROP_MODE, PATCH_SIZE)."
            )
        logger.info(f"Combined concept bank: {_n_concepts} concepts ({raw_bank_path})")

        # Regrounding (subprocess — loads model again)
        logger.info(f"START: Reground Concepts ({method})")
        
        reground_args = [
            str(ROOT_DIR / "src" / "analyse_features.py"),
            "--model_name", config.vlm_model,
            "--analysis_name", f"redefine_activations_text_grounding_{method}",
            "--analysis_saving_path", str(method_dir / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"),
            "--module_to_decompose", config.layer_path,
            "--decomposition_method", method,
            "--save_filename", f"combined_concept_{method}_{config.ratio_tag}_gl_regrounded",
            "--save_dir", str(method_dir),
            "--load_matched_features",
        ]
        
        _run_python_subprocess(reground_args, logger=logger)
        
        logger.info(f"DONE: Reground Concepts ({method})")
        
        # Cleanup intermediate directory
        if intermediate_dir.exists():
            shutil.rmtree(intermediate_dir, ignore_errors=True)
    
    # If any method was re-computed, cascade delete downstream outputs
    if step_4_recomputed:
        _delete_downstream_outputs(config, 5, logger)


def step_6_vlm_explainer(config: PipelineConfig, logger: logging.Logger):
    """Step 6: VLM explainer per method.

    Runs as a subprocess — loads VLM model for explanation generation.
    EXPL_MAX_IMAGES: -1 = all IMAGE_ROOT images, 0 = skip this step,
    N = first N images.
    """

    if config.expl_max_images == 0:
        logger.info("Skip Step 6 (EXPL_MAX_IMAGES=0 — explanation disabled); steps 7-8 will have no input.")
        return

    step_5_recomputed = False  # Track if any method was re-computed
    
    for method in config.decomp_methods:
        method = method.strip()
        concept_path = config.decomp_dir / method / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        out_dir = config.explain_dir / method
        out_json = out_dir / "vlm_explanations.json"
        
        if out_json.exists():
            logger.info(f"Skip Explainer ({method}) (found {out_json})")
            continue
        
        # Mark that step 5 is being re-computed for this method
        step_5_recomputed = True
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"START: Explainer ({method})")
        
        explainer_args = [
            str(ROOT_DIR / "inference" / "vlm_explainer_multibatch.py"),
            "--model_name", config.vlm_model,
            "--concept_path", str(concept_path),
            "--layer_path", config.layer_path,
            "--image_root", str(config.image_root),
            "--top_n", str(config.top_n),
            "--out_json", str(out_json),
            "--prompt_mode", config.expl_prompt_mode,
            "--max_images", str(config.expl_max_images),
        ]
        
        if config.expl_label:
            explainer_args.extend(["--prompt_label", config.expl_label])
        if config.expl_choices:
            explainer_args.extend(["--choices", config.expl_choices])
        if config.single_object:
            explainer_args.append("--single_object")

        _run_python_subprocess(explainer_args, logger=logger)
        
        logger.info(f"DONE: Explainer ({method})")
    
    # If any method was re-computed, cascade delete downstream outputs
    if step_5_recomputed:
        _delete_downstream_outputs(config, 6, logger)


def step_7_concept_deletion_eval(config: PipelineConfig, logger: logging.Logger):
    """Step 7: Concept deletion eval per method.
    
    Each eval run is a subprocess — loads VLM model for token evaluation.
    """
    
    step_6_recomputed = False  # Track if any eval was re-computed
    
    for method in config.decomp_methods:
        method = method.strip()
        concept_path = config.decomp_dir / method / f"combined_concept_{method}_{config.ratio_tag}_raw.pth"
        in_json = config.explain_dir / method / "vlm_explanations.json"
        out_dir = config.eval_dir / method

        if not in_json.exists():
            logger.info(
                f"Skip Eval ({method}) — no explanations at {in_json} "
                "(step 6 skipped via EXPL_MAX_IMAGES=0 or produced nothing)."
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        
        for rank in range(0, config.top_n):
            rank_idx = rank + 1
            insertion_csv = out_dir / f"c_insertion_token_rank{rank_idx}.csv"
            deletion_csv = out_dir / f"c_deletion_token_rank{rank_idx}.csv"
            insertion_random_csv = out_dir / f"c_insertion_token_rank{rank_idx}_random.csv"
            deletion_random_csv = out_dir / f"c_deletion_token_rank{rank_idx}_random.csv"

            # Insertion eval
            if insertion_csv.exists():
                logger.info(f"Skip Eval Insert (rank={rank_idx}, {method})")
            else:
                step_6_recomputed = True
                logger.info(f"START: Eval Insert (rank={rank_idx}, {method})")

                insert_args = [
                    str(ROOT_DIR / "eval" / "concept_deletion_eval.py"),
                    "--results_json", str(in_json),
                    "--concept_path", str(concept_path),
                    "--model_name", config.vlm_model,
                    "--layer_path", config.layer_path,
                    "--mode", "token",
                    "--num_points", str(config.num_points),
                    "--out_dir", str(out_dir),
                    "--device", config.device,
                    "--rank", str(rank_idx),
                    "--insertion",
                ]

                _run_python_subprocess(insert_args, logger=logger)

                logger.info(f"DONE: Eval Insert (rank={rank_idx}, {method})")

            # Insertion eval -- random-order baseline (chance-level faithfulness)
            if insertion_random_csv.exists():
                logger.info(f"Skip Eval Insert Random (rank={rank_idx}, {method})")
            else:
                step_6_recomputed = True
                logger.info(f"START: Eval Insert Random (rank={rank_idx}, {method})")

                insert_random_args = [
                    str(ROOT_DIR / "eval" / "concept_deletion_eval.py"),
                    "--results_json", str(in_json),
                    "--concept_path", str(concept_path),
                    "--model_name", config.vlm_model,
                    "--layer_path", config.layer_path,
                    "--mode", "token",
                    "--num_points", str(config.num_points),
                    "--out_dir", str(out_dir),
                    "--device", config.device,
                    "--rank", str(rank_idx),
                    "--insertion",
                    "--order_mode", "random",
                ]

                _run_python_subprocess(insert_random_args, logger=logger)

                logger.info(f"DONE: Eval Insert Random (rank={rank_idx}, {method})")

            # Deletion eval
            if deletion_csv.exists():
                logger.info(f"Skip Eval Delete (rank={rank_idx}, {method})")
            else:
                step_6_recomputed = True
                logger.info(f"START: Eval Delete (rank={rank_idx}, {method})")

                delete_args = [
                    str(ROOT_DIR / "eval" / "concept_deletion_eval.py"),
                    "--results_json", str(in_json),
                    "--concept_path", str(concept_path),
                    "--model_name", config.vlm_model,
                    "--layer_path", config.layer_path,
                    "--mode", "token",
                    "--num_points", str(config.num_points),
                    "--out_dir", str(out_dir),
                    "--device", config.device,
                    "--rank", str(rank_idx),
                ]

                _run_python_subprocess(delete_args, logger=logger)

                logger.info(f"DONE: Eval Delete (rank={rank_idx}, {method})")

            # Deletion eval -- random-order baseline
            if deletion_random_csv.exists():
                logger.info(f"Skip Eval Delete Random (rank={rank_idx}, {method})")
            else:
                step_6_recomputed = True
                logger.info(f"START: Eval Delete Random (rank={rank_idx}, {method})")

                delete_random_args = [
                    str(ROOT_DIR / "eval" / "concept_deletion_eval.py"),
                    "--results_json", str(in_json),
                    "--concept_path", str(concept_path),
                    "--model_name", config.vlm_model,
                    "--layer_path", config.layer_path,
                    "--mode", "token",
                    "--num_points", str(config.num_points),
                    "--out_dir", str(out_dir),
                    "--device", config.device,
                    "--rank", str(rank_idx),
                    "--order_mode", "random",
                ]

                _run_python_subprocess(delete_random_args, logger=logger)

                logger.info(f"DONE: Eval Delete Random (rank={rank_idx}, {method})")

        # Top-k semantic table (BERT/CLIP) + random activation baseline
        logger.info(f"START: Eval BERT/CLIP Top-K Table ({method})")
        clip_bert_args = [
            str(ROOT_DIR / "eval" / "clip_bert_score_eval.py"),
            "--json_path", str(in_json),
            "--concept_path", str(concept_path),
            "--max_k", str(config.top_n),
            "--seed", str(config.seed),
            "--out_dir", str(out_dir),
            "--output_prefix", "clip_bert_topk",
        ]
        _run_python_subprocess(clip_bert_args, logger=logger)
        logger.info(f"DONE: Eval BERT/CLIP Top-K Table ({method})")

        # AUC summary table from concept insertion/deletion curves for ranks 1..TOP_N
        logger.info(f"START: Eval AUC Table ({method})")
        auc_args = [
            str(ROOT_DIR / "eval" / "concept_curve_auc_eval.py"),
            "--out_dir", str(out_dir),
            "--top_n", str(config.top_n),
            "--mode", "token",
            "--output_prefix", "concept_curve_auc_token",
        ]
        _run_python_subprocess(auc_args, logger=logger)
        logger.info(f"DONE: Eval AUC Table ({method})")
    
    # If any eval was re-computed, cascade delete downstream outputs (plots)
    if step_6_recomputed:
        _delete_downstream_outputs(config, 7, logger)


def step_8_plots(config: PipelineConfig, logger: logging.Logger):
    """Step 8: Plots per method (optional)."""
    
    plot_token_script = ROOT_DIR / "scripts" / "plot_concept_deletion_eval_token.py"
    
    if plot_token_script.exists():
        sys.path.insert(0, str(ROOT_DIR / "scripts"))
        
        for method in config.decomp_methods:
            method = method.strip()
            plot_dir = config.eval_dir / method
            plot_output = plot_dir / f"{method}_concept_token_curves.png"
            
            if plot_dir.exists() and any(plot_dir.glob("c_*_token_rank*.csv")):
                if plot_output.exists():
                    logger.info(f"Skip Plot Token ({method}) (found {plot_output})")
                    continue
                
                logger.info(f"START: Plot Token ({method})")
                
                # Import and run plot script
                try:
                    from scripts.plot_concept_deletion_eval_token import main as plot_token_main
                    
                    plot_args = [
                        "--out_dir", str(plot_dir),
                        "--ymin", str(config.plot_ymin),
                        "--ymax", str(config.plot_ymax),
                    ]
                    
                    original_argv = sys.argv
                    sys.argv = ["plot_concept_deletion_eval_token.py"] + plot_args
                    try:
                        plot_token_main()
                    finally:
                        sys.argv = original_argv
                    
                    logger.info(f"DONE: Plot Token ({method})")
                except ImportError:
                    logger.warning(f"Could not import plot script; skipping {method}")
            else:
                logger.warning(f"No CSVs in {plot_dir} for plotting; skipping {method}")
    else:
        logger.warning("Plot script not found; skipping per-method plots.")
    
    # Summary plots
    summary_script = ROOT_DIR / "scripts" / "plot_eval_summary_across_methods.py"
    summary_output = config.plots_dir / "summary_comparison.png"
    
    if summary_script.exists():
        if summary_output.exists():
            logger.info(f"Skip Plot Summary (found {summary_output})")
        else:
            logger.info("START: Plot Summary Across Methods")
            
            try:
                from scripts.plot_eval_summary_across_methods import main as summary_main
                
                summary_args = [
                    "--eval_dir", str(config.eval_dir),
                    "--out_dir", str(config.plots_dir),
                    "--methods", ",".join(config.decomp_methods),
                    "--ymin", str(config.plot_ymin),
                    "--ymax", str(config.plot_ymax),
                ]
                
                original_argv = sys.argv
                sys.argv = ["plot_eval_summary_across_methods.py"] + summary_args
                try:
                    summary_main()
                finally:
                    sys.argv = original_argv
                
                logger.info("DONE: Plot Summary Across Methods")
            except ImportError:
                logger.warning("Could not import summary plot script; skipping overlay plots.")
    else:
        logger.warning("Summary plotter not found; skipping overlay plots.")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XL-VLMs Pipeline - Native Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Root dataset/images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Root output directory",
    )
    parser.add_argument(
        "--decomp",
        type=str,
        help="Comma-separated decomposition methods",
    )
    parser.add_argument(
        "--plot-ymin",
        type=float,
        help="Y-axis min for plots",
    )
    parser.add_argument(
        "--plot-ymax",
        type=float,
        help="Y-axis max for plots",
    )
    parser.add_argument(
        "--skip-to-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Skip to a specific step (for resuming)",
    )
    parser.add_argument(
        "--only-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run only a specific step",
    )
    
    return parser.parse_args()


def main():
    """Main pipeline entry point."""
    # Parse CLI args
    args = parse_args()
    
    # Load configuration
    config = PipelineConfig()
    config.update_from_args(args)
    
    # Set environment variables
    os.environ["HF_HOME"] = str(config.hf_home)
    # Do NOT override CUDA_VISIBLE_DEVICES — device placement is handled by
    # device_utils.parse_device_config() using the DEVICE env var. Export the
    # resolved value (incl. the DEVICE_ID fallback) so every subprocess —
    # also the ones that take no --device flag — uses the same GPU(s).
    os.environ["DEVICE"] = config.device
    os.environ["DETECTION_BATCH_SIZE"] = str(config.detection_batch_size)
    os.environ["POSITIVE_NEGATIVE_SEGMENT"] = str(config.positive_negative_segment)
    os.environ["MASK_CONTEXT_PIXELS"] = str(config.mask_context_pixels)
    os.environ["OUTPUT_DIR"] = str(config.output_dir)
    # Debug mode must be opt-in: when "1" it forces Step 4 (features) to rerun
    # and cascade-deletes every downstream output (decomposition, explanations,
    # eval, plots) on each invocation.
    os.environ["DEBUG_SAVE_VLM_INPUTS"] = os.environ.get("DEBUG_SAVE_VLM_INPUTS", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Create directories and setup logging
    create_directories(config, logging.getLogger())
    logger = setup_logging(config.logs_dir)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("XL-VLMs Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Root:       {config.root_dir}")
    logger.info(f"Input dir:  {config.input_dir}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Model:      {config.vlm_model} | Batch: {config.batch_size} | Seed: {config.seed} | Device: {config.device}")
    logger.info(f"Decompose:  {', '.join(config.decomp_methods)}")
    logger.info(f"Crops: detector={config.object_detector} masks={config.masks_per_image} concept={config.concept_masks_per_image} patch={config.patch_size} min={config.min_images_per_tag} max={config.max_images_per_tag}")
    logger.info(f"Masking: ctx={config.mask_context_pixels} pos_neg_segment={config.positive_negative_segment}")
    logger.info(f"Resize:     ref_width={config.image_size_width} (height auto by aspect ratio)")
    logger.info(f"Explainer:  layer={config.layer_path} image_root={config.image_root} top_n={config.top_n} mode={config.expl_prompt_mode}")
    logger.info(f"Hooks:      {config.hook_names}")
    if config.logit_lens_layer_selection:
        logger.info(
            f"LayerSelect: ON mode={config.logit_lens_mode} layers={config.logit_lens_layers} "
            f"regions={config.logit_lens_num_patches} (from crops.json)"
        )
    logger.info(f"Grounding:  num_most_activating_samples={config.num_most_activating_samples}")
    logger.info(f"Plots Y:    [{config.plot_ymin}, {config.plot_ymax}]")
    logger.info("=" * 60)
    
    # Define pipeline steps
    steps = [
        (1, "Dataset Inference", step_1_dataset_inference),
        (2, "Build Crops JSON", step_2_build_crops_json),
        (3, "Select Layers (Logit Lens)", step_3_select_layers),
        (4, "Generate Features", step_4_generate_features),
        (5, "Decompose Features", step_5_decompose_features),
        (6, "VLM Explainer", step_6_vlm_explainer),
        (7, "Concept Deletion Eval", step_7_concept_deletion_eval),
        (8, "Plots", step_8_plots),
    ]
    
    # Determine which steps to run
    start_step = args.skip_to_step if args.skip_to_step else 1
    only_step = args.only_step
    
    try:
        for step_num, step_name, step_func in steps:
            if only_step is not None and step_num != only_step:
                continue
            if step_num < start_step:
                logger.info(f"Skipping Step {step_num}: {step_name}")
                continue
            
            logger.info(f"Running Step {step_num}: {step_name}")
            step_func(config, logger)
        
        logger.info("=" * 60)
        logger.info(f"Pipeline completed. Outputs: {config.output_dir}")
        logger.info(f"Logs: {config.logs_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()