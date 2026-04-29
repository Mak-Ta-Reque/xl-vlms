#!/usr/bin/env python3
"""
XL-VLMs Pipeline - Native Python Implementation

This script is a Python conversion of run_full_pipeline_without_coroping.sh
for easier debugging and development in VS Code.

Steps:
1) Dataset inference -> concepts map
2) Build crops JSON from concept→image mapping
3) Generate features from crops JSON (on-the-fly cropping)
4) Decompose features (one or more methods)
5) Run VLM explainer per method
6) Concept deletion eval per method
7) (Optional) Plots per method + summary

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

import inflect

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))
p = inflect.engine()

# Try to import dotenv for loading .env file
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv


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
        self.device = self._get_str("DEVICE", "cuda:0")
        
        # Crops JSON generation
        self.masks_per_image = self._get_int("MASKS_PER_IMAGE", 5)
        self.concept_masks_per_image = self._get_int("CONCEPT_MASKS_PER_IMAGE", 1)
        self.patch_size = self._get_int("PATCH_SIZE", 200)
        self.min_images_per_tag = self._get_int("MIN_IMAGES_PER_TAG", 10)
        self.max_images_per_tag = self._get_int("MAX_IMAGES_PER_TAG", 128)
        self.crop_mode = self._get_str("CROP_MODE", "random").strip().lower()
        if self.crop_mode in {"random", "sliding_window"}:
            self.object_detector = "none"
        elif self.crop_mode in {"sam3", "langsam"}:
            self.object_detector = self.crop_mode
        else:
            raise ValueError(
                "Invalid CROP_MODE. Use one of: random, sliding_window, langsam, sam3"
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
        self.num_concept = self._get_int("NUM_CONCEPT", -1)  # -1 = use all tags, else top N tags by crop count
        self.concepts_vocab = self._get_path("CONCEPTS_VOCAB", ROOT_DIR / "src" / "assets" / "vocab.txt")
        self.dataset_size = self._get_int("BAG_SIZE", 100)
        self.delete_intermediate_files = self._get_int("DELETE_INTERMEDIATE_FILES", 0) == 1
        
        # Explainer/Eval
        self.layer_path = self._get_str("LAYER_PATH", "model.language_model.norm")
        self.top_n = self._get_int("TOP_N", 5)
        self.num_most_activating_samples = self._get_int("NUM_MOST_ACTIVATING_SAMPLES", 10)
        self.num_points = self._get_int("NUM_POINTS", 70)
        self.expl_prompt_mode = self._get_str("EXPL_PROMPT_MODE", "unsupervised")
        self.expl_label = self._get_str("EXPL_LABEL", "")
        self.expl_choices = self._get_str("EXPL_CHOICES", "")
        
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
    if env_overrides:
        env.update(env_overrides)
    
    cmd = [sys.executable] + script_args
    if logger:
        logger.debug(f"Subprocess: {' '.join(cmd)}")
    
    proc = subprocess.run(
        cmd, env=env, cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Relay subprocess output through logger
    if proc.stdout:
        for line in proc.stdout.splitlines():
            if logger:
                logger.info(f"  [sub] {line}")
            else:
                print(line)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {proc.returncode}): {' '.join(cmd)}"
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
            (config.features_dir / "features", "Features"),
            (config.decomp_dir, "Decomposed concepts"),
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 2:
        # If step 2 reruns, delete outputs from step 3 onwards
        downstream_deletions = [
            (config.features_dir / "features", "Features"),
            (config.decomp_dir, "Decomposed concepts"),
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 3:
        # If step 3 reruns, delete outputs from step 4 onwards
        downstream_deletions = [
            (config.decomp_dir, "Decomposed concepts"),
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 4:
        # If step 4 reruns, delete outputs from step 5 onwards
        downstream_deletions = [
            (config.explain_dir, "Explanations"),
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 5:
        # If step 5 reruns, delete outputs from step 6 onwards
        downstream_deletions = [
            (config.eval_dir, "Evaluations"),
            (config.plots_dir, "Plots"),
        ]
    elif step_num <= 6:
        # If step 6 reruns, delete outputs from step 7 (plots)
        downstream_deletions = [
            (config.plots_dir, "Plots"),
        ]
    # Step 7 has no downstream dependencies
    
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


def _load_concept_vocab(vocab_path: Path) -> Optional[set]:
    if not vocab_path.exists():
        return None
    vocab = set()
    with open(vocab_path, "r", encoding="utf-8") as handle:
        for line in handle:
            concept = line.strip()
            if not concept or concept.startswith("#"):
                continue
            vocab.add(_normalize_concept_name(concept))
    return vocab


def _normalize_concept_name(name: str) -> str:
    """Normalize concept names for case-insensitive singular/plural matching."""
    value = name.strip().casefold()
    if not value:
        return value
    return " ".join(p.singular_noun(word) or word for word in value.split())


def _ensure_top_concept_map(
    concept_map_json: Path,
    num_concept: int,
    logger: logging.Logger,
    concepts_vocab: Optional[Path] = None,
) -> Path:
    """Create and return a filtered concept map containing vocab-filtered top-N concepts."""
    if num_concept <= 0:
        return concept_map_json

    if not concept_map_json.exists():
        raise FileNotFoundError(f"Concept map not found: {concept_map_json}")

    with open(concept_map_json, "r", encoding="utf-8") as handle:
        concept_mapping = json.load(handle)

    if not isinstance(concept_mapping, dict):
        raise ValueError(f"Expected a JSON object in {concept_map_json}")

    vocab = None
    if concepts_vocab is not None:
        vocab = _load_concept_vocab(concepts_vocab)
        if vocab is None:
            logger.warning(f"Concept vocab not found: {concepts_vocab}. Falling back to all concepts.")
        else:
            logger.info(f"Loaded {len(vocab)} concept names from {concepts_vocab}")

    filtered_candidates = concept_mapping.items()
    if vocab is not None:
        filtered_candidates = [
            (tag, images)
            for tag, images in concept_mapping.items()
            if _normalize_concept_name(tag) in vocab
        ]
        logger.info(
            f"Vocab filter kept {len(filtered_candidates)} of {len(concept_mapping)} concepts from {concept_map_json}"
        )

    sorted_concepts = sorted(
        filtered_candidates,
        key=lambda item: (-len(item[1]), item[0]),
    )
    selected_concepts = sorted_concepts[: min(num_concept, len(sorted_concepts))]
    filtered_mapping = {tag: images for tag, images in selected_concepts}

    vocab_suffix = ""
    if concepts_vocab is not None:
        vocab_suffix = f"_{concepts_vocab.stem}"
    filtered_path = concept_map_json.with_name(
        f"{concept_map_json.stem}{vocab_suffix}_top{len(filtered_mapping)}{concept_map_json.suffix}"
    )
    if filtered_path.exists():
        return filtered_path

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_path, "w", encoding="utf-8") as handle:
        json.dump(filtered_mapping, handle, indent=2, ensure_ascii=False)

    logger.info(
        f"Filtered concept map saved to {filtered_path} using top {len(filtered_mapping)} of {len(sorted_concepts)} candidate tags"
    )
    logger.info(f"Selected tags: {[tag for tag, _ in selected_concepts]}")
    return filtered_path


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


def step_2_build_crops_json(config: PipelineConfig, logger: logging.Logger):
    """Step 2: Build crops JSON from concept→image map.
    
    Runs as a subprocess to isolate detector GPU memory.
    """
    
    if config.crops_json.exists():
        logger.info(f"Skip Step 2 (found {config.crops_json})")
        return
    
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


def step_3_generate_features(config: PipelineConfig, logger: logging.Logger):
    """Step 3: Generate features from crops JSON.
    
    Runs as a subprocess to isolate VLM GPU memory from Step 4.
    """
    
    features_path = config.features_dir / "features"
    debug_save_enabled = os.environ.get("DEBUG_SAVE_VLM_INPUTS", "0") == "1"
    if features_path.exists() and any(features_path.glob("*.pth")):
        if debug_save_enabled:
            logger.info(
                f"Debug mode enabled (DEBUG_SAVE_VLM_INPUTS=1): forcing Step 3 rerun even though features exist under {features_path}"
            )
            # Cascade delete downstream outputs
            _delete_downstream_outputs(config, 3, logger)
        else:
            logger.info(f"Skip Step 3 (found features under {features_path})")
            return
    else:
        # Step will re-compute; cascade delete downstream outputs
        _delete_downstream_outputs(config, 3, logger)
    
    logger.info("START: Generate Features")
    annotation_file = config.crops_json
    selected_concept_count = _count_concepts_in_json(config.active_concept_map_json)
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
        "--hook_names", "save_hidden_states_mean",
        "--modules_to_hook", config.layer_path,
            "--prompt_template", config.prompt_template,
        "--save_dir", str(config.features_dir),
        "--batch_size", str(config.batch_size),
        "--generation_mode",
        "--save_only_generated_tokens",
        "--exact_match_modules_to_hook",
    ]
    
    _run_python_subprocess(features_args, logger=logger)
    
    logger.info("DONE: Generate Features")


def step_4_decompose_features(config: PipelineConfig, logger: logging.Logger):
    """Step 4: Decompose features across methods.
    
    Each analyse_features call runs as a subprocess to isolate GPU memory,
    since loading the model's lm_head requires significant VRAM.
    """
    
    step_4_recomputed = False  # Track if any method was re-computed
    
    for method in config.decomp_methods:
        method = method.strip()
        method_dir = config.decomp_dir / method
        out_raw = method_dir / f"combined_concept_{method}_raw.pth"
        
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
        
        analyse_args = [
            str(ROOT_DIR / "src" / "analyse_features.py"),
            "--model_name", config.vlm_model,
            "--analysis_name", f"{base_analysis_name}_{method}",
            "--features_path", str(config.features_dir / "features"),
            "--module_to_decompose", config.layer_path,
            "--num_concepts", str(config.decomp_components),
            "--decomposition_method", method,
            "--num_most_activating_samples", str(config.num_most_activating_samples),
            "--save_dir", str(intermediate_dir),
        ]
        
        _run_python_subprocess(analyse_args, logger=logger)
        
        logger.info(f"DONE: Decompose:{method} (batch)")
        
        # Combine concepts (lightweight, no GPU — runs in-process)
        logger.info(f"START: Combine Concepts ({method})")
        
        combine_cli_args = [
            str(ROOT_DIR / "src" / "combine_concepts.py"),
            "--input_dir", str(intermediate_dir),
            "--output_path", str(method_dir / f"combined_concept_{method}.pth"),
            "--normalization", "gl",
        ]
        if config.delete_intermediate_files:
            combine_cli_args.append("--delete")
        
        _run_python_subprocess(combine_cli_args, logger=logger)
        
        logger.info(f"DONE: Combine Concepts ({method})")
        
        # Regrounding (subprocess — loads model again)
        logger.info(f"START: Reground Concepts ({method})")
        
        reground_args = [
            str(ROOT_DIR / "src" / "analyse_features.py"),
            "--model_name", config.vlm_model,
            "--analysis_name", f"redefine_activations_text_grounding_{method}",
            "--analysis_saving_path", str(method_dir / f"combined_concept_{method}_raw.pth"),
            "--module_to_decompose", config.layer_path,
            "--decomposition_method", method,
            "--save_filename", f"combined_concept_{method}_gl_regrounded",
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
        _delete_downstream_outputs(config, 4, logger)


def step_5_vlm_explainer(config: PipelineConfig, logger: logging.Logger):
    """Step 5: VLM explainer per method.
    
    Runs as a subprocess — loads VLM model for explanation generation.
    """
    
    step_5_recomputed = False  # Track if any method was re-computed
    
    for method in config.decomp_methods:
        method = method.strip()
        concept_path = config.decomp_dir / method / f"combined_concept_{method}_raw.pth"
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
        ]
        
        if config.expl_label:
            explainer_args.extend(["--prompt_label", config.expl_label])
        if config.expl_choices:
            explainer_args.extend(["--choices", config.expl_choices])
        
        _run_python_subprocess(explainer_args, logger=logger)
        
        logger.info(f"DONE: Explainer ({method})")
    
    # If any method was re-computed, cascade delete downstream outputs
    if step_5_recomputed:
        _delete_downstream_outputs(config, 5, logger)


def step_6_concept_deletion_eval(config: PipelineConfig, logger: logging.Logger):
    """Step 6: Concept deletion eval per method.
    
    Each eval run is a subprocess — loads VLM model for token evaluation.
    """
    
    step_6_recomputed = False  # Track if any eval was re-computed
    
    for method in config.decomp_methods:
        method = method.strip()
        concept_path = config.decomp_dir / method / f"combined_concept_{method}_raw.pth"
        in_json = config.explain_dir / method / "vlm_explanations.json"
        out_dir = config.eval_dir / method

        out_dir.mkdir(parents=True, exist_ok=True)
        
        for rank in range(0, config.top_n):
            rank_idx = rank + 1
            insertion_csv = out_dir / f"c_insertion_token_rank{rank_idx}.csv"
            deletion_csv = out_dir / f"c_deletion_token_rank{rank_idx}.csv"

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
        _delete_downstream_outputs(config, 6, logger)


def step_7_plots(config: PipelineConfig, logger: logging.Logger):
    """Step 7: Plots per method (optional)."""
    
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
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Skip to a specific step (for resuming)",
    )
    parser.add_argument(
        "--only-step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
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
    # device_utils.parse_device_config() using the DEVICE env var.
    os.environ["DETECTION_BATCH_SIZE"] = str(config.detection_batch_size)
    os.environ["POSITIVE_NEGATIVE_SEGMENT"] = str(config.positive_negative_segment)
    os.environ["MASK_CONTEXT_PIXELS"] = str(config.mask_context_pixels)
    os.environ["OUTPUT_DIR"] = str(config.output_dir)
    os.environ["DEBUG_SAVE_VLM_INPUTS"] = os.environ.get("DEBUG_SAVE_VLM_INPUTS", "1")
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
    logger.info(f"Grounding:  num_most_activating_samples={config.num_most_activating_samples}")
    logger.info(f"Plots Y:    [{config.plot_ymin}, {config.plot_ymax}]")
    logger.info("=" * 60)
    
    # Define pipeline steps
    steps = [
        (1, "Dataset Inference", step_1_dataset_inference),
        (2, "Build Crops JSON", step_2_build_crops_json),
        (3, "Generate Features", step_3_generate_features),
        (4, "Decompose Features", step_4_decompose_features),
        (5, "VLM Explainer", step_5_vlm_explainer),
        (6, "Concept Deletion Eval", step_6_concept_deletion_eval),
        (7, "Plots", step_7_plots),
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