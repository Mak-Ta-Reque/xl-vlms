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
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

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
        self.device_id = self._get_int("DEVICE_ID", 0)
        
        # Crops JSON generation
        self.concept_crops_per_image = self._get_int("CONCEPT_CROPS_PER_IMAGE", 20)
        self.patch_size = self._get_int("PATCH_SIZE", 200)
        self.min_images_per_tag = self._get_int("MIN_IMAGES_PER_TAG", 10)
        self.max_images_per_tag = self._get_int("MAX_IMAGES_PER_TAG", 128)
        self.patches_per_image = self._get_int("PATCHES_PER_IMAGE", 10)
        self.concept_mode = self._get_int("CONCEPT_MODE", 1)
        self.object_detector = self._get_str("OBJECT_DETECTOR", "none")  # 'none', 'langsam', 'sam3'
        self.detection_batch_size = self._get_int("DETECTION_BATCH_SIZE", 5)
        self.detection_topn = self._get_int("DETECTION_TOPN", 5)
        
        # Inference prompt and image preprocessing
        self.prompt = self._get_str(
            "PROMPT",
            "Identify every visible object, item, concept, and pattern in the image at the most fine-grained level. Output only single words in a strict comma-separated list, no sentences or explanations."
        )
        # Use IMAGE_SIZE_WIDTH as the reference; height is derived per-image via aspect ratio.
        self.image_size_width = self._get_int("IMAGE_SIZE_WIDTH", 512)
        # Keep for backward compatibility / debugging; not used as a fixed resize target.
        self.image_size_height = self._get_int("IMAGE_SIZE_HEIGHT", 512)
        self.image_size = (self.image_size_width, self.image_size_height)
        self.image_budget = self._get_int("IMAGE_BUDGET", 2000)# reduce for test , use 200 -> for a better run
        self.box_threshold = self._get_float("BOX_THRESHOLD", 0.5)
        
        # Decomposition methods
        self.decomp_methods = self._get_str("DECOMP_METHODS", "snmf").split(",")
        self.num_concepts = self._get_int("NUM_CONCEPTS", 2)
        self.dataset_size = self._get_int("BAG_SIZE", 100)
        self.delete_intermediate_files = self._get_int("DELETE_INTERMEDIATE_FILES", 0) == 1
        
        # Explainer/Eval
        self.layer_path = self._get_str("LAYER_PATH", "model.language_model.norm")
        self.top_n = self._get_int("TOP_N", 5)
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
    
    # Check if already done
    if config.objects_csv.exists() and config.concept_map_json.exists():
        logger.info(f"Skip Dataset Inference (found {config.objects_csv} and {config.concept_map_json})")
        return
    
    config.objects_csv.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    # Build concept map
    logger.info("START: Build Concept Map")
    
    from concept_image_mapping import main as concept_mapping_main
    
    mapping_args = [
        "--input", str(config.objects_csv),
        "--output", str(config.concept_map_json),
    ]
    
    sys.argv = ["concept_image_mapping.py"] + mapping_args
    try:
        concept_mapping_main()
    finally:
        sys.argv = original_argv
    
    logger.info("DONE: Build Concept Map")


def step_2_build_crops_json(config: PipelineConfig, logger: logging.Logger):
    """Step 2: Build crops JSON from concept→image map."""
    
    if config.crops_json.exists():
        logger.info(f"Skip Crops JSON (found {config.crops_json})")
        return
    
    config.crops_json.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("START: Crops JSON")
    
    sys.path.insert(0, str(ROOT_DIR / "preprocessing"))
    from preprocessing.crops_to_json import main as crops_to_json_main
    
    crops_args = [
        "--input_root", str(config.input_dir),
        "--json_mapping", str(config.concept_map_json),
        "--output_json", str(config.crops_json),
        "--patch_size", str(config.patch_size),
        "--patches_per_image", str(config.patches_per_image),
        "--min_images_per_tag", str(config.min_images_per_tag),
        "--max_images_per_tag", str(config.max_images_per_tag),
        "--seed", str(config.seed),
        "--device", f"cuda:{config.device_id}",
        "--image_size_width", str(config.image_size_width),
    ]
    
    if config.concept_mode == 1:
        crops_args.extend([
            "--concept_mode",
            "--concept_crops_per_image", str(config.concept_crops_per_image),
        ])
    
    # Object detector: 'none' = random only, 'langsam'/'sam3' = detector + random
    if config.object_detector in ("langsam", "sam3"):
        crops_args.extend([
            "--object_detector", config.object_detector,
            "--batch_size", str(config.detection_batch_size),
            "--topn", str(config.detection_topn),
        ])
    
    original_argv = sys.argv
    sys.argv = ["crops_to_json.py"] + crops_args
    try:
        crops_to_json_main()
    finally:
        sys.argv = original_argv
    
    logger.info("DONE: Crops JSON")


def step_3_generate_features(config: PipelineConfig, logger: logging.Logger):
    """Step 3: Generate features from crops JSON."""
    
    features_path = config.features_dir / "features"
    if features_path.exists() and any(features_path.glob("*.pth")):
        logger.info(f"Skip Feature Generation (found features under {features_path})")
        return
    
    logger.info("START: Generate Features")
    
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from src.save_features import main as save_features_main
    
    features_args = [
        "--model_name", config.vlm_model,
        "--dataset_name", "json_crop_map",
        "--dataset_size", str(config.dataset_size),
        "--data_dir", str(config.input_dir),
        "--annotation_file", str(config.crops_json),
        "--split", "train",
        "--hook_names", "save_hidden_states_mean",
        "--modules_to_hook", config.layer_path,
        "--prompt_template", "cgdl",
        "--save_dir", str(config.features_dir),
        "--batch_size", str(config.batch_size),
        "--generation_mode",
        "--save_only_generated_tokens",
        "--exact_match_modules_to_hook",
    ]
    
    original_argv = sys.argv
    sys.argv = ["save_features.py"] + features_args
    try:
        save_features_main()
    finally:
        sys.argv = original_argv
    
    logger.info("DONE: Generate Features")


def step_4_decompose_features(config: PipelineConfig, logger: logging.Logger):
    """Step 4: Decompose features across methods."""
    
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from src.analyse_features import main as analyse_features_main
    from src.combine_concepts import main as combine_concepts_main
    
    for method in config.decomp_methods:
        method = method.strip()
        method_dir = config.decomp_dir / method
        out_raw = method_dir / f"combined_concept_{method}_raw.pth"
        
        if out_raw.exists():
            logger.info(f"Skip Decompose ({method}) (found {out_raw})")
            continue
        
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyse features
        base_analysis_name = "decompose_activations_text_grounding_image_grounding"
        intermediate_dir = method_dir / f"intermediate_{method}"
        
        logger.info(f"START: Decompose:{method} (batch)")
        
        analyse_args = [
            "--model_name", config.vlm_model,
            "--analysis_name", f"{base_analysis_name}_{method}",
            "--features_path", str(config.features_dir / "features"),
            "--module_to_decompose", config.layer_path,
            "--num_concepts", str(config.num_concepts),
            "--decomposition_method", method,
            "--save_dir", str(intermediate_dir),
        ]
        
        original_argv = sys.argv
        sys.argv = ["analyse_features.py"] + analyse_args
        try:
            analyse_features_main()
        finally:
            sys.argv = original_argv
        
        logger.info(f"DONE: Decompose:{method} (batch)")
        
        # Combine concepts
        logger.info(f"START: Combine Concepts ({method})")
        
        combine_args = [
            "--input_dir", str(intermediate_dir),
            "--output_path", str(method_dir / f"combined_concept_{method}.pth"),
            "--normalization", "gl",
        ]
        
        # Add --delete flag if configured
        if config.delete_intermediate_files:
            combine_args.append("--delete")
        
        sys.argv = ["combine_concepts.py"] + combine_args
        try:
            # combine_concepts.main() takes args as parameter, need to parse them
            import argparse as ap
            combine_parser = ap.ArgumentParser()
            combine_parser.add_argument("--input_dir", type=str, required=True)
            combine_parser.add_argument("--output_path", type=str, required=True)
            combine_parser.add_argument("--normalization", type=str, default="gl")
            combine_parser.add_argument("--delete", action="store_true", default=False)
            combine_parsed = combine_parser.parse_args(combine_args)
            combine_concepts_main(combine_parsed)
        finally:
            sys.argv = original_argv
        
        logger.info(f"DONE: Combine Concepts ({method})")
        
        # Regrounding
        logger.info(f"START: Reground Concepts ({method})")
        
        reground_args = [
            "--model_name", config.vlm_model,
            "--analysis_name", f"redefine_activations_text_grounding_{method}",
            "--analysis_saving_path", str(method_dir / f"combined_concept_{method}_raw.pth"),
            "--module_to_decompose", config.layer_path,
            "--decomposition_method", method,
            "--save_filename", f"combined_concept_{method}_gl_regrounded",
            "--save_dir", str(method_dir),
            "--load_matched_features",
        ]
        
        sys.argv = ["analyse_features.py"] + reground_args
        try:
            analyse_features_main()
        finally:
            sys.argv = original_argv
        
        logger.info(f"DONE: Reground Concepts ({method})")
        
        # Cleanup intermediate directory
        if intermediate_dir.exists():
            shutil.rmtree(intermediate_dir, ignore_errors=True)


def step_5_vlm_explainer(config: PipelineConfig, logger: logging.Logger):
    """Step 5: VLM explainer per method."""
    
    sys.path.insert(0, str(ROOT_DIR / "inference"))
    from inference.vlm_explainer_multibatch import main as vlm_explainer_main
    
    for method in config.decomp_methods:
        method = method.strip()
        concept_path = config.decomp_dir / method / f"combined_concept_{method}_raw.pth"
        out_dir = config.explain_dir / method
        out_json = out_dir / "vlm_explanations.json"
        
        if out_json.exists():
            logger.info(f"Skip Explainer ({method}) (found {out_json})")
            continue
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"START: Explainer ({method})")
        
        explainer_args = [
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
        
        original_argv = sys.argv
        sys.argv = ["vlm_explainer_multibatch.py"] + explainer_args
        try:
            vlm_explainer_main()
        finally:
            sys.argv = original_argv
        
        logger.info(f"DONE: Explainer ({method})")


def step_6_concept_deletion_eval(config: PipelineConfig, logger: logging.Logger):
    """Step 6: Concept deletion eval per method."""
    
    sys.path.insert(0, str(ROOT_DIR / "eval"))
    from eval.concept_deletion_eval import main as concept_deletion_eval_main
    
    for method in config.decomp_methods:
        method = method.strip()
        concept_path = config.decomp_dir / method / f"combined_concept_{method}_raw.pth"
        in_json = config.explain_dir / method / "vlm_explanations.json"
        out_dir = config.eval_dir / method
        
        if out_dir.exists() and any(out_dir.glob("*.csv")):
            logger.info(f"Skip Eval (Token) - {method} (CSVs exist)")
            continue
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for rank in range(0, config.top_n):
            # Insertion eval
            logger.info(f"START: Eval Insert (rank={rank+1}, {method})")
            
            insert_args = [
                "--results_json", str(in_json),
                "--concept_path", str(concept_path),
                "--model_name", config.vlm_model,
                "--layer_path", config.layer_path,
                "--mode", "token",
                "--num_points", str(config.num_points),
                "--out_dir", str(out_dir),
                "--device", config.device_id,
                "--rank", str(rank+1),
                "--insertion",
            ]
            
            original_argv = sys.argv
            sys.argv = ["concept_deletion_eval.py"] + insert_args
            try:
                concept_deletion_eval_main()
            finally:
                sys.argv = original_argv
            
            logger.info(f"DONE: Eval Insert (rank={rank+1}, {method})")
            
            # Deletion eval
            logger.info(f"START: Eval Delete (rank={rank+1}, {method})")
            
            delete_args = [
                "--results_json", str(in_json),
                "--concept_path", str(concept_path),
                "--model_name", config.vlm_model,
                "--layer_path", config.layer_path,
                "--mode", "token",
                "--num_points", str(config.num_points),
                "--out_dir", str(out_dir),
                "--device", config.device_id,
                "--rank", str(rank+1),
            ]
            
            sys.argv = ["concept_deletion_eval.py"] + delete_args
            try:
                concept_deletion_eval_main()
            finally:
                sys.argv = original_argv
            
            logger.info(f"DONE: Eval Delete (rank={rank+1}, {method})")


def step_7_plots(config: PipelineConfig, logger: logging.Logger):
    """Step 7: Plots per method (optional)."""
    
    plot_token_script = ROOT_DIR / "scripts" / "plot_concept_deletion_eval_token.py"
    
    if plot_token_script.exists():
        sys.path.insert(0, str(ROOT_DIR / "scripts"))
        
        for method in config.decomp_methods:
            method = method.strip()
            plot_dir = config.eval_dir / method
            
            if plot_dir.exists() and any(plot_dir.glob("c_*_token_rank*.csv")):
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
    
    if summary_script.exists():
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
    
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
    logger.info(f"Model:      {config.vlm_model} | Batch: {config.batch_size} | Seed: {config.seed} | Device: cuda:{config.device_id}")
    logger.info(f"Decompose:  {', '.join(config.decomp_methods)}")
    logger.info(f"Crops: input={config.input_dir} k={config.concept_crops_per_image} patch={config.patch_size} min={config.min_images_per_tag} max={config.max_images_per_tag}")
    logger.info(f"Resize:     ref_width={config.image_size_width} (height auto by aspect ratio)")
    logger.info(f"Explainer:  layer={config.layer_path} image_root={config.image_root} top_n={config.top_n} mode={config.expl_prompt_mode}")
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
