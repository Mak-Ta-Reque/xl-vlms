#!/usr/bin/env python3
"""
Dataset Inference Script for Vision Language Models

This script iterates through a dataset of images organized in subfolders or directly in the root directory,
runs inference using Hugging Face vision language models,
and saves the results to a CSV file.

The script supports both online model loading from Hugging Face Hub and offline loading from local directories.
For offline loading, use the --local_model_path argument or set the HF_LOCAL_MODEL_PATH environment variable.

Usage:
    # Online loading from Hugging Face Hub
    python dataset_inference.py --dataset_path /path/to/dataset --model_name google/gemma-3n-E4B --output_csv results.csv
    
    # Offline loading from local model directory
    python dataset_inference.py --dataset_path /path/to/dataset --model_name google/gemma-3n-E4B --local_model_path /path/to/local/model --output_csv results.csv
    
    # Using environment variable for local model path
    export HF_LOCAL_MODEL_PATH=/path/to/local/models
    python dataset_inference.py --dataset_path /path/to/dataset --model_name google/gemma-3n-E4B --output_csv results.csv
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

# Try to import Gemma3nForConditionalGeneration, fallback if not available
try:
    from transformers import Gemma3nForConditionalGeneration
    GEMMA3N_AVAILABLE = True
except ImportError:
    GEMMA3N_AVAILABLE = False
    print("Warning: Gemma3nForConditionalGeneration not available. Using fallback for Gemma models.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Popular Hugging Face vision language models
POPULAR_MODELS = {
    'gemma-3n': 'google/gemma-3n-e4b',
    'gemma-3n-e4b': 'google/gemma-3n-e4b',
    'qwen2-vl-7b': 'Qwen/Qwen2-VL-7B-Instruct',
    'qwen2-vl-2b': 'Qwen/Qwen2-VL-2B-Instruct',
    'llava-1.5-7b': 'llava-hf/llava-1.5-7b-hf',
    'llava-1.5-13b': 'llava-hf/llava-1.5-13b-hf',
    'idefics2-8b': 'HuggingFaceM4/idefics2-8b',
    'pixtral-12b': 'mistral-community/Pixtral-12B-2409',
}


def get_image_files(dataset_path: str) -> List[Tuple[str, str, str]]:
    """
    Get all image files from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        List of tuples containing (root_path, subfolder, image_name)
    """
    image_files = []
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # First, check for images directly in the root directory
    for image_file in dataset_path.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append((
                str(dataset_path),
                "root",  # Use "root" as subfolder name for images in root directory
                image_file.name
            ))
    
    # Then iterate through all subdirectories
    for subfolder in dataset_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            
            # Get all image files in this subfolder
            for image_file in subfolder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append((
                        str(dataset_path),
                        subfolder_name,
                        image_file.name
                    ))
    
    logger.info(f"Found {len(image_files)} images across {len(list(dataset_path.iterdir()))} subfolders and root directory")
    return image_files


def resize_image(image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) or None for no resizing
        
    Returns:
        Resized PIL Image
    """
    if target_size is None:
        return image
    
    # Calculate the aspect ratio
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate scaling factor to maintain aspect ratio
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new image with target size and paste the resized image
    final_image = Image.new('RGB', target_size, (255, 255, 255))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    final_image.paste(resized_image, (paste_x, paste_y))
    
    return final_image


def load_huggingface_model(model_name: str, trust_remote_code: bool = True, local_model_path: str = None) -> Tuple[object, object]:
    """
    Load Hugging Face vision language model and processor.
    
    Args:
        model_name: Hugging Face model name or path
        trust_remote_code: Whether to trust remote code
        local_model_path: Optional local path to model directory. If provided, loads from local path.
        
    Returns:
        Tuple of (model, processor)
    """
    # Set up Hugging Face environment for local loading if needed
    if local_model_path or os.path.exists(model_name):
        # If local_model_path is provided, use it; otherwise check if model_name is a local path
        if local_model_path:
            # If model_name looks like a Hugging Face model ID (contains '/'), 
            # construct the path by joining local_model_path with just the model name part
            if '/' in model_name and not os.path.exists(model_name):
                model_folder_name = model_name.replace('/', '--')
                model_path = os.path.join(local_model_path, model_folder_name)
                # If that doesn't exist, try with the full model name
                if not os.path.exists(model_path):
                    model_path = os.path.join(local_model_path, model_name.split('/')[-1])
                # If still doesn't exist, try the original path
                if not os.path.exists(model_path):
                    model_path = local_model_path
            else:
                model_path = local_model_path
        else:
            model_path = model_name
            
        logger.info(f"Loading model from local path: {model_path}")
        
        # Set environment variables for offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # Verify the local path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Local model path does not exist: {model_path}")
            
    else:
        model_path = model_name
        logger.info(f"Loading model from Hugging Face Hub: {model_name}")
        
        # Ensure we're not in offline mode for Hub downloads
        if 'TRANSFORMERS_OFFLINE' in os.environ:
            del os.environ['TRANSFORMERS_OFFLINE']
        if 'HF_DATASETS_OFFLINE' in os.environ:
            del os.environ['HF_DATASETS_OFFLINE']
    
    try:
        # Try different model classes based on model name
        if 'qwen' in model_path.lower():
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                local_files_only=local_model_path is not None or os.path.exists(model_name),
            )
        elif 'gemma' in model_path.lower():
            # Use specific Gemma3nForConditionalGeneration if available
            if GEMMA3N_AVAILABLE and ('gemma-3n' in model_path.lower() or 'gemma3n' in model_path.lower()):
                model = Gemma3nForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_model_path is not None or os.path.exists(model_name),
                ).eval()
            else:
                # Fallback to AutoModelForCausalLM for other Gemma models
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_model_path is not None or os.path.exists(model_name),
                )
        else:
            # Generic approach - try Vision2Seq first, then CausalLM
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_model_path is not None or os.path.exists(model_name),
                )
            except:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_model_path is not None or os.path.exists(model_name),
                )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_model_path is not None or os.path.exists(model_name),
        )
        
        logger.info(f"Model loaded successfully on device: {model.device}")
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        raise


def prepare_input_for_model(image: Image.Image, text: str, processor: object, model_name: str) -> dict:
    """
    Prepare input for different model types.
    
    Args:
        image: PIL Image
        text: Input text/prompt
        processor: Model processor
        model_name: Name of the model
        
    Returns:
        Processed inputs dict
    """
    if 'qwen' in model_name.lower():
        # Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        
        # Apply chat template if available
        if hasattr(processor, 'apply_chat_template'):
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True
            )
    elif 'gemma' in model_name.lower():
        # Gemma3n format with image soft token
        if '<image_soft_token>' not in text:
            # Prepend the image soft token to the prompt
            text = f"<image_soft_token> {text}"
        
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )
    else:
        # Generic format for other models
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )
    
    return inputs


def infer_image_description(
    model: object, 
    processor: object, 
    image_path: str, 
    prompt: str = "Describe this image.",
    image_size: Optional[Tuple[int, int]] = None,
    model_name: str = ""
) -> str:
    """
    Run inference on a single image.
    
    Args:
        model: Loaded Hugging Face model
        processor: Model processor
        image_path: Path to the image file
        prompt: Text prompt for the model
        image_size: Optional image resize dimensions (width, height)
        model_name: Name of the model for format-specific handling
        
    Returns:
        Generated text description
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image if specified
        if image_size:
            image = resize_image(image, image_size)
        
        # Prepare inputs
        inputs = prepare_input_for_model(image, prompt, processor, model_name)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            if 'gemma' in model_name.lower():
                # Gemma-specific generation with inference mode
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None
                    )
                # Extract only the new tokens (remove input tokens)
                new_tokens = outputs[0][input_len:]
            else:
                # Standard generation for other models
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None
                )
                new_tokens = outputs[0]
        
        # Decode the response
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor
        
        # For Gemma models, decode only the new tokens
        if 'gemma' in model_name.lower():
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            # For some models, we need to extract only the new tokens
            if 'input_ids' in inputs:
                generated_text = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            else:
                generated_text = tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                ).strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return f"ERROR: {str(e)}"


def process_dataset(
    dataset_path: str,
    model_name: str,
    output_csv: str,
    prompt: str = "Describe this image.",
    image_size: Optional[Tuple[int, int]] = None,
    trust_remote_code: bool = True,
    resume: bool = False,
    local_model_path: str = None
) -> None:
    """
    Process the entire dataset and save results to CSV.
    
    Args:
        dataset_path: Path to the dataset directory
        model_name: Hugging Face model name or path
        output_csv: Path to output CSV file
        prompt: Text prompt for the model
        image_size: Optional image resize dimensions (width, height)
        trust_remote_code: Whether to trust remote code
        resume: Whether to resume from existing CSV file
        local_model_path: Optional local path to model directory for offline loading
    """
    # Get all image files
    image_files = get_image_files(dataset_path)
    
    # Check if resuming from existing file
    processed_files = set()
    if resume and os.path.exists(output_csv):
        with open(output_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle root directory images differently
                if row['subfolder'] == "root":
                    processed_files.add(os.path.join(row['root_path'], row['image_name']))
                else:
                    processed_files.add(os.path.join(row['root_path'], row['subfolder'], row['image_name']))
        logger.info(f"Resuming: {len(processed_files)} files already processed")
    
    # Filter out already processed files if resuming
    if resume:
        image_files = [
            (root, subfolder, img) for root, subfolder, img in image_files
            if (os.path.join(root, img) if subfolder == "root" else os.path.join(root, subfolder, img)) not in processed_files
        ]
        logger.info(f"Remaining files to process: {len(image_files)}")
    
    # Load the model and processor
    model, processor = load_huggingface_model(model_name, trust_remote_code, local_model_path)
    
    # Prepare CSV file
    fieldnames = ['root_path', 'subfolder', 'image_name', 'predicted_text', 'prompt_used']
    mode = 'a' if resume and os.path.exists(output_csv) else 'w'
    
    with open(output_csv, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if starting fresh
        if mode == 'w':
            writer.writeheader()
        
        # Process each image
        for root_path, subfolder, image_name in tqdm(image_files, desc="Processing images"):
            # Handle root directory images differently
            if subfolder == "root":
                image_path = os.path.join(root_path, image_name)
            else:
                image_path = os.path.join(root_path, subfolder, image_name)
            
            # Run inference
            predicted_text = infer_image_description(
                model, processor, image_path, prompt, image_size, model_name
            )
            
            # Write to CSV
            writer.writerow({
                'root_path': root_path,
                'subfolder': subfolder,
                'image_name': image_name,
                'predicted_text': predicted_text,
                'prompt_used': prompt
            })
            
            # Flush to ensure data is written
            csvfile.flush()
    
    logger.info(f"Processing complete! Results saved to {output_csv}")


def main():
    """Main function to run the dataset inference script."""
    parser = argparse.ArgumentParser(
        description="Run inference on image dataset using Hugging Face vision language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Gemma 3n model
  python dataset_inference.py --dataset_path /path/to/dataset --model_name google/gemma-3n-E4B

  # Using Qwen2-VL with custom prompt and image resizing
  python dataset_inference.py --dataset_path /path/to/dataset --model_name Qwen/Qwen2-VL-7B-Instruct \\
    --prompt "What objects are in this image?" --image_size 512 512

  # Resume interrupted processing
  python dataset_inference.py --dataset_path /path/to/dataset --model_name google/gemma-3n-E4B --resume

Popular models:
  - google/gemma-3n-E4B (Gemma 3n)
  - Qwen/Qwen2-VL-7B-Instruct (Qwen2-VL 7B)
  - Qwen/Qwen2-VL-2B-Instruct (Qwen2-VL 2B)
  - llava-hf/llava-1.5-7b-hf (LLaVA 1.5 7B)
  - HuggingFaceM4/idefics2-8b (Idefics2 8B)
        """
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the dataset directory containing image subfolders and/or images in the root directory'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='google/gemma-3n-E4B',
        help='Hugging Face model name or path (default: google/gemma-3n-E4B)'
    )
    
    parser.add_argument(
        '--local_model_path',
        type=str,
        default=None,
        help='Local path to model directory for offline loading. If specified, loads model from this path instead of downloading from Hugging Face Hub.'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default='dataset_inference_results.csv',
        help='Output CSV file path (default: dataset_inference_results.csv)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='Describe this image.',
        help='Text prompt for the model (default: "Describe this image.")'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help='Resize images to specified dimensions (width height), e.g., --image_size 512 512'
    )
    
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        default=True,
        help='Trust remote code when loading models (default: True)'
    )
    
    parser.add_argument(
        '--no_trust_remote_code',
        action='store_true',
        help='Do not trust remote code when loading models'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing CSV file (skip already processed images)'
    )
    
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List popular model names and exit'
    )
    
    args = parser.parse_args()
    
    # Handle local model path from environment variable if not provided via argument
    if not args.local_model_path and 'HF_LOCAL_MODEL_PATH' in os.environ:
        args.local_model_path = os.environ['HF_LOCAL_MODEL_PATH']
        logger.info(f"Using local model path from environment variable: {args.local_model_path}")
    
    # Handle model listing
    if args.list_models:
        print("Popular Hugging Face Vision Language Models:")
        print("=" * 50)
        for short_name, full_name in POPULAR_MODELS.items():
            print(f"{short_name:<15} : {full_name}")
        print("\nYou can use either the short name or full Hugging Face model path.")
        return
    
    # Handle trust_remote_code flag
    trust_remote_code = args.trust_remote_code and not args.no_trust_remote_code
    
    # Convert short model names to full names if needed
    model_name = args.model_name
    if model_name in POPULAR_MODELS:
        model_name = POPULAR_MODELS[model_name]
        print(f"Using model: {model_name}")
    
    # Convert image_size to tuple if provided
    image_size = tuple(args.image_size) if args.image_size else None
    if image_size:
        print(f"Images will be resized to: {image_size[0]}x{image_size[1]}")
    
    # Run the processing
    try:
        process_dataset(
            dataset_path=args.dataset_path,
            model_name=model_name,
            output_csv=args.output_csv,
            prompt=args.prompt,
            image_size=image_size,
            trust_remote_code=trust_remote_code,
            resume=args.resume,
            local_model_path=args.local_model_path
        )
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Processing failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
