#!/usr/bin/env python3
"""
GLIMPSE Example Usage Script
Demonstrates how to use GLIMPSE explainer with real Hugging Face models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq, 
    AutoModelForCausalLM, 
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor
)
import argparse
import os
import sys
from pathlib import Path

# Add src to path to import GLIMPSE explainer
script_dir = Path(__file__).parent
src_dir = script_dir / "glimpse"
sys.path.append(str(src_dir))

# Import the GLIMPSE explainer
from glimpse_explainer import GLIMPSEExplainer


def load_sample_image(image_path_or_url: str = None) -> np.ndarray:
    """Load a sample image for testing."""
    
    if image_path_or_url is None:
        # Use a default sample image URL
        image_path_or_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    
    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
        else:
            # Load from local path
            image = Image.open(image_path_or_url)
        
        # Convert to RGB and numpy array
        image = image.convert('RGB')
        return np.array(image)
        
    except Exception as e:
        print(f"Could not load image: {e}")
        # Return random image as fallback
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


def run_glimpse_example(
    model_name: str = "microsoft/git-base",
    prompt: str = "Describe what you see in this image.",
    image_path: str = None,
    output_dir: str = "glimpse_outputs",
    device: str = "auto",
    auto_quantize: bool = True,
    memory_margin: float = 2.0,
    force_4bit: bool = False
):
    """
    Run GLIMPSE explanation on a vision-language model.
    
    Args:
        model_name: Hugging Face model name
        prompt: Text prompt for the model
        image_path: Path or URL to image
        output_dir: Directory to save outputs
        device: Device to use ('auto', 'cpu', 'cuda')
    """
    
    print(f"GLIMPSE Example with {model_name}")
    print("=" * 60)
    
    # Setup device
    import sys as _sys
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from device_utils import get_device_config  # type: ignore
    _dc = get_device_config(device if device != 'auto' else None)
    device = _dc.primary_device
    print(f"Using device: {device} (gpu_ids={_dc.gpu_ids})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model and processor
        print("Loading model and processor...")
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except ValueError as e:
            if "size must contain" in str(e):
                print(f"⚠️  Processor loading failed: {e}")
                print("🔄 Trying to load processor with default image size config...")
                
                # Try loading with explicit image processor config
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    image_processor = AutoImageProcessor.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        size={"shortest_edge": 224, "longest_edge": 224}
                    )
                    
                    # Create a simple processor wrapper
                    class SimpleProcessor:
                        def __init__(self, tokenizer, image_processor):
                            self.tokenizer = tokenizer
                            self.image_processor = image_processor
                        
                        def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
                            result = {}
                            if text:
                                text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
                                result.update(text_inputs)
                            if images:
                                image_inputs = self.image_processor(images, return_tensors=return_tensors)
                                result.update(image_inputs)
                            return result
                    
                    processor = SimpleProcessor(tokenizer, image_processor)
                    print("✓ Created custom processor wrapper")
                except Exception as fallback_e:
                    print(f"❌ Fallback processor creation failed: {fallback_e}")
                    raise e
            else:
                raise e
        
        # Load model with appropriate class
        try:
            # First try with AutoModelForVision2Seq (most vision-language models)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True
            )
            if device.type == "cuda":
                model = model.to(device)
        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                print(f"⚠️  Model {model_name} is not supported by AutoModelForVision2Seq")
                print("🔄 Trying alternative loading methods...")
                
                # Try with trust_remote_code and AutoModel
                try:
                    model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                        trust_remote_code=True
                    )
                    if device.type == "cuda":
                        model = model.to(device)
                    print(f"✓ Loaded {model_name} using AutoModel")
                except Exception as e2:
                    print(f"❌ Failed to load {model_name}: {e2}")
                    print("\n💡 Supported models include:")
                    print("  - microsoft/Phi-3.5-vision-instruct")
                    print("  - Qwen/Qwen2-VL-2B-Instruct")
                    print("  - Qwen/Qwen2-VL-7B-Instruct")
                    print("  - HuggingFaceM4/idefics2-8b")
                    print("  - microsoft/git-base")
                    raise e2
                    print("  - HuggingFaceM4/idefics2-8b")
                    print("  - llava-hf/llava-1.5-7b-hf")
                    raise e2
            else:
                raise e
            
        if device.type == "cuda":
            model = model.to(device)
        
        print(f"✓ Loaded {model_name}")
        
        # Load image
        print("Loading image...")
        image_array = load_sample_image(image_path)
        image_pil = Image.fromarray(image_array)
        print(f"✓ Loaded image with shape: {image_array.shape}")
        
        # Process inputs using the same logic as dataset_inference.py
        print("Processing inputs...")
        print(f"Model name for input processing: {model_name}")
        
        # Use the prepare_input_for_model function logic
        if 'qwen' in model_name.lower():
            # Qwen2-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {"type": "text", "text": prompt},
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
                    images=[image_pil],
                    return_tensors="pt",
                    padding=True
                )
            else:
                inputs = processor(
                    text=prompt,
                    images=image_pil,
                    return_tensors="pt",
                    padding=True
                )
        elif 'gemma' in model_name.lower():
            # Gemma3n format using chat template
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {"type": "text", "text": prompt}
                    ],
                },
            ]
            
            # Apply chat template and process inputs
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        elif 'idefics' in model_name.lower() or 'smolvlm' in model_name.lower():
            # Idefics and SmolVLM models require image tokens in the text
            # Use the special image token <image> to mark where the image should be placed
            text_with_image = f"<image>{prompt}"
            inputs = processor(
                text=text_with_image,
                images=[image_pil],
                return_tensors="pt",
                padding=True
            )
        elif 'llava' in model_name.lower():
            # LLaVA models often use USER/ASSISTANT format
            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = processor(
                text=formatted_prompt,
                images=image_pil,
                return_tensors="pt",
                padding=True
            )
        elif 'internvl' in model_name.lower():
            # InternVL models use different input format
            try:
                # Try the standard InternVL format
                inputs = processor(
                    images=image_pil,
                    text=prompt,
                    return_tensors="pt"
                )
                print("✓ Used InternVL format")
            except Exception as e:
                print(f"InternVL format failed: {e}")
                # Fallback to basic text-only processing
                inputs = processor(
                    text=prompt,
                    return_tensors="pt",
                    padding=True
                )
                print("✓ Used fallback text-only format")
        else:
            # Generic format for other models - try both approaches
            try:
                # First try with image token placeholder
                text_with_image = f"<image>{prompt}"
                inputs = processor(
                    text=text_with_image,
                    images=[image_pil],
                    return_tensors="pt",
                    padding=True
                )
                print("✓ Used generic format with image token")
            except (ValueError, TypeError) as e:
                if "number of images" in str(e).lower():
                    # Try without image token
                    inputs = processor(
                        text=prompt,
                        images=image_pil,
                        return_tensors="pt",
                        padding=True
                    )
                    print("✓ Used generic format without image token")
                else:
                    print(f"❌ Generic processing failed: {e}")
                    raise e
        
        # Move inputs to device, preserve integer dtypes for embeddings
        model_dtype = next(model.parameters()).dtype
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype in [torch.int32, torch.int64, torch.long]:
                    # Keep integer types as-is for embedding layers
                    processed_inputs[k] = v.to(device)
                elif v.dtype.is_floating_point:
                    # Convert floating point to model dtype
                    processed_inputs[k] = v.to(device, dtype=model_dtype)
                else:
                    # Keep other types as-is
                    processed_inputs[k] = v.to(device)
        
        inputs = processed_inputs
        print(f"✓ Processed inputs: {list(inputs.keys())} (model dtype: {model_dtype})")
        
        # Do generation of output and print the output

        print("Generating model output...")
        with torch.no_grad():
            if hasattr(model, 'generate'):
                generated_ids = model.generate(**inputs, max_new_tokens=50)
                if hasattr(processor, 'tokenizer'):
                    output_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:
                    output_text = processor.decode(generated_ids[0], skip_special_tokens=True)
            else:
                # For models without generate method, do a forward pass
                outputs = model(**inputs)
                if hasattr(processor, 'tokenizer'):
                    output_text = processor.tokenizer.batch_decode(torch.argmax(outputs.logits, dim=-1), skip_special_tokens=True)[0]
                else:
                    output_text = processor.decode(torch.argmax(outputs.logits, dim=-1)[0], skip_special_tokens=True)   
        print(f"✓ Model output: {output_text}")    

        

        # Initialize GLIMPSE explainer with memory management
        print("Initializing GLIMPSE explainer...")
        explainer = GLIMPSEExplainer(
            model=model,
            tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
            lambda_head=1.0,
            lambda_depth=0.1,
            lambda_flow=0.5,
            device=device
        )
        print("✓ GLIMPSE explainer initialized")
        
        # Determine visual tokens (model-specific)
        if "git" in model_name.lower():
            visual_tokens = 197  # GIT model typically uses 197 visual tokens
        elif "blip" in model_name.lower():
            visual_tokens = 577  # BLIP models
        elif "qwen" in model_name.lower():
            # Qwen models use variable patches based on image size
            visual_tokens = 256  # Default for Qwen2.5-VL
        elif "smolvlm" in model_name.lower():
            visual_tokens = 196  # SmolVLM uses 14x14 patches = 196 tokens
        elif "internvl" in model_name.lower():
            visual_tokens = 256  # InternVL typically uses 256 visual tokens
        else:
            visual_tokens = 196  # Default for 14x14 patches
        
        print(f"Using {visual_tokens} visual tokens for {model_name}")
        
        # Run GLIMPSE explanation
        print("Running GLIMPSE explanation...")
        try:
            explanations = explainer.interpret(
                inputs=inputs,  # Pass the entire inputs dict
                attention_mask=None,  # Already included in inputs
                pixel_values=None,   # Already included in inputs
                generated_tokens=None,
                visual_tokens=visual_tokens,
                target_token_idx=-1
            )
        except Exception as e:
            print(f"❌ Error during GLIMPSE analysis: {e}")
            raise
        
        print("✓ GLIMPSE explanation completed!")
        print(f"Generated explanation keys: {list(explanations.keys())}")
        
        # Print summary statistics
        if 'visual_saliency' in explanations:
            vs = explanations['visual_saliency'].cpu().numpy()
            print(f"Visual saliency range: [{vs.min():.4f}, {vs.max():.4f}]")
        
        if 'prompt_saliency' in explanations:
            ps = explanations['prompt_saliency'].cpu().numpy()
            print(f"Prompt saliency range: [{ps.min():.4f}, {ps.max():.4f}]")
        
        # Save explanations
        explanation_path = os.path.join(output_dir, "explanations.pth")
        torch.save(explanations, explanation_path)
        print(f"✓ Saved explanations to: {explanation_path}")
        
        # Generate visualization
        print("Generating visualization...")
        viz_path = os.path.join(output_dir, "glimpse_visualization.png")
        explainer.visualize_explanations(
            explanations=explanations,
            input_text=prompt,
            image=image_array,
            save_path=viz_path,
            #show_plot=False
        )
        print(f"✓ Saved visualization to: {viz_path}")
        
        # Save input image for reference
        image_save_path = os.path.join(output_dir, "input_image.jpg")
        image_pil.save(image_save_path)
        print(f"✓ Saved input image to: {image_save_path}")
        
        # Create summary report
        report_path = os.path.join(output_dir, "glimpse_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"GLIMPSE Explanation Report\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Visual tokens: {visual_tokens}\n\n")
            
            f.write(f"Explanation Statistics:\n")
            f.write(f"- Sequence length: {explanations.get('sequence_length', 'N/A')}\n")
            f.write(f"- Number of layers: {explanations.get('num_layers', 'N/A')}\n")
            
            if 'visual_saliency' in explanations:
                vs = explanations['visual_saliency'].cpu().numpy()
                f.write(f"- Visual saliency: min={vs.min():.4f}, max={vs.max():.4f}, mean={vs.mean():.4f}\n")
            
            if 'prompt_saliency' in explanations:
                ps = explanations['prompt_saliency'].cpu().numpy()
                f.write(f"- Prompt saliency: min={ps.min():.4f}, max={ps.max():.4f}, mean={ps.mean():.4f}\n")
            
            if 'cross_modal_relevance' in explanations:
                cr = explanations['cross_modal_relevance'].cpu().numpy()
                f.write(f"- Cross-modal relevance: min={cr.min():.4f}, max={cr.max():.4f}, mean={cr.mean():.4f}\n")
        
        print(f"✓ Saved report to: {report_path}")
        
        print("\n🎉 GLIMPSE analysis completed successfully!")
        print(f"All outputs saved to: {output_dir}")
        
        return explanations
        
    except Exception as e:
        print(f"❌ Error during GLIMPSE analysis: {str(e)}")
        raise


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(description="GLIMPSE Vision-Language Model Explainer")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="microsoft/git-base",
        help="Hugging Face model name (default: microsoft/git-base)"
    )
    
    parser.add_argument(
        "--prompt", 
        type=str,
        default="Describe what you see in this image.",
        help="Text prompt for the model"
    )
    parser.add_argument(
        "--target", 
        type=str,
        default="Describe what you see in this image.",
        help="Text prompt for the model"
    )
    
    parser.add_argument(
        "--image", 
        type=str,
        default=None,
        help="Path or URL to image (default: sample image)"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        default="glimpse_outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device", 
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for computation"
    )
    
    parser.add_argument(
        "--no-quantize", 
        action="store_true",
        help="Disable automatic quantization even if memory is insufficient"
    )
    
    parser.add_argument(
        "--force-4bit", 
        action="store_true",
        help="Force 4-bit quantization regardless of memory availability"
    )
    
    parser.add_argument(
        "--memory-margin", 
        type=float,
        default=2.0,
        help="Memory safety margin in GB (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Run GLIMPSE example
    explanations = run_glimpse_example(
        model_name=args.model,
        prompt=args.prompt,
        image_path=args.image,
        output_dir=args.output,
        device=args.device,
        auto_quantize=not args.no_quantize,
        memory_margin=args.memory_margin,
        force_4bit=args.force_4bit
    )
    
    return explanations


if __name__ == "__main__":
    main()
