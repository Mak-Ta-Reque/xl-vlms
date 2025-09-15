# glimpse_local_image.py
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(attention_map, save_path="attention.png"):
    """Visualize attention map as a heatmap."""
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path)
    print(f"Attention visualization saved to {save_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", type=str, default="What is in the image?")
    parser.add_argument("--target", type=str, default="dishes")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="glimpse_local_image")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(args.model).to(device)
    model.eval()

    # Load image
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Assume last cross-attention layer
    cross_attentions = outputs.cross_attentions[-1]  # shape: (batch, num_heads, query_len, kv_len)
    target_index = 0  # simple example: first token of the prompt
    attention_map = cross_attentions[0, 0, target_index].cpu().numpy()
    attention_map = attention_map.reshape((image.size[1], image.size[0]))  # reshape to image

    # Visualize
    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, "attention.png")
    visualize_attention(attention_map, save_path)

if __name__ == "__main__":
    main()
