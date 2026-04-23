# Sliding Window Crop Generation Guide

## Overview

The crop generation pipeline now supports two modes when `OBJECT_DETECTOR="none"`:
- **Random crops** (default): Generate random bounding boxes with configurable size and aspect ratio
- **Sliding window crops** (new): Generate systematic, overlapping crops using a sliding window pattern

Both modes use CLIP-based similarity filtering to remove near-duplicate crops, ensuring diverse training data.

## Configuration

Edit `.env` or `.env_demo` to configure crop generation:

```bash
# Choose crop generation mode
export CROP_MODE="random"              # Options: "random" or "sliding_window"

# CLIP similarity threshold for deduplication (shared between both modes)
export CLIP_SIMILARITY_THRESHOLD=0.5   # Range: 0.0-1.0; lower = more permissive, higher = stricter

# For sliding window mode only
export SLIDING_WINDOW_STRIDE_RATIO=0.3 # Stride as fraction of window width (controls overlap)
```

## Mode Comparison

### Random Crops
- **When to use**: When you want unpredictable, diverse crops covering different parts of images
- **Characteristics**:
  - Variable size (15-55% of image area) and aspect ratio (0.75-1.33)
  - Random placement
  - May miss important regions in small images
- **Configuration**:
  - `CROP_MODE="random"`
  - `CLIP_SIMILARITY_THRESHOLD` (default 0.5)

### Sliding Window Crops  
- **When to use**: When you want systematic, complete coverage of images; useful for structured analysis
- **Characteristics**:
  - Fixed window size (based on image area percentage)
  - Systematic grid-like pattern with controlled overlap
  - Guarantees coverage of all image regions
  - More predictable and reproducible
- **Configuration**:
  - `CROP_MODE="sliding_window"`
  - `SLIDING_WINDOW_STRIDE_RATIO` (default 0.3): Controls overlap
    - `0.1-0.3`: High overlap, very comprehensive coverage (~70-90% overlap)
    - `0.3-0.5`: Moderate overlap (~30-70% overlap)  
    - `0.5-0.8`: Low overlap (~20-50% overlap)
  - `CLIP_SIMILARITY_THRESHOLD` (default 0.5)

## CLIP Deduplication

Both modes use CLIP to filter out visually similar crops:

1. **Candidate generation**: Mode generates candidate crops (random or sliding window)
2. **CLIP encoding**: Crops are encoded using OpenAI CLIP (ViT-B/32) in batches
3. **Greedy filtering**: Candidates are filtered greedily:
   - Accept first candidate
   - For each subsequent candidate, compute CLIP cosine similarity to all accepted crops
   - If max similarity >= `CLIP_SIMILARITY_THRESHOLD`, skip candidate
   - Otherwise, add candidate to accepted set

**Threshold interpretation**:
- `0.3-0.4`: Very strict (removes most similar crops; small, diverse dataset)
- `0.5-0.6`: Moderate (recommended range; balances diversity and quantity)
- `0.7-0.8`: Permissive (keeps more crops; allows some visual similarity)
- `0.9+`: Very permissive (minimal filtering)

## Usage Examples

### Setup 1: Balanced Random Crops (Default)
```bash
export OBJECT_DETECTOR="none"
export CROP_MODE="random"
export CLIP_SIMILARITY_THRESHOLD=0.5
# Other CLIP vars use defaults
```
**Result**: Random crops with moderate deduplication

### Setup 2: Systematic Complete Coverage
```bash
export OBJECT_DETECTOR="none"
export CROP_MODE="sliding_window"
export SLIDING_WINDOW_STRIDE_RATIO=0.4
export CLIP_SIMILARITY_THRESHOLD=0.5
```
**Result**: Sliding window crops with 40% stride (60% overlap), CLIP filtered

### Setup 3: Very Diverse Crops
```bash
export OBJECT_DETECTOR="none"
export CROP_MODE="random"
export CLIP_SIMILARITY_THRESHOLD=0.3
```
**Result**: Random crops with strict CLIP deduplication (very diverse, smaller dataset)

### Setup 4: Minimal Deduplication
```bash
export OBJECT_DETECTOR="none"
export CROP_MODE="sliding_window"
export SLIDING_WINDOW_STRIDE_RATIO=0.5
export CLIP_SIMILARITY_THRESHOLD=0.8
```
**Result**: Sliding window crops with low overlap and permissive deduplication

## Running with Sliding Window

```bash
# Set configuration
export CROP_MODE="sliding_window"
export SLIDING_WINDOW_STRIDE_RATIO=0.3

# Run step 2 (crops generation)
python scripts/run_full_pipeline.py --only-step 2

# Or run entire pipeline
python scripts/run_full_pipeline.py
```

## Implementation Details

### Sliding Window Algorithm

1. **Window sizing**: 
   - Default range: 15-55% of total image area
   - Uses midpoint of range for consistency (35% of area)
   - Maintains square aspect ratio

2. **Stride calculation**:
   - Stride = window_width × stride_ratio
   - Generates grid pattern: (0,0), (stride,0), (2×stride,0), ...

3. **Edge handling**:
   - Adds crops at right and bottom edges if remaining space
   - Ensures no gaps in coverage

4. **CLIP filtering**:
   - Batches: up to 32 crops per CLIP encoding batch
   - Processes progressively: generate batch → encode → filter → repeat until target count reached

### Code Structure

- `_sliding_window_bboxes_for_image()`: Generate all possible sliding window bboxes
- `_build_sliding_window_pairs_for_image()`: Generate crops + apply CLIP filtering
- Main pipeline: Calls appropriate builder based on `crop_mode` environment variable

### Performance

- **Random crops**: ~1-3 seconds per 100 images (no compute overhead)
- **Sliding window**: ~2-5 seconds per 100 images (depends on image size and stride)
- **CLIP encoding**: ~0.5-2 seconds per 1000 crops (GPU accelerated if available)

Total per-image overhead with both modes: <100ms when GPU available, <500ms on CPU

## Testing

Run unit tests:

```bash
python -m unittest tests.test_crops_to_json_unit.TestSlidingWindowBboxes -v
```

Tests verify:
- Bbox generation covers image completely
- Window sizes stay within configuration range
- Small image edge cases handled correctly

## Troubleshooting

**Issue**: "CLIP model not found"
- **Solution**: Ensure CLIP is installed: `pip install git+https://github.com/openai/CLIP.git`

**Issue**: Memory errors with large images
- **Solution**: Reduce batch size or image size via `--image_size_width` parameter

**Issue**: Too many/too few crops generated
- **Solution**: Adjust `CLIP_SIMILARITY_THRESHOLD` (lower = fewer crops) or `SLIDING_WINDOW_STRIDE_RATIO` (higher = fewer crops)

**Issue**: GPU not used for CLIP
- **Solution**: Check `CLIP_DEVICE` environment variable; defaults to CUDA if available, CPU otherwise
  ```bash
  export CLIP_DEVICE="cuda:0"  # Force GPU
  ```

## See Also

- [Main README](README.md)
- [Original issue with random overlap](plan-segmentationMasksYoloIntegration.prompt.md)
- [CLIP documentation](https://github.com/openai/CLIP)
