# XL-VLMS Pipeline API Documentation

FastAPI application for Vision-Language Model (VLM) concept explanation and grounded graph visualization.

**Base URL:** `http://iml-cube.sb.dfki.de:8000`  
**Interactive Docs:** `http://iml-cube.sb.dfki.de:8000/docs`  
**ReDoc:** `http://iml-cube.sb.dfki.de:8000/redoc`

---

## Table of Contents

1. [Setup](#setup)
2. [Endpoints](#endpoints)
   - [GET `/` - API Info](#get----api-info)
   - [POST `/run` - Fast Explainer](#post-run---fast-explainer)
   - [POST `/run-full` - Full Pipeline](#post-run-full---full-pipeline)
   - [POST `/crop` - Crop Image](#post-crop---crop-image)
   - [GET `/concept-projection/config` - Projection Config](#get-concept-projectionconfig---projection-configuration)
   - [POST `/concept-projection/run` - Run Projection](#post-concept-projectionrun---run-projection)
3. [Response Formats](#response-formats)
4. [Error Handling](#error-handling)
5. [Examples](#examples)

---

## Setup

### Prerequisites

- Python 3.8+
- Conda environment (recommended)
- CUDA-capable GPU (for model inference)

### Installation

```bash
# Activate conda environment
conda activate your_env_name

# Install dependencies
pip install -r api/requirements.txt

# Run API
cd api
python main.py
```

The API will start at `http://localhost:8000` (or `http://0.0.0.0:8000` for external access).

---

## Endpoints

### GET `/` - API Info

Get API status and information.

**Request:**
```bash
curl http://iml-cube.sb.dfki.de:8000/
```

**Response:**
```json
{
  "message": "XL-VLMS Pipeline API",
  "usage": "POST /run with an image file to run the pipeline and get JSON results",
  "status": "idle",
  "model_loaded": true
}
```

**Fields:**
- `status`: `"idle"` or `"running"` - Current pipeline status
- `model_loaded`: `true` if the VLM model is loaded in memory

---

### POST `/run` - Fast Explainer ⚡

**Fast endpoint** that explains images using pre-built concept vectors. **Recommended for most use cases.**

- **Speed:** ~30 seconds
- **Uses:** Pre-built concepts from `combined_concept_snmf_raw.pth`
- **Model:** Stays loaded in memory for faster subsequent requests

**Request:**
```bash
curl -X POST "http://iml-cube.sb.dfki.de:8000/run" \
  -F "file=@image.jpg" \
  -F "prompt_mode=unsupervised" \
  -F "top_n=5"
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | ✅ Yes | - | Image file to explain (JPEG, PNG, etc.) |
| `prompt_mode` | string | No | `"unsupervised"` | VLM explainer mode: `"unsupervised"`, `"binary"`, or `"mcq"` |
| `label` | string | No | - | Label for binary mode (e.g., `"cat"`) |
| `choices` | string | No | - | Comma-separated choices for MCQ mode (e.g., `"cat,dog,bird"`) |
| `top_n` | integer | No | `5` | Number of top concepts to return per token |
| `prompt` | string | No | - | Custom prompt to identify objects (optional) |
| `projection_method` | string | No | - | Dimensionality reduction method: `"pca"` or `"umap"` |
| `projection_dim` | integer | No | - | Projection dimensions: `2` or `3` |

**Prompt Modes:**

1. **`unsupervised`** (default)
   - General object detection and classification
   - No additional parameters needed

2. **`binary`**
   - Yes/no classification
   - Requires `label` parameter
   - Example: `prompt_mode=binary&label=cat`

3. **`mcq`** (Multiple Choice Question)
   - Choose from a list of options
   - Requires `choices` parameter (comma-separated)
   - Example: `prompt_mode=mcq&choices=cat,dog,bird,fish`

**Response:**
```json
{
  "success": true,
  "message": "VLM explainer completed successfully",
  "nodes": [...],
  "links": [...],
  "textualOutput": "fruits and vegetables...",
  "colorMap": {
    "It": "#E57373",
    "looks": "#F06292",
    "diagram": "#BA68C8"
  }
}
```

**Response Fields:**

- `nodes`: Array of graph nodes with concept embeddings, positions, and grounding
- `links`: Array of graph links with similarity values and colored RGBA format
- `textualOutput`: Model-generated text output
- `colorMap`: Mapping of token text to hex colors for visualization

**Features:**

- ✅ **Unique concept_index**: Each token gets a unique `concept_index` (based on `token_index`) to ensure all tokens appear in the colored graph
- ✅ **Image resizing**: Automatically resizes/crops images to 512x512
- ✅ **Model caching**: Model stays loaded in memory for faster requests
- ✅ **Optional projections**: Can compute PCA/UMAP projections for token embeddings

**Example with Binary Mode:**
```bash
curl -X POST "http://iml-cube.sb.dfki.de:8000/run" \
  -F "file=@image.jpg" \
  -F "prompt_mode=binary" \
  -F "label=cat" \
  -F "top_n=5"
```

**Example with Projections:**
```bash
curl -X POST "http://iml-cube.sb.dfki.de:8000/run" \
  -F "file=@image.jpg" \
  -F "projection_method=pca" \
  -F "projection_dim=2"
```

---

### POST `/run-full` - Full Pipeline ⚠️

**Slow endpoint** that generates concepts from scratch. Use only when you need custom concepts.

- **Speed:** ~30 minutes
- **Generates:** New concepts from your image
- **Use case:** When you need domain-specific concepts not in the pre-built set

**Request:**
```bash
curl -X POST "http://iml-cube.sb.dfki.de:8000/run-full" \
  -F "file=@image.jpg" \
  -F "prompt=What objects are visible in this image?" \
  -F "prompt_mode=unsupervised"
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | ✅ Yes | - | Image file to process |
| `prompt` | string | No | - | Custom prompt for dataset inference |
| `prompt_mode` | string | No | `"unsupervised"` | VLM explainer mode |
| `label` | string | No | - | Label for binary mode |
| `choices` | string | No | - | Comma-separated choices for MCQ mode |

**Response:**
```json
{
  "success": true,
  "message": "Pipeline completed successfully",
  "prompt_used": "What objects are visible in this image?",
  "prompt_mode": "unsupervised",
  "concept_data": { ... },
  "vlm_explanations_data": { ... }
}
```

**What Happens:**

1. Clears `data/train/` directory
2. Saves image to `data/train/<image_name>/`
3. Runs full pipeline script:
   - Dataset inference → concept discovery
   - Build crops JSON
   - Generate features
   - Decompose features → creates concept vectors
   - VLM explainer
   - Evaluation
4. Converts PTH files to JSON
5. Returns concept data and explanations

**Note:** Only one pipeline can run at a time. Returns `409 Conflict` if another request is in progress.

---

### POST `/crop` - Crop Image

Crop an image from `data/train-og/` using bounding boxes.

**Request:**
```bash
curl -X POST "http://iml-cube.sb.dfki.de:8000/crop" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "flecked/flecked_0052.jpg",
    "bboxes": [[100, 200, 300, 400]]
  }'
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | string | ✅ Yes | Relative path under `data/train-og/` (e.g., `"concept/image.jpg"`) |
| `bboxes` | array | No | List of bounding boxes `[[x1, y1, x2, y2], ...]`. If omitted, returns full image. |

**Response:**
- **Content-Type:** `image/jpeg`
- **Body:** Cropped image as JPEG bytes

**Path Format:**
- ✅ Valid: `"flecked/flecked_0052.jpg"`
- ❌ Invalid: `"/flecked/flecked_0052.jpg"` (no leading slash)
- ❌ Invalid: `"../flecked/flecked_0052.jpg"` (no path traversal)

**Bounding Boxes:**
- Format: `[x1, y1, x2, y2]` (top-left and bottom-right corners)
- Multiple boxes: Union of all boxes is used for cropping
- Empty array: Returns full image

**Example:**
```bash
# Crop with single bounding box
curl -X POST "http://iml-cube.sb.dfki.de:8000/crop" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "flecked/flecked_0052.jpg", "bboxes": [[100, 200, 300, 400]]}' \
  --output cropped.jpg

# Get full image (no bboxes)
curl -X POST "http://iml-cube.sb.dfki.de:8000/crop" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "flecked/flecked_0052.jpg"}' \
  --output full.jpg
```

---

### GET `/concept-projection/config` - Projection Configuration

Get available configuration options for dimensionality reduction methods (PCA, LDA, UMAP).

**Request:**
```bash
curl http://iml-cube.sb.dfki.de:8000/concept-projection/config
```

**Response:**
```json
{
  "success": true,
  "message": "Concept projection configuration.",
  "config": {
    "source_file": "/path/to/combined_concept_snmf_gl.pth",
    "data_shape": {
      "n_concepts": 100,
      "n_features": 2048
    },
    "methods": {
      "pca": {
        "description": "Principal Component Analysis on the concept embedding matrix.",
        "params": {
          "n_components": {
            "type": "int",
            "min": 1,
            "max": 3,
            "default": 2
          }
        }
      },
      "lda": {
        "description": "Linear Discriminant Analysis on concept embeddings.",
        "params": {
          "n_components": {
            "type": "int",
            "min": 1,
            "max": 2,
            "default": 1
          }
        }
      }
    }
  }
}
```

**Use Case:** Check available projection methods and their parameter ranges before calling `/concept-projection/run`.

---

### POST `/concept-projection/run` - Run Projection

Apply dimensionality reduction (PCA/LDA) to concept embeddings and generate a graph structure.

**Request:**
```bash
curl -X POST "http://iml-cube.sb.dfki.de:8000/concept-projection/run" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "pca",
    "n_components": 3
  }'
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `method` | string | ✅ Yes | - | Projection method: `"pca"` or `"lda"` |
| `n_components` | integer | No | `3` | Number of components (1-3). For 2D, always uses 2. For 3D, uses this value. |

**Response:**
```json
{
  "success": true,
  "message": "Concept graph processed successfully.",
  "method": "pca",
  "params": {
    "n_components_2d": 2,
    "n_components_3d": 3,
    "similarity_threshold": 0.95
  },
  "cached": false,
  "cache_file": "/path/to/cache.json",
  "result": {
    "nodes": [
      {
        "id": 0,
        "name": "concept_0",
        "x2d": [0.123, 0.456],
        "y2d": [0.789, 0.012],
        "x3d": [0.1, 0.2, 0.3],
        "y3d": [0.4, 0.5, 0.6],
        "z3d": [0.7, 0.8, 0.9],
        "images": [...],
        "bboxes": [...]
      }
    ],
    "links": [
      {
        "source": 0,
        "target": 1,
        "value": 0.85,
        "color": "rgba(0, 254, 0, 0.923)"
      }
    ]
  }
}
```

**Response Fields:**

- `nodes`: Array of concept nodes with:
  - `id`: Concept index
  - `name`: Concept name
  - `x2d`, `y2d`: 2D PCA coordinates
  - `x3d`, `y3d`, `z3d`: 3D PCA coordinates
  - `images`: Image grounding paths
  - `bboxes`: Bounding boxes
- `links`: Array of concept similarity links with:
  - `source`, `target`: Node IDs
  - `value`: Cosine distance (1 - similarity)
  - `color`: RGBA color with alpha based on similarity strength

**Caching:**

- Results are cached based on configuration
- Identical requests return cached results (faster)
- Cache key: `method-{method}_nc3-{n_components}_thr-{threshold}`

**Example:**
```bash
# PCA with 3 components
curl -X POST "http://iml-cube.sb.dfki.de:8000/concept-projection/run" \
  -H "Content-Type: application/json" \
  -d '{"method": "pca", "n_components": 3}'

# LDA with 2 components
curl -X POST "http://iml-cube.sb.dfki.de:8000/concept-projection/run" \
  -H "Content-Type: application/json" \
  -d '{"method": "lda", "n_components": 2}'
```

---

## Response Formats

### VLM Explanations Data

Structure returned by `/run` and `/run-full`:

```json
{
  "vlm_explanations_data": {
    "model_card": "Qwen/Qwen2.5-VL-3B-Instruct",
    "layer_path": "model.language_model.norm",
    "results": [
      {
        "image_path": "input/image.jpg",
        "model_output": "fruits and vegetables...",
        "generated_token_ids": [1626, 11797, ...],
        "per_token_concepts": [
          {
            "token_index": 0,
            "token_id": 1626,
            "token_text": "fr",
            "concept_index": 0,
            "top_concepts": [
              {
                "rank": 1,
                "concept_index": 3,
                "similarity": 0.267,
                "distance": 0.733,
                "text_grounding": ["kitchen", "pot"],
                "image_grounding_path": [...],
                "image_grounding_bboxes": [[75, 173, 275, 373], ...]
              }
            ]
          }
        ],
        "top_concepts_over_sequence": [...]
      }
    ]
  }
}
```

**Key Fields:**

- `concept_index`: **Unique per token** (uses `token_index`) - ensures all tokens appear in colored graph
- `top_concepts`: Array of matched concepts with similarity scores
- `text_grounding`: Text tokens that ground the concept
- `image_grounding_path`: Image paths showing the concept
- `image_grounding_bboxes`: Bounding boxes in images

---

## Error Handling

### Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid parameters or file format |
| `404` | Not Found | Resource not found (e.g., concept file) |
| `409` | Conflict | Pipeline already running (only one at a time) |
| `500` | Internal Error | Server error during processing |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

**409 Conflict - Pipeline Busy:**
```json
{
  "detail": "Another request is already running. Please wait for completion."
}
```
**Solution:** Wait for the current request to finish, or use `/run` (faster, doesn't block).

**400 Bad Request - Invalid File:**
```json
{
  "detail": "File must be an image"
}
```
**Solution:** Ensure you're uploading a valid image file (JPEG, PNG, etc.).

**404 Not Found - Concept File Missing:**
```json
{
  "detail": "Concept file not found: /path/to/combined_concept_snmf_raw.pth"
}
```
**Solution:** Ensure the concept file exists. Run `/run-full` once to generate it, or check file paths.

---

## Examples

### Python

```python
import requests

# Fast explainer
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://iml-cube.sb.dfki.de:8000/run",
        files={"file": f},
        data={
            "prompt_mode": "unsupervised",
            "top_n": 5
        }
    )

data = response.json()
print(f"Success: {data['success']}")
print(f"Nodes: {len(data.get('nodes', []))}")
print(f"Links: {len(data.get('links', []))}")
print(f"ColorMap: {data.get('colorMap', {})}")
```

### JavaScript/TypeScript

```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('prompt_mode', 'unsupervised');
formData.append('top_n', '5');

const response = await fetch('http://iml-cube.sb.dfki.de:8000/run', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log('Nodes:', data.nodes.length);
console.log('Links:', data.links.length);
console.log('ColorMap:', data.colorMap);
```

### cURL - Complete Example

```bash
# 1. Check API status
curl http://iml-cube.sb.dfki.de:8000/

# 2. Fast explainer (recommended)
curl -X POST "http://iml-cube.sb.dfki.de:8000/run" \
  -F "file=@my_image.jpg" \
  -F "prompt_mode=unsupervised" \
  -F "top_n=5" \
  -o result.json

# 3. Get projection config
curl http://iml-cube.sb.dfki.de:8000/concept-projection/config

# 4. Run projection
curl -X POST "http://iml-cube.sb.dfki.de:8000/concept-projection/run" \
  -H "Content-Type: application/json" \
  -d '{"method": "pca", "n_components": 3}' \
  -o graph.json
```

---

## Notes

### Performance

- **`/run`**: ~30 seconds (uses pre-built concepts, model stays in memory)
- **`/run-full`**: ~30 minutes (generates concepts from scratch)
- **Model Loading**: First request to `/run` may take longer (model loads into memory)
- **Caching**: Concept projections are cached for faster subsequent requests

### Concurrency

- Only **one request** can run at a time (prevents resource conflicts)
- Concurrent requests return `409 Conflict`
- Use `/run` for faster turnaround (doesn't block as long)

### File Paths

- **Input images**: Automatically resized to 512x512
- **Output**: Saved to `outputs/api_runs/` (for `/run-full`)
- **Concepts**: Loaded from `outputs/screen_run/concept/snmf/combined_concept_snmf_raw.pth`

### Unique Concept IDs

- Each token in `/run` response has a **unique `concept_index`** (based on `token_index`)
- This ensures all tokens appear in the colored graph visualization
- Previously, duplicate `concept_index` values caused some tokens to be skipped

---

## Testing

Run the test suite:

```bash
cd api
python test_api.py
```

Or test individual endpoints:

```bash
# Test fast endpoint
curl -X POST "http://localhost:8000/run" \
  -F "file=@../data/grids/grid_4_1__orange__banana__cat__dog.jpg"
```

---

## Support

For issues or questions:
- Check the interactive docs: `http://iml-cube.sb.dfki.de:8000/docs`
- Review server logs for detailed error messages
- Ensure all required files (concept PTH files) are present
