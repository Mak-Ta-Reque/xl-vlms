# XL-VLM Demo — Interactive VLM Concept Explorer

Upload an image, ask a question, click detected nouns to reveal which learned
visual concepts the model activates.

---

## GPU Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GPU with ≥ 12 GB VRAM | NVIDIA RTX 3090 (24 GB) or better |
| **CUDA** | 12.6+ | 12.6 |
| **Driver** | ≥ 535.x | ≥ 550.x |

The **Qwen2.5-VL-3B-Instruct** model loads in ~6 GB VRAM at fp16.  
With activation hooks + concept scoring the peak usage is **~10–12 GB**.  
A 24 GB card (RTX 3090 / 4090 / A5000) gives comfortable headroom.

---

## Quick Start — Docker (recommended)

### Prerequisites

- Docker ≥ 20.10
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-docker2` or `nvidia-container-toolkit`)
- Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`

### 1. Build the image

```bash
cd xl-vlms
docker build -t xl-vlm-demo .
# The image will be named 'xl-vlm-demo'.
# (Optional) Use 'docker scan xl-vlm-demo' to check for vulnerabilities.
```

Build takes ~10–15 min the first time (downloads PyTorch, spaCy model, etc.).

### 2. Run the demo

```bash
docker run --rm --gpus all \
  -p 8501:8501 \
  -v /path/to/huggingface/hub:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  xl-vlm-demo
```
```
sudo docker run --gpus all -p 8501:8501 --name xl-vlm-demo --rm \
    -v /mnt/abka03/xlvlm_data:/mnt/abka03/xlvlm_data:ro \
    -v /mnt/abka03/huggingface/hub:/root/.cache/huggingface \
    xl-vlms-demo
```
Then open **http://localhost:8501** in your browser.

#### Mount options

| Flag | Purpose |
|---|---|
| `--gpus all` | Expose all GPUs to the container |
| `-p 8501:8501` | Map Streamlit port |
| `-v /path/to/huggingface/hub:...` | Share cached HF models (avoids re-downloading ~7 GB) |
| `-e HF_TOKEN=hf_xxx` | Pass your Hugging Face token if the model is gated |
| `-e DEVICE_ID=0` | Select a specific GPU |

#### Run the classification demo instead

```bash
docker run --rm --gpus all \
  -p 8501:8501 \
  -v /path/to/huggingface/hub:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  xl-vlm-demo \
  streamlit run demo/vlm_classify_demo.py --server.port 8501 --server.address 0.0.0.0
```

### 3. Custom concept bank

If your concept bank `.pth` files live outside the repo, mount them in:

```bash
docker run --rm --gpus all \
  -p 8501:8501 \
  -v /path/to/concept/snmf:/app/outputs/imagenet_3_class_v2/concept/snmf \
  -v /path/to/huggingface/hub:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  xl-vlm-demo
```

---

## Quick Start — Conda (without Docker)

### 1. Create environment

```bash
conda create -n xlvlm-v1 python=3.10 -y
conda activate xlvlm-v1
```

### 2. Install dependencies

```bash
# PyTorch with CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Core packages
pip install tqdm "git+https://github.com/bckim92/language-evaluation.git" \
    psutil spacy timm accelerate transformers python-dotenv

# spaCy model
python -m spacy download en_core_web_sm

# Evaluation (BERTScore & CLIPScore)
pip install bert-score
pip install "git+https://github.com/openai/CLIP.git"

# Qwen VL support
pip install qwen-vl-utils

# Additional conda packages
conda install -c conda-forge inflect scikit-learn openjdk -y

# COCO evaluation data
python -c "import language_evaluation; language_evaluation.download('coco')"

# Lang-Segment-Anything
pip install -U "git+https://github.com/luca-medeiros/lang-segment-anything.git"

# Streamlit + demo dependencies
pip install streamlit streamlit-image-coordinates matplotlib
```

### 3. Configure `.env`

Edit `.env` in the project root (see `.env.example`):

```dotenv
export VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
export LAYER_PATH="model.language_model.norm"
export OUTPUT_DIR="/path/to/outputs/imagenet_3_class_v2"
export HF_HOME="/path/to/huggingface/hub"
export DEVICE_ID=0
export IMAGE_SIZE_WIDTH=512
```

### 4. Run the demo

```bash
# Interactive VLM Concept Explorer (upload → ask → click nouns → concepts)
conda run -n xlvlm-v1 streamlit run demo/vlm_binary_demo.py

# Compact Classification Demo (upload → auto-detect → click bboxes → concepts)
conda run -n xlvlm-v1 streamlit run demo/vlm_classify_demo.py
```

Or use the launch script:

```bash
bash demo/run_demo.sh
```

---

## Available Demos

| File | Description |
|---|---|
| `demo/vlm_binary_demo.py` | **Concept Explorer** — free-text question, noun pill selection, concept cards + prototypes |
| `demo/vlm_classify_demo.py` | **Classification Demo** — auto-classify, bbox grounding, click-to-explain, compact layout |

---

## Classify API Resize Sync Contract

For the FastAPI classify flow (`demo/classify_api.py`), image resizing and bbox
coordinates are synchronized via explicit response fields.

- `POST /api/classify` returns:
  - `downsample_ratio`
  - `original_size` (`[width, height]`)
  - `processed_size` (`[width, height]`)
- `POST /api/ground` returns:
  - `bbox_space` (`processed` or `original`)
  - `bbox_image_size` (the image size that bbox coordinates are defined in)
  - `original_size`, `processed_size`, `downsample_ratio`
  - `sync_contract_version`

Recommended setting for robust visualization:

```dotenv
export GROUND_BBOX_COORD_SPACE=processed
```

Frontend rule:

- Always scale/draw bounding boxes using `bbox_image_size` from `/api/ground`,
  not guessed dimensions.
- Prefer displaying the backend image from `GET /api/image/{image_id}` so the
  rendered image matches inference/grounding space.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Reduce `IMAGE_SIZE_WIDTH` or use a GPU with more VRAM |
| Model downloads slowly | Pre-download with `huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct` and mount the cache |
| `spaCy model not found` | Run `python -m spacy download en_core_web_sm` |
| Port already in use | Change port: `--server.port 8502` |
| Docker: `could not select device driver` | Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
