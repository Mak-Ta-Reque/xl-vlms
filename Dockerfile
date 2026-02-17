# ═══════════════════════════════════════════════════════════════════
#  XL-VLMs Demo — GPU-accelerated Streamlit app
#  Base: NVIDIA CUDA 12.6 + cuDNN 9 on Ubuntu 22.04
# ═══════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git wget curl ca-certificates unzip \
        # fonts for PIL/ImageFont
        fonts-dejavu-core \
        # Java (for language-evaluation / COCO metrics)
        default-jre-headless \
        # OpenCV runtime deps (headless)
        libglib2.0-0 libgl1-mesa-glx libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.6) ─────────────────────────────────────────
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu126

# ── Core ML / NLP packages ──────────────────────────────────────
RUN pip install --no-cache-dir \
        "transformers>=4.55,<5" accelerate \
        tqdm psutil spacy timm \
        numpy Pillow scikit-learn scipy inflect \
        python-dotenv requests pandas nltk \
        opencv-python-headless \
        "git+https://github.com/bckim92/language-evaluation.git"

# spaCy model + NLTK data
RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# ── Evaluation packages ─────────────────────────────────────────
# CLIP's setup.py uses pkg_resources which was removed in setuptools>=82.
# Pin setuptools<82, then install CLIP without build isolation so it uses
# the container's setuptools rather than fetching the latest in isolation.
RUN pip install --no-cache-dir "setuptools<82" bert-score && \
    pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/openai/CLIP.git"

# ── Qwen VL support ─────────────────────────────────────────────
RUN pip install --no-cache-dir qwen-vl-utils

# ── Segment Anything (Lang-SAM) ─────────────────────────────────
RUN pip install --no-cache-dir \
        "git+https://github.com/luca-medeiros/lang-segment-anything.git"

# ── Streamlit + click-coordinates + vis packages ─────────────────
RUN pip install --no-cache-dir \
        streamlit streamlit-image-coordinates matplotlib \
        pycocotools

# ── COCO evaluation data ────────────────────────────────────────
RUN python -c "import language_evaluation; language_evaluation.download('coco')"

# ── Project files ────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# Create temp dir for uploaded images
RUN mkdir -p /app/api/temp_images

# ── Default environment variables ────────────────────────────────
# These can be overridden at runtime via docker run -e or --env-file
ENV DEVICE_ID=0 \
    VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct" \
    LAYER_PATH="model.language_model.norm" \
    IMAGE_SIZE_WIDTH=512 \
    OUTPUT_DIR="/app/outputs/imagenet_3_class_v2" \
    HF_HOME="/root/.cache/huggingface"

# Streamlit config: disable CORS/XSRF for container use, listen on all interfaces
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# ── Entrypoint ───────────────────────────────────────────────────
# NOTE: Concept grounding images are stored as absolute host paths in the
# .pth file.  Mount the original data directory at runtime so the images
# are reachable.  Also mount the HuggingFace cache to avoid re-downloading
# models.  Example:
#
#   sudo docker run --gpus all -p 8501:8501 --name xl-vlms-demo --rm \
#       -v /mnt/abka03/xlvlm_data:/mnt/abka03/xlvlm_data:ro \
#       -v /mnt/abka03/huggingface/hub:/root/.cache/huggingface \
#       xl-vlms-demo
CMD ["streamlit", "run", "demo/vlm_classify_demo.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
