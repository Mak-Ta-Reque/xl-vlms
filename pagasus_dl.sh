# from /mnt/abka03/Projects/xl-vlms
HF_HOME=/netscratch/kadir/huggingface/hub \
VLM_MODEL=google/gemma-3n-E4B-it \
LAYER_PATH="model.language_model.norm" \
IMAGE_ROOT="/netscratch/kadir/xl-vlms/data/val" \
TOP_N=5 \
NUM_POINTS=80 \
BATCH_SIZE=10 \
DEVICE=cuda \
SPLIT=train \
BASE_DATA_DIR="/netscratch/kadir/xl-vlms/data/train" \
bash scripts/run_full_pipeline_dl.sh \
  --output-dir "/netscratch/kadir/xl-vlms/outputs/run_dl_$(date +%Y%m%d_%H%M%S)" \
  --decomp "snmf,pca"