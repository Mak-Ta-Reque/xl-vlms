#!/bin/bash
set -e

ENV_NAME="lsam_env"
PYTHON_VER="3.12"

echo "Creating fresh conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VER -y
conda activate $ENV_NAME

echo "Setting conda-forge channel priority"
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

echo "Installing modern C++ runtime"
conda install -c conda-forge "libstdcxx-ng>=12" "gcc>=12" -y

echo "Setting LD_LIBRARY_PATH for permanent conda lib usage"
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh <<'EOF'
unset LD_LIBRARY_PATH
EOF

echo "Installing PyTorch (CPU version, change for GPU if needed)"
conda install -c pytorch -c conda-forge pytorch torchvision torchaudio cpuonly -y

echo "Installing other dependencies (conda where possible)"
conda install -c conda-forge tqdm psutil spacy timm accelerate scikit-learn openjdk inflect -y

echo "Installing pip-only / git packages"
pip install git+https://github.com/bckim92/language-evaluation.git bert-score qwen-vl-utils
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git

echo "Download SpaCy English model"
python -m spacy download en_core_web_sm

echo "Download COCO evaluation data"
python -c "import language_evaluation; language_evaluation.download('coco')"

echo "Installation complete!"
echo "Activate the environment with: conda activate $ENV_NAME"
