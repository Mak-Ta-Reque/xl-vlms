# Project Setup and Concept Feature Extraction

## 📦 Installation




# Installation

1. Clone this repository and navigate to LLaVA folder
```bash
git clone ....
cd project_root
```

2. Install Package only coda forge
activate


conda create --prefix /mnt/abka03/.conda/envs/rsml3 python=3.9 --no-default-packages
press yes

3. Install other dependencies

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install tqdm


pip install git+https://github.com/bckim92/language-evaluation.git

conda install -c conda-forge openjdk


python -c "import language_evaluation; language_evaluation.download('coco')"

pip install bert-score

pip install clip


pip install psutil

pip install spacy

python -m spacy download en_core_web_sm

pip install timm

pip install qwen-vl-utils #if you use qwen model

pip install accelerate 


---

## 🧠 Demo

Please open [Jupyter notebook](playground/explaining_binary_task_rsml.ipynb)

For increasing number of [concept/ class] use [feature_gen sceript](scripts/run_feature_gen_cgdl.sh) and (scripts/run_feature_decompose_cgdl.sh) MAX_ITERATIONS=10 or more , takes more time 