# XL-VLMS: Multimodal Concept Feature Extraction

## Overview
This project provides tools for extracting, analyzing, and decomposing concept features from multimodal data (images and text) using large vision-language models. It includes scripts for feature generation, decomposition, and analysis, as well as a demo notebook for binary task explanation.

## Demo Output


Given a grid of 2x2 images (total 4 different images), the model is tasked to predict the object in each grid cell. For each predicted token, the corresponding residual stream is mapped to relevant concepts. The demo outputs are provided as image files in the `grounding_per_token` directory. On the left side of each .png, the image grid is displayed (small, but can be zoomed in); on the right, each row presents the top 5 concepts and their similarity scores for the corresponding predicted token. Please use zoom to examine the details in the images carefully.

Example demo output files:
- [grounding_per_token/per_token_viz_1.png](grounding_per_token/per_token_viz_1.png)
- [grounding_per_token/per_token_viz_3.png](grounding_per_token/per_token_viz_3.png)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd xl-vlms
   ```

2. **Set up Python environment**
   - Recommended: Python 3.10 or higher.
   - You can use Conda or a virtualenv:
     ```bash
     conda create -n xlvlms python=3.10
     conda activate xlvlms
     # or
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**
   - Using pip:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
     pip install tqdm git+https://github.com/bckim92/language-evaluation.git bert-score clip psutil spacy timm accelerate
     python -m spacy download en_core_web_sm
     # For Qwen model support:
     pip install qwen-vl-utils
     conda install -c conda-forge inflect
     conda install -c conda-forge scikit-learn
     # For Java support (if needed):
     conda install -c conda-forge openjdk
     # Download COCO evaluation data:
     python -c "import language_evaluation; language_evaluation.download('coco')"

     pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
     ```




### Full pipeline 
# Our method 
scripts/run_full_pipeline.sh
Set up the values of the data path and model names


Al result will be found in eval folder of outputs


## Expert usages

Data processing 
run run_dataset_inference.sh
for a given directory it will find all the image concepts and make map of concept -> image files



# Crop the images metioned in the concept - mage json files

run run_crop_images.sh

# create crops given the json map of concept-> image
# use script run_feature_gen_cgdl.sh to generate concept features
# use script run_feature_decompose_cgdl.sh to decompose the features
for all the decomposition options
run vlm explainer script to get the explanations
for all the vlm_explainer files  evaluate 
Now with output evaluate the concept Q-del and Q-Insertion






## 🧠 Demo & Usage


1. **Demo notebook**
   - Open and run [`explaining_binary_task_rsml.ipynb`](explaining_binary_task_rsml.ipynb) for a guided demo.

2. **Feature Generation & Decomposition**
   - To generate concept features:
     ```bash
     bash scripts/run_feature_gen_cgdl.sh
     ```
   - To decompose features:
     ```bash
     bash scripts/run_feature_decompose_cgdl.sh
     ```
   - You can set `MAX_ITERATIONS=10` or higher for more concepts/classes (longer runtime).

## 📂 Folder Structure

- `src/` : Main source code (models, datasets, metrics, helpers, analysis)
- `scripts/` : Feature generation and decomposition scripts
- `preprocessing/` : Preprocessing utilities
- `explaining_binary_task_rsml.ipynb` : Demo notebook
- `install.sh` : Installation script

## 📄 License
See [LICENSE](LICENSE) for details.

---

For questions or issues, please refer to the repository or contact the maintainers.