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
   - Recommended: Python 3.10
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

## ⚙️ Configuration

### Environment Setup with `.env`

The pipeline uses a `.env` file for easy configuration management. This keeps your custom paths separate from the code.

1. **Copy the example configuration**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your paths**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Key configuration variables**
   
   The `.env` file supports the following variables:
   
   ```bash
   # Data & Output Paths
   export INPUT_DIR="/path/to/your/dataset"
   export OUTPUT_DIR="$ROOT_DIR/outputs/your_run_name"
   export HF_HOME="/path/to/huggingface/models"
   
   # Crops Configuration
   export MIN_IMAGES_PER_TAG=20  # Use 10 for dummy data at xl-vlms/data
   
   # Optional: Model Configuration
   export VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
   export BATCH_SIZE=48
   export DEVICE_ID=0
   export DECOMP_METHODS="snmf"
   ```

4. **Using `$ROOT_DIR` variable**
   
   The `.env` file has access to `$ROOT_DIR` (the xl-vlms project directory):
   ```bash
   # Use $ROOT_DIR for paths inside the project
   export OUTPUT_DIR="$ROOT_DIR/outputs/my_experiment"
   export IMAGE_ROOT="$ROOT_DIR/data/grids"
   
   # Use absolute paths for external resources
   export INPUT_DIR="/mnt/sda/datasets/food-101"
   export HF_HOME="/mnt/sdz/models"
   ```

5. **Quick configuration switching**
   
   For dummy/test data:
   ```bash
   export INPUT_DIR="$ROOT_DIR/data"
   export MIN_IMAGES_PER_TAG=10
   export OUTPUT_DIR="$ROOT_DIR/outputs/test_run"
   ```
   
   For full datasets:
   ```bash
   export INPUT_DIR="/path/to/full/dataset"
   export MIN_IMAGES_PER_TAG=20
   export OUTPUT_DIR="$ROOT_DIR/outputs/production_run"
   ```

**Note:** The `.env` file is ignored by git (in `.gitignore`) to keep your local paths private. Use `.env.example` as a template for sharing configuration structure.

### Full Pipeline (Recommended)

The easiest way to run the complete pipeline:

1. **Configure your environment** (see Configuration section above)
   ```bash
   cp .env.example .env
   nano .env  # Edit with your paths
   ```

2. **Run the full pipeline**
   ```bash
   ./scripts/run_full_pipeline_without_coroping.sh
   ```
   
   The script automatically loads your `.env` configuration and runs:
   - Dataset inference → concept mapping
   - Crop generation with concept grounding
   - Feature extraction
   - Feature decomposition (SNMF/NMF/PCA)
   - VLM explanation
   - Concept deletion/insertion evaluation
   - Visualization plots

3. **Results location**
   ```
   $OUTPUT_DIR/
   ├── inference/           # Concepts and crops
   ├── features/            # Extracted features
   ├── concept/             # Decomposed concepts per method
   ├── explanations/        # VLM explanations per method
   ├── eval/                # Evaluation results (CSV files)
   └── plots/               # Visualizations
   ```

4. **Command-line overrides** (optional)
   ```bash
   # Override specific variables without editing .env
   MIN_IMAGES_PER_TAG=5 OUTPUT_DIR="./outputs/quick_test" \
     ./scripts/run_full_pipeline_without_coroping.sh
   
   # Or use flags
   ./scripts/run_full_pipeline_without_coroping.sh \
     --input-dir /path/to/data \
     --output-dir ./outputs/experiment_1 \
     --decomp snmf,nmf,pca
   ```

All results will be found in the `eval/` folder of your configured output directory.


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