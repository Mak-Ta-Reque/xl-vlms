# Project Setup and Concept Feature Extraction

## 📦 Installation

Please refer to the [installation guide](docs/installation.md) for detailed setup instructions.

---

## 🧠 Concept and Feature Extraction

### Step 1: Generate Model Activations

From the project root, run the following script to extract hooked outputs (model activations):

```bash
./run_feature_gen_cgdl.sh
```

> **Note:** Update the data path inside this script before running.

### Step 2: Decompose Features into Concepts

After generating activations, run:

```bash
./run_feature_decompose_cgdl.sh
```

This will extract features and corresponding concepts, saving them in the designated concept directory.

> **Tip:** Ensure you've set the correct output directories for both `features` and `concepts`.

---

## 📊 Visualize Features and Concepts

Open the notebook below to visualize extracted features, concepts, and generate local explanations:

```bash
playground/local_explantion_vlm_presentation.ipynb
```

---

## 🗂️ Data Directory Structure

Your data should be organized as follows:

```
data/
├── train/
│   ├── class_name/concept_name_1/
│   │   ├── example_1
│   │   ├── example_2
│   │   └── example_3
│   └── class_name/concept_name_2/
│       ├── example_1
│       ├── example_2
│       └── example_3
└── val/
    ├── class_name/concept_name_1/
    │   ├── example_1
    │   ├── example_2
    │   └── example_3
    └── class_name/concept_name_2/
        ├── example_1
        ├── example_2
        └── example_3
```

---

## ✅ Validation and Evaluation

1. Run feature generation on validation data:

```bash
./run_feature_gen_cgdl.sh
```

2. Then run the evaluation script:

```bash
./run_evalutaion.sh
```

> **Important:** Edit `run_evalutaion.sh` to set the correct paths for:
> - Feature directory
> - Concept directory

---

## 🛠️ Notes

- Always double-check and customize paths in the `.sh` scripts to match your local setup.
- Ensure your dataset follows the correct structure for successful processing.

---