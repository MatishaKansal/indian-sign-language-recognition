# Static 15-Class ISL Sign Recognition

This project classifies static Indian Sign Language images into 15 configured classes. It uses MediaPipe hand landmarks and a MobileNetV3-Small image model. The train, validation, and test splits are signer-independent.

## Current models and result

- **Compact baseline:** one-hidden-layer MLP (64 units, 9,359 parameters) using hand landmarks.
- **Improved:** ensemble of landmark MLPs using hand-shape and wrist position/scale features, combined with the MobileNetV3-Small image branch.

On the current 2,250-image held-out test split, the compact baseline accuracy is **63.02%** (macro F1 **60.30%**) and the improved-model accuracy is **73.33%** (macro F1 **70.30%**). The compact baseline uses one 64-unit hidden layer and has fewer parameters than the previous two-hidden-layer MLP. Its architecture is selected as a simpler comparator, not tuned to a target accuracy. These results are exploratory: signer IDs are inferred from folder names, and near-duplicate screening may miss matches. Keep the test split for reporting, not model selection.

## Setup

Use Python 3.11 and a project-local virtual environment. In PowerShell, from the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a compatible PyTorch and torchvision pair using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the project dependencies:

```powershell
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip install ipykernel
```

Select `.venv` as the notebook kernel in VS Code. Set `data.dataset_root` in `configs/config.yaml` to a directory containing one subdirectory per sign class. The configured 15-class allowlist is in the same file.

## Run the pipeline

Run notebooks from the project directory in this order:

1. `01_dataset_exploration.ipynb` — optional inspection of the local image dataset.
2. `02_data_preprocessing.ipynb` — create the signer-independent data splits.
3. `03_landmark_extraction.ipynb` — extract and cache MediaPipe features.
4. `04_landmark_baseline.ipynb` — train and save the compact baseline.
5. `05_image_head_training.ipynb` — train the improved model's image branch.
6. `07_signer_cross_validation.ipynb` — select landmark features using training signers.
7. `06_improved_ensemble.ipynb` — train the improved ensemble using the selected features.
8. `08_model_comparison.ipynb` — report the final compact-baseline/improved comparison.
9. `09_webcam_testing.ipynb` — run local webcam inference.

Run notebook 08 only after both model checkpoints have been created. Do not use its test results to select features or training settings.

Run the unit tests from PowerShell with:

```powershell
python -m unittest discover -s tests -q
```

## Project paths

- `configs/config.yaml` — dataset path, class list, model, and output configuration.
- `src/` — data, feature extraction, training, evaluation, and inference code.
- `notebooks/` — ordered data and model workflow.
- `data/` — local dataset, split manifest, and landmark cache.
- `outputs/checkpoints/` — trained model checkpoints.
- `outputs/predictions/` — evaluation metrics, predictions, and reports.

Keep the original dataset and generated checkpoints backed up separately.
