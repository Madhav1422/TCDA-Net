# TCDA-Net: Geometry-Conditioned Differential Attention Network

## 1. Overview

This repository implements TCDA-Net for brain tumour classification using MRI images.

Classes:

* glioma
* meningioma
* pituitary
* notumor

The framework evaluates baseline CNN backbones and their TCDA-enhanced variants across multiple random seeds with statistical analysis.

---

## 2. Environment

Tested configuration:

* Python 3.10
* PyTorch 2.x
* CUDA-enabled GPU (recommended)

Install dependencies:
pip install torch torchvision timm numpy pandas matplotlib seaborn scikit-learn scipy

---

## 3. Dataset

Two datasets are used.

Training dataset:

* Source: Kaggle Brain Tumor MRI dataset
* Link: https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images

Testing dataset:

* Source: Mendeley Brain Tumor dataset
* Link: https://data.mendeley.com/datasets/zwr4ntf94j/1

This setup evaluates cross-dataset generalisation.

---

## 4. Dataset Structure

## 4. Dataset Structure

Organise the data exactly as follows:

`
Training/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/

Test/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
`


---

## 5. Configuration

Open `TCDA.py` and update:

```python
TRAIN_PATH = "path_to/Training/"
TEST_PATH  = "path_to/Test/"
```

Optional:

```python
gpu_id = 0
```

---

## 6. Running the Experiments

Run:
python TCDA.py

This will:

1. Train models:

   * BASE_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}
   * TCDA_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}

2. Use seeds:

   * 42, 43, 44, 45, 46

3. Perform:

   * Training and validation
   * Test inference
   * GradCAM generation
   * Metric computation
   * Statistical testing (Wilcoxon, McNemar)

No additional scripts are required.

---

## 7. Quick Verification (Optional)

To reduce runtime, modify:

```python
SEEDS = [42]
MODELS = ["TCDA_ResNet50"]
```

This runs a minimal configuration for validation.

---

## 8. Outputs

All outputs are saved in:

TCDA_2026_imp/

Structure:

TCDA_2026_imp/
BASE_*/
TCDA_*/
metrics/
curves/
gradcam/
all_results.csv
summary_aggregated.csv
wilcoxon_n.csv
mcnemar_pooled.csv
superiority_table.csv

Generated results:

* Accuracy, F1-score, Precision, Recall, AUC
* Confusion matrices (CSV, PNG)
* ROC curves (CSV, PNG)
* Training curves
* GradCAM visualisations
* Statistical comparisons

---

## 9. Reproducibility

* Fixed seeds: 42, 43, 44, 45, 46
* Deterministic PyTorch configuration enabled
* Identical preprocessing and augmentation pipeline
* All experiments executed from a single script

Running the script reproduces all reported results.

---

## 10. Hardware

Experiments were conducted on a Dell Alienware m16 laptop with:

* GPU: NVIDIA GeForce RTX [model] ([VRAM] GB)
* CPU: Intel Core [exact model name]
* RAM: [amount] GB

Training was performed using CUDA-enabled PyTorch.

---

## 11. Runtime

The full experimental run (all models across all seeds) requires approximately 4 days on the specified hardware.

Runtime may vary depending on GPU capability.

A single model run (one seed) typically takes several hours.

---

## 12. Notes

* Only dataset paths need modification
* No manual intervention is required after execution starts
* Ensure correct folder naming and class labels
* All outputs are automatically saved

---

## 13. Citation

If you use this code, cite:

"TCDA-Net: Geometry-Conditioned Differential Attention Using Riesz-Based Multi-Scale Features for Brain Tumour Classification"

---

## 14. Contact

Before reporting issues, ensure:

* Dataset structure matches Section 4
* Paths are correctly configured
* Dependencies are installed

No additional configuration is required.
