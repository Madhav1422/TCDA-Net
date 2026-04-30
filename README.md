# TCDA-Net: Geometry-Conditioned Differential Attention Network

## 1. Overview

This repository implements TCDA-Net  for brain tumour classification using MRI images.

Classes:

* glioma
* meningioma
* pituitary
* notumor

The framework evaluates both baseline CNN backbones and their TCDA-enhanced variants across multiple random seeds, with full statistical analysis.

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

Two different datasets are used.

Training dataset:

* Source: Kaggle Brain Tumor MRI dataset
* Link: https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images
Testing dataset:

* Source: Mendeley Brain Tumor dataset
* Link: https://data.mendeley.com/datasets/zwr4ntf94j/1

This setup is intentional and evaluates cross-dataset generalisation.

---

## 4. Dataset Structure

Organise the data exactly as follows:

Training/
glioma/
meningioma/
pituitary/
notumor/

Test/
glioma/
meningioma/
pituitary/
notumor/

Each class folder must contain the corresponding MRI images.

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

Execute:

python TCDA.py

This will automatically:

1. Train all models:

   * BASE_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}
   * TCDA_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}

2. Use fixed seeds:

   * 42, 43, 44, 45, 46

3. Perform:

   * Training and validation
   * Test inference
   * GradCAM generation
   * Metric computation
   * Statistical testing (Wilcoxon, McNemar)

No additional scripts are required.

---

## 7. Outputs

All outputs are saved in:

TCDA_2026_imp/

Directory structure:

TCDA_2026_imp/
├── BASE_/
├── TCDA_/
│     ├── metrics/
│     ├── curves/
│     ├── gradcam/
├── all_results.csv
├── summary_aggregated.csv
├── wilcoxon_n.csv
├── mcnemar_pooled.csv
├── superiority_table.csv

Generated results include:

* Accuracy, F1-score, Precision, Recall, AUC
* Confusion matrices (CSV and PNG)
* ROC curves (CSV and PNG)
* Training curves
* GradCAM visualisations
* Statistical comparisons

---

## 8. Reproducibility

The experiments are fully reproducible:

* Fixed seeds: 42, 43, 44, 45, 46
* Deterministic PyTorch configuration enabled
* Identical preprocessing and augmentation pipeline
* All model configurations defined within a single script

Running the script once reproduces all reported results.

---

## 9. Hardware Requirements

* GPU recommended (NVIDIA CUDA-supported)
* Minimum: 8 GB VRAM (for batch size = 16)

Approximate runtime:

* Several hours depending on GPU capability
* Full run includes all models and seeds

---

## 10. Notes

* Only dataset paths need modification before execution
* No manual intervention is required after starting the script
* Ensure correct folder naming and class labels
* Outputs are automatically saved and organised

---

## 11. Citation

If you use this code, cite:

"TCDA-Net: Geometry-Conditioned Differential Attention Using Riesz-Based Multi-Scale Features for Brain Tumour Classification"

---

## 12. Contact

For issues related to reproduction, ensure:

* Dataset structure matches Section 4
* Paths are correctly set
* Dependencies are installed

No additional configuration is required beyond what is described above.
