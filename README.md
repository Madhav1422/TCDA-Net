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
## 4. Dataset Structure

Organise the data exactly as follows:

```
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
```

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

## Output Structure

```
TCDA_2026_imp/
    BASE_*/
        metrics/
        curves/
        gradcam/
    TCDA_*/
        metrics/
        curves/
        gradcam/
    all_results.csv
    summary_aggregated.csv
    wilcoxon_n.csv
    mcnemar_pooled.csv
    superiority_table.csv
```


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

* GPU: NVIDIA GeForce RTX [4060] ([8] GB)
* CPU: Intel Core [Intel Core Ultra 9 185H ]
* RAM: [16] GB

Addtional Support (HPC):

* NVIDIA H200 GPUs (HPC cluster)



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
---




2. Ablation Study (Ablation.py)

The ablation experiments reported in the manuscript follow a progressive module integration strategy designed to evaluate the cumulative contribution of the proposed TCDA components.

Ablation Configurations
Configuration	Description
ABL0	Baseline backbone with Global Average Pooling (GAP) and classification head only
ABL1	ABL0 + Multi-scale Riesz-inspired geometric feature extraction
ABL2	ABL1 + Channel-wise Differential Attention
ABL3	ABL2 + Topology-inspired Gating
ABL4	Full TCDA architecture including residual bypass pathway
Implementation Details

The ablation hierarchy follows the implementation described in the manuscript:

ABL0 → ABL1 → ABL2 → ABL3 → ABL4

where each subsequent configuration adds one additional TCDA component.

Reused Configurations

To ensure consistency with the main experiments:

ABL0 corresponds exactly to the baseline model (BASE_*).
ABL4 corresponds exactly to the full TCDA model (TCDA_*).

Therefore, ABL0 and ABL4 are not retrained separately.

Independently Trained Configurations

The following intermediate ablation models are trained independently:

ABL1
ABL2
ABL3

Results from ABL0 and ABL4 are directly reused from the baseline and full TCDA experiments, respectively.

Running Ablation Experiments

The ablation study is executed through:

python SMVIB_main_v3.py

The script automatically:

Trains ABL1, ABL2, and ABL3.
Reuses the results of the baseline model as ABL0.
Reuses the results of the full TCDA model as ABL4.
Computes performance metrics, statistical tests, and ablation plots.
Notes

The ablation study presented in this repository follows a progressive integration design rather than a leave-one-out component removal strategy. The objective is to assess the cumulative contribution of the TCDA modules and the effect of progressively increasing geometric and attention-based modelling capacity.
