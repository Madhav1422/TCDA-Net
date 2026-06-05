# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
SMVIB_main_v3.py
================
TCDA-Net v2 : Topology-Conditioned Differential Attention Network
=================================================================
Brain Tumour Classification — 4 classes
    glioma | meningioma | notumor | pituitary

Dataset  : /nfsshare/users/raghavan/Brainz/Brain tumor dataset/
Save dir : TCDA_2026_imp/

DESIGN DECISIONS
----------------
1.  TCDA_EfficientNetB4 EXCLUDED everywhere — not trained, not plotted, not tested.
2.  Main models  (4 backbones each):
        BASE_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}
        TCDA_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}
3.  Ablation per backbone — 5 progressive conditions:
        ABL0_{bb} = Baseline (GAP only, same weights reused from BASE_)
        ABL1_{bb} = + Riesz feature extraction only
        ABL2_{bb} = + Riesz + Channel-wise Differential Attention
        ABL3_{bb} = + Riesz + ChannelAttn + Topology Gate
        ABL4_{bb} = Full TCDA (same weights reused from TCDA_)
    -> ABL0 and ABL4 are NOT retrained; results are copied from main runs.
    -> ABL1, ABL2, ABL3 are trained fresh.
    -> By architecture, each additional block can only add capacity;
       full TCDA (ABL4) is the ceiling in the visualisations.
4.  Statistical tests: Wilcoxon signed-rank + pooled McNemar ONLY.
    - Main models: one test set (all MAIN_MODELS pairwise).
    - Ablation: per-backbone pairwise tests.
5.  NO GradCAM anywhere.
6.  Every metric saved individually: its own CSV + boxplot PNG.
7.  Confusion matrices and ROC curves saved per seed per model.
"""

import gc
import math
import random
import itertools
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.v2 as v2

import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Global figure style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.weight":        "bold",
    "axes.titleweight":   "bold",
    "axes.labelweight":   "bold",
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    10,
    "figure.titlesize":   15,
    "figure.titleweight": "bold",
    "xtick.major.width":  1.4,
    "ytick.major.width":  1.4,
    "axes.linewidth":     1.4,
    "lines.linewidth":    2.2,
})

# ==========================================================================
# Configuration
# ==========================================================================
TRAIN_PATH    = "/nfsshare/users/raghavan/brainzz/Brain tumor dataset/Training/"
TEST_PATH     = "/nfsshare/users/raghavan/brainzz/Brain tumor dataset/Test/"
SAVE_DIR      = Path("TCDA_2026_imp_abl")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

gpu_id        = 5
DEVICE        = torch.device(
    f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu"
)
IMG_SIZE      = 224
LR            = 3e-4
WARMUP_EPOCHS = 3
EPOCHS        = 25
PATIENCE      = 15
DESC_DIM      = 512
RIESZ_SCALES  = [1.0, 2.0, 4.0]
LABEL_SMOOTH  = 0.08
CUTMIX_ALPHA  = 1.0
MIXUP_ALPHA   = 0.4
EPS           = 1e-8
ALPHA_STAT    = 0.05
BATCH         = 16
STOCH_DEPTH_P = 0.10

SEEDS = [42, 43, 44, 45, 46]

# EfficientNetB4 intentionally excluded from ALL experiments
BACKBONE_REGISTRY = {
    "ResNet50":     "resnet50.a1_in1k",
    "DenseNet121":  "densenet121.ra_in1k",
    "MobileNetV3L": "mobilenetv3_large_100.ra_in1k",
    "ConvNeXtTiny": "convnext_tiny.in12k_ft_in1k",
}
BACKBONE_NAMES = list(BACKBONE_REGISTRY.keys())

MAIN_MODELS = (
    [f"BASE_{b}" for b in BACKBONE_NAMES] +
    [f"TCDA_{b}" for b in BACKBONE_NAMES]
)

ABL_LEVELS = [0, 1, 2, 3, 4]
ABL_LABELS = {
    0: "Baseline\n(GAP only)",
    1: "+Riesz\nExtractor",
    2: "+Channel\nAttn",
    3: "+Topology\nGate",
    4: "Full TCDA\n(+Residual)",
}
# All ablation model names (used for per-metric saves and stats)
ABLATION_MODELS = [
    f"ABL{lvl}_{b}" for b in BACKBONE_NAMES for lvl in ABL_LEVELS
]
# Only the ones we actually train (ABL0 and ABL4 are reused)
ABL_TRAIN_LEVELS = [1, 2, 3]
ABL_TRAIN_MODELS = [
    f"ABL{lvl}_{b}" for b in BACKBONE_NAMES for lvl in ABL_TRAIN_LEVELS
]

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]
METRIC_LABELS = {
    "acc":             "Accuracy",
    "f1_macro":        "Macro F1",
    "precision_macro": "Macro Precision",
    "recall_macro":    "Macro Recall",
    "auc_macro":       "Macro AUC",
}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ==========================================================================
# Directory helpers
# ==========================================================================

def get_dirs(model_name: str) -> dict:
    root = SAVE_DIR / model_name
    for sub in ["metrics", "curves"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return {"root": root, "metrics": root / "metrics", "curves": root / "curves"}


def get_paths(model_name: str, seed: int, dirs: dict) -> dict:
    m = dirs["metrics"]
    c = dirs["curves"]
    return {
        "test_metrics": m / f"test_metrics_seed{seed}.csv",
        "cm_csv":       m / f"cm_seed{seed}.csv",
        "cm_png":       m / f"cm_seed{seed}.png",
        "roc_data":     m / f"roc_data_seed{seed}.csv",
        "roc_png":      m / f"roc_seed{seed}.png",
        "history_csv":  m / f"history_seed{seed}.csv",
        "curves_png":   c / f"curves_seed{seed}.png",
    }


# ==========================================================================
# Reproducibility
# ==========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ==========================================================================
# Augmentation helpers
# ==========================================================================

def rand_bbox(size, lam):
    W, H   = size[2], size[3]
    cut_r  = math.sqrt(1.0 - lam)
    cut_w  = int(W * cut_r)
    cut_h  = int(H * cut_r)
    cx     = random.randint(0, W)
    cy     = random.randint(0, H)
    x1     = max(cx - cut_w // 2, 0)
    y1     = max(cy - cut_h // 2, 0)
    x2     = min(cx + cut_w // 2, W)
    y2     = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    lam   = float(np.random.beta(alpha, alpha))
    idx   = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam   = 1.0 - (x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3) + EPS)
    return mixed, y, y[idx], lam


def mixup_data(x, y, alpha=0.4):
    lam   = float(np.random.beta(alpha, alpha))
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def aug_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)


# ==========================================================================
# LR schedule — cosine + warmup
# ==========================================================================

def get_lr(epoch, total, warmup, base, min_lr=1e-6):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    prog = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))


# ==========================================================================
# Focal Loss
# ==========================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smooth=LABEL_SMOOTH):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(
            label_smoothing=label_smooth, reduction="none"
        )

    def forward(self, logits, target):
        ce  = self.ce(logits, target)
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ==========================================================================
# NOVEL MODULE — Riesz Extractor (multi-scale, L2-normalised)
# ==========================================================================

class RieszExtractor(nn.Module):
    """
    Analytically exact Riesz transform pyramid.
    L2-normalises input per channel before convolution.
    Returns list of (phi, A, theta) — one tuple per scale.
    """

    def __init__(self, in_channels: int, sigmas=None):
        super().__init__()
        sigmas      = sigmas or RIESZ_SCALES
        self.sigmas = list(sigmas)
        k           = 3
        for i, sigma in enumerate(self.sigmas):
            for d in ("x", "y"):
                kern = self._gauss_deriv(sigma, d, k)
                self.register_buffer(
                    f"kern_{i}_{d}", kern.view(1, 1, 2 * k + 1, 2 * k + 1)
                )

    @staticmethod
    def _gauss_deriv(sigma, direction, k):
        ax      = np.arange(-k, k + 1, dtype=np.float32)
        xx, yy  = np.meshgrid(ax, ax)
        g       = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kern    = (-xx / sigma ** 2 * g) if direction == "x" \
                  else (-yy / sigma ** 2 * g)
        kern   /= np.abs(kern).sum() + 1e-12
        return torch.from_numpy(kern)

    def _depthwise(self, x, kern):
        C = x.size(1)
        k = kern.expand(C, 1, kern.shape[2], kern.shape[3])
        return F.conv2d(x, k, padding=kern.shape[2] // 2, groups=C)

    def forward(self, x):
        norm    = x.norm(p=2, dim=(2, 3), keepdim=True).clamp(min=EPS)
        x_n     = x / norm
        results = []
        for i in range(len(self.sigmas)):
            kx   = getattr(self, f"kern_{i}_x")
            ky   = getattr(self, f"kern_{i}_y")
            R1   = self._depthwise(x_n, kx)
            R2   = self._depthwise(x_n, ky)
            phi  = torch.atan2(torch.sqrt(R1 ** 2 + R2 ** 2 + EPS),
                               x_n.abs() + EPS)
            A    = torch.sqrt(x_n ** 2 + R1 ** 2 + R2 ** 2 + EPS)
            theta = torch.atan2(R2 + EPS, R1 + EPS)
            results.append((phi, A, theta))
        return results


# ==========================================================================
# NOVEL MODULE — TCDA v2 (configurable blocks for ablation)
# ==========================================================================

class TCDA(nn.Module):
    """
    Topology-Conditioned Differential Attention v2.

    Flags control which blocks are active:
        use_riesz    — Riesz multi-scale feature extraction
        use_ch_attn  — Channel-wise differential attention
        use_topo     — Topology gate (beta0 proxy)
        use_residual — Residual bypass (gradient highway)

    Ablation mapping:
        ABL0 -> all False (but BaselineModel is used directly, not this class)
        ABL1 -> use_riesz=True only
        ABL2 -> use_riesz + use_ch_attn
        ABL3 -> use_riesz + use_ch_attn + use_topo
        ABL4 -> all True  (full TCDA)
    """

    def __init__(
        self,
        nf: int,
        use_riesz:    bool = True,
        use_ch_attn:  bool = True,
        use_topo:     bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.nf           = nf
        self.use_riesz    = use_riesz
        self.use_ch_attn  = use_ch_attn
        self.use_topo     = use_topo
        self.use_residual = use_residual
        self.n_scale      = len(RIESZ_SCALES)

        if use_riesz:
            self.riesz   = RieszExtractor(nf)
            self.scale_w = nn.Parameter(torch.ones(self.n_scale))

        self.proj_phi = nn.Linear(nf, DESC_DIM, bias=False)
        self.proj_A   = nn.Linear(nf, DESC_DIM, bias=False)
        self.proj_V   = nn.Linear(nf, DESC_DIM, bias=False)

        if use_ch_attn:
            self.proj_Q1 = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
            self.proj_K1 = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
            self.proj_Q2 = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
            self.proj_K2 = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
            self.lam     = nn.Parameter(torch.tensor(0.5))
            self.register_buffer("sigma_theta", torch.tensor(0.5))

        if use_topo:
            self.gamma = nn.Parameter(torch.tensor(0.3))

        self.norm = nn.LayerNorm(DESC_DIM)
        self.out  = nn.Linear(DESC_DIM, DESC_DIM)

        if use_residual:
            self.bypass = nn.Linear(nf, DESC_DIM, bias=False)
            nn.init.zeros_(self.bypass.weight)

    @staticmethod
    def _beta0_proxy(phi):
        """Spatial topology proxy. Returns (B, 1)."""
        s     = torch.sin(phi).mean(dim=1)           # (B, H, W)
        sign  = torch.sign(s + EPS)
        zc_w  = (sign[:, :, 1:] * sign[:, :, :-1] < 0).float().sum(dim=(1, 2))
        zc_h  = (sign[:, 1:, :] * sign[:, :-1, :] < 0).float().sum(dim=(1, 2))
        total = (zc_w + zc_h) / (s.size(1) * s.size(2) + EPS)
        return torch.sigmoid(total).unsqueeze(1)      # (B, 1)

    def forward(self, feat):
        feat_pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)   # (B, nf)

        if self.use_riesz:
            scales    = self.riesz(feat)
            sw        = F.softmax(self.scale_w, dim=0)
            phi_fused = sum(
                sw[i] * F.adaptive_avg_pool2d(scales[i][0], 1).flatten(1)
                for i in range(self.n_scale)
            )
            A_fused   = sum(
                sw[i] * F.adaptive_avg_pool2d(scales[i][1], 1).flatten(1)
                for i in range(self.n_scale)
            )
            theta_mean = F.adaptive_avg_pool2d(
                scales[0][2], 1
            ).flatten(1).mean(1)                       # (B,)
            topo_phi   = scales[0][0]                  # (B, C, H, W)
        else:
            phi_fused  = feat_pool
            A_fused    = feat_pool
            theta_mean = torch.zeros(feat.size(0), device=feat.device)
            topo_phi   = feat                          # fallback

        Q_base = self.proj_phi(phi_fused)              # (B, DESC_DIM)
        K_base = self.proj_A(A_fused)                  # (B, DESC_DIM)
        V      = self.proj_V(feat_pool)                # (B, DESC_DIM)

        if self.use_ch_attn:
            Q1           = self.proj_Q1(Q_base)
            K1           = self.proj_K1(K_base)
            Q2           = self.proj_Q2(Q_base)
            K2           = self.proj_K2(K_base)
            scale_factor = math.sqrt(DESC_DIM)
            a1           = (Q1 * K1) / scale_factor
            a2           = (Q2 * K2) / scale_factor
            g_orient     = torch.exp(
                -theta_mean.abs() ** 2 / (2.0 * self.sigma_theta ** 2 + EPS)
            ).unsqueeze(1)
            T            = (1.0 / (g_orient + EPS)).clamp(0.5, 4.0)
            lam          = self.lam.clamp(0.0, 1.0)
            A_diff       = torch.sigmoid(a1 / T) - lam * torch.sigmoid(a2 / T)
        else:
            A_diff = V                                 # identity path

        if self.use_topo:
            beta0     = self._beta0_proxy(topo_phi)
            gamma     = self.gamma.clamp(0.0, 1.0)
            topo_gate = 1.0 - beta0 * gamma           # (B, 1)
        else:
            topo_gate = torch.ones(feat.size(0), 1, device=feat.device)

        desc = 0.5 * (A_diff * topo_gate * V)

        if self.use_residual:
            out = self.norm(self.out(desc) + self.bypass(feat_pool))
        else:
            out = self.norm(self.out(desc))

        return out                                     # (B, DESC_DIM)


# ==========================================================================
# Backbone factory
# ==========================================================================

def _build_backbone(timm_name: str):
    bb = timm.create_model(timm_name, pretrained=True, features_only=True)
    nf = bb.feature_info[-1]["num_chs"]
    return bb, nf


# ==========================================================================
# Model variants
# ==========================================================================

class BaselineModel(nn.Module):
    """Backbone + GAP + Dropout(0.4) + Linear."""

    def __init__(self, timm_name: str, nc: int):
        super().__init__()
        self.bb, nf = _build_backbone(timm_name)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.drop    = nn.Dropout(0.4)
        self.head    = nn.Linear(nf, nc)

    def forward(self, x):
        feat   = self.bb(x)[-1]
        pooled = self.gap(feat).flatten(1)
        return (self.head(self.drop(pooled)),)


class TCDAModel(nn.Module):
    """
    Backbone + TCDA (configurable) + Dropout(0.3) + Linear.
    Stochastic depth: training only, probability STOCH_DEPTH_P ? plain GAP.
    """

    def __init__(self, timm_name: str, nc: int,
                 use_riesz=True, use_ch_attn=True,
                 use_topo=True, use_residual=True):
        super().__init__()
        self.bb, nf = _build_backbone(timm_name)
        self.tcda   = TCDA(
            nf,
            use_riesz=use_riesz,
            use_ch_attn=use_ch_attn,
            use_topo=use_topo,
            use_residual=use_residual,
        )
        self.gap_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf, DESC_DIM, bias=False),
            nn.LayerNorm(DESC_DIM),
        )
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(DESC_DIM, nc)

    def forward(self, x):
        feat = self.bb(x)[-1]
        if self.training and random.random() < STOCH_DEPTH_P:
            desc = self.gap_proj(feat)
        else:
            desc = self.tcda(feat)
        return (self.head(self.drop(desc)),)


# Ablation level ? TCDA kwargs
_ABL_KWARGS = {
    1: dict(use_riesz=True,  use_ch_attn=False, use_topo=False, use_residual=False),
    2: dict(use_riesz=True,  use_ch_attn=True,  use_topo=False, use_residual=False),
    3: dict(use_riesz=True,  use_ch_attn=True,  use_topo=True,  use_residual=False),
    4: dict(use_riesz=True,  use_ch_attn=True,  use_topo=True,  use_residual=True),
}


def create_model(name: str, nc: int) -> nn.Module:
    """
    Supported name patterns:
        BASE_{backbone}
        TCDA_{backbone}
        ABL{0-4}_{backbone}
    """
    if name.startswith("BASE_"):
        bname = name[5:]
        if bname not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {bname}")
        return BaselineModel(BACKBONE_REGISTRY[bname], nc)

    if name.startswith("TCDA_"):
        bname = name[5:]
        if bname not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {bname}")
        return TCDAModel(BACKBONE_REGISTRY[bname], nc, **_ABL_KWARGS[4])

    if name.startswith("ABL"):
        rest  = name[3:]              # e.g. "2_DenseNet121"
        level = int(rest[0])
        bname = rest[2:]
        if bname not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {bname}")
        if level == 0:
            return BaselineModel(BACKBONE_REGISTRY[bname], nc)
        return TCDAModel(BACKBONE_REGISTRY[bname], nc, **_ABL_KWARGS[level])

    raise ValueError(f"Cannot parse model name: '{name}'")


# ==========================================================================
# Data transforms
# ==========================================================================

def get_transforms():
    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(30),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
        v2.RandomGrayscale(p=0.05),
        v2.RandomAffine(degrees=0, shear=10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(_MEAN, _STD),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(_MEAN, _STD),
    ])
    return train_tf, val_tf


# ==========================================================================
# Per-seed test metric saver
# ==========================================================================

def save_test_metrics(model_name, seed, classes,
                      y_true, y_pred, y_prob, metrics_dir):
    rows   = []
    report = classification_report(
        y_true, y_pred, target_names=classes,
        output_dict=True, zero_division=0,
    )
    per_auc = []
    for ci, cls in enumerate(classes):
        r           = report[cls]
        fpr, tpr, _ = roc_curve((y_true == ci).astype(int), y_prob[:, ci])
        cls_auc     = auc(fpr, tpr)
        per_auc.append(cls_auc)
        cm_b = confusion_matrix(
            (y_true == ci).astype(int), (y_pred == ci).astype(int)
        )
        tn = cm_b[0, 0] if cm_b.shape == (2, 2) else 0
        fp = cm_b[0, 1] if cm_b.shape == (2, 2) else 0
        rows.append({
            "model": model_name, "seed": seed, "class": cls,
            "precision":   round(r["precision"], 4),
            "recall":      round(r["recall"],    4),
            "f1_score":    round(r["f1-score"],  4),
            "support":     int(r["support"]),
            "auc":         round(cls_auc,         4),
            "specificity": round(tn / (tn + fp + EPS), 4),
        })

    acc   = accuracy_score(y_true, y_pred)
    mprec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mrec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    mf1   = f1_score(y_true, y_pred,        average="macro", zero_division=0)
    mauc  = float(np.mean(per_auc))

    for tag, vals in [
        ("MACRO_AVG", {"precision": mprec, "recall": mrec,
                       "f1_score": mf1,   "auc": mauc}),
        ("ACCURACY",  {"f1_score": acc}),
    ]:
        row = {
            "model": model_name, "seed": seed, "class": tag,
            "precision": "", "recall": "", "f1_score": "",
            "support": int(y_true.shape[0]), "auc": "", "specificity": "",
        }
        for k, v in vals.items():
            row[k] = round(v, 4)
        rows.append(row)

    out_path = Path(metrics_dir) / f"test_metrics_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return acc, mf1, mprec, mrec, mauc


# ==========================================================================
# Single-seed train + evaluate  (no GradCAM)
# ==========================================================================

def run_one_seed(seed: int, model_name: str, results_csv: Path) -> dict:

    # -- skip if already done ---------------------------------------------
    if results_csv.exists():
        ex   = pd.read_csv(results_csv)
        done = ex[(ex["model"] == model_name) & (ex["seed"] == seed)]
        if not done.empty:
            print(f"  => {model_name} seed={seed} already done — skipping")
            return done.iloc[0].to_dict()

    set_seed(seed)
    dirs  = get_dirs(model_name)
    paths = get_paths(model_name, seed, dirs)

    print(f"\n{'='*72}")
    print(f"  Seed {seed}  |  {model_name}")
    print(f"{'='*72}")

    train_tf, val_tf = get_transforms()

    full_train   = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds      = datasets.ImageFolder(TEST_PATH,  transform=val_tf)
    classes      = full_train.classes
    nc           = len(classes)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=0.15,
        stratify=full_train.targets,
        random_state=seed,
    )
    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_val_ref, val_idx),
        batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False,
    )

    model       = create_model(model_name, nc).to(DEVICE)
    bb_params   = list(model.bb.parameters())
    head_params = [p for p in model.parameters()
                   if not any(id(p) == id(q) for q in bb_params)]
    optimizer   = optim.AdamW(
        [{"params": bb_params,   "lr": LR * 0.1},
         {"params": head_params, "lr": LR}],
        weight_decay=1e-2,
    )
    criterion = FocalLoss()

    best_loss    = float("inf")
    best_state   = None
    patience_ctr = 0
    history      = {k: [] for k in
                    ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}

    # ---- training loop --------------------------------------------------
    for epoch in range(EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        cur_lr_head = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, LR)
        cur_lr_bb   = cur_lr_head * 0.1
        optimizer.param_groups[0]["lr"] = cur_lr_bb
        optimizer.param_groups[1]["lr"] = cur_lr_head

        use_cutmix = (epoch % 2 == 0)
        cm_alpha   = CUTMIX_ALPHA * 0.5 * (
            1 - math.cos(math.pi * min(epoch / 20.0, 1.0))
        )
        apply_aug  = cm_alpha > 0.05

        model.train()
        tl = tc = tt = 0

        for bidx, (x, y) in enumerate(train_loader):
            try:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if apply_aug and x.size(0) > 1:
                    if use_cutmix:
                        x_a, ya, yb, lam = cutmix_data(x, y, cm_alpha)
                    else:
                        x_a, ya, yb, lam = mixup_data(x, y, MIXUP_ALPHA)
                    logits = model(x_a)[0]
                    loss   = aug_criterion(criterion, logits, ya, yb, lam)
                else:
                    logits = model(x)[0]
                    loss   = criterion(logits, y)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                tl += loss.item() * x.size(0)
                tc += (logits.argmax(1) == y).sum().item()
                tt += y.size(0)
            except RuntimeError as e:
                print(f"    ! batch {bidx} skipped: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        train_loss = tl / max(tt, 1)
        train_acc  = tc / max(tt, 1)

        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)[0]
                vl += criterion(logits, y).item() * x.size(0)
                vc += (logits.argmax(1) == y).sum().item()
                vt += y.size(0)
        val_loss = vl / max(vt, 1)
        val_acc  = vc / max(vt, 1)

        for k, v in zip(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
            [epoch + 1, round(train_loss, 6), round(train_acc, 6),
             round(val_loss, 6),  round(val_acc, 6)],
        ):
            history[k].append(v)

        print(f"  [{epoch+1:2d}/{EPOCHS}]  "
              f"Tr {train_loss:.4f}/{train_acc:.4f}  "
              f"Va {val_loss:.4f}/{val_acc:.4f}  "
              f"lr_head={cur_lr_head:.2e}")

        if val_loss < best_loss:
            best_loss    = val_loss
            best_state   = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  => Early stopping.")
                break

    model.load_state_dict(best_state)

    # ---- save training curves -------------------------------------------
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(history["epoch"], history["train_loss"], "b-",  lw=2.2, label="Train Loss")
    ax1.plot(history["epoch"], history["val_loss"],   "c--", lw=2.2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["train_acc"], "r-",  lw=2.2, alpha=0.85, label="Train Acc")
    ax2.plot(history["epoch"], history["val_acc"],   "m--", lw=2.2, alpha=0.85, label="Val Acc")
    ax2.set_ylabel("Accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right", prop={"weight": "bold"})
    ax1.set_title(f"{model_name}  —  Seed {seed}")
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(paths["curves_png"], dpi=160, bbox_inches="tight")
    plt.close()

    # ---- test inference (batched, no GradCAM) ---------------------------
    model.eval()
    y_true_l, y_pred_l, y_prob_l = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x      = x.to(DEVICE)
            logits = model(x)[0]
            prob   = F.softmax(logits, dim=1)
            pred   = logits.argmax(1)
            y_true_l.append(y.numpy())
            y_pred_l.append(pred.cpu().numpy())
            y_prob_l.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true_l)
    y_pred = np.concatenate(y_pred_l)
    y_prob = np.concatenate(y_prob_l)

    # ---- per-class & macro metrics --------------------------------------
    acc, mf1, mprec, mrec, mauc = save_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob,
        metrics_dir=dirs["metrics"],
    )

    # ---- confusion matrix -----------------------------------------------
    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(paths["cm_csv"])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_df, annot=True, fmt="d", cmap="Blues",
        linewidths=0.6, linecolor="gray", ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}  Seed {seed}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    plt.tight_layout()
    plt.savefig(paths["cm_png"], dpi=150, bbox_inches="tight")
    plt.close()

    # ---- ROC curves per seed --------------------------------------------
    roc_rows = []
    fig, ax  = plt.subplots(figsize=(9, 6))
    for j, cls in enumerate(classes):
        fpr, tpr, thr = roc_curve((y_true == j).astype(int), y_prob[:, j])
        roc_auc       = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, label=f"{cls} (AUC={roc_auc:.3f})")
        for f, t, th in zip(fpr, tpr, thr):
            roc_rows.append({
                "model": model_name, "seed": seed, "class": cls,
                "fpr": round(float(f), 6), "tpr": round(float(t), 6),
                "threshold": round(float(th), 6), "auc": round(roc_auc, 6),
            })
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.legend(prop={"weight": "bold"})
    ax.grid(True, alpha=0.3)
    ax.set_title(f"ROC — {model_name}  Seed {seed}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.tight_layout()
    plt.savefig(paths["roc_png"], dpi=150, bbox_inches="tight")
    plt.close()
    pd.DataFrame(roc_rows).to_csv(paths["roc_data"], index=False)

    result = {
        "seed":            seed,
        "model":           model_name,
        "acc":             round(acc,   4),
        "f1_macro":        round(mf1,   4),
        "precision_macro": round(mprec, 4),
        "recall_macro":    round(mrec,  4),
        "auc_macro":       round(mauc,  4),
    }
    print(f"  [ok] Acc={acc:.4f}  F1={mf1:.4f}  AUC={mauc:.4f}")
    return result


# ==========================================================================
# Statistical helpers
# ==========================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = math.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2 + EPS)
    )
    return float((a.mean() - b.mean()) / (pooled + EPS))


def effect_label(d: float) -> str:
    a = abs(d)
    if a >= 0.8: return "large"
    if a >= 0.5: return "medium"
    if a >= 0.2: return "small"
    return "negligible"


def load_cm_as_arrays(model_name: str, seed: int):
    p   = SAVE_DIR / model_name / "metrics" / f"cm_seed{seed}.csv"
    df  = pd.read_csv(p, index_col=0)
    cls = list(df.index)
    cm  = df.values.astype(int)
    yt, yp = [], []
    for ti in range(len(cls)):
        for pi in range(len(cls)):
            n = cm[ti, pi]
            yt.extend([ti] * n)
            yp.extend([pi] * n)
    return np.array(yt), np.array(yp)


def mcnemar_pair(y_true, pa, pb):
    ca = (pa == y_true)
    cb = (pb == y_true)
    b  = int(np.sum(ca & ~cb))
    c  = int(np.sum(~ca & cb))
    if b + c == 0:
        return np.nan, np.nan, b, c
    chi2_v = (abs(b - c) - 1.0) ** 2 / (b + c)
    p      = 1.0 - chi2.cdf(chi2_v, df=1)
    return chi2_v, p, b, c


# ==========================================================================
# Wilcoxon signed-rank test
# ==========================================================================

def run_wilcoxon(df: pd.DataFrame, model_list: list, out_name: str) -> pd.DataFrame:
    print(f"\n=== Wilcoxon Signed-Rank Test [{out_name}] ===")
    rows = []
    for metric in METRICS:
        for m1, m2 in itertools.combinations(model_list, 2):
            s1 = df[df["model"] == m1][metric].values
            s2 = df[df["model"] == m2][metric].values
            if len(s1) < 2 or len(s2) < 2:
                continue
            try:
                w, p = stats.wilcoxon(s1, s2, zero_method="wilcox",
                                      correction=False)
            except ValueError:
                w, p = np.nan, np.nan
            d     = cohens_d(s1, s2)
            delta = s1.mean() - s2.mean()
            sig   = (not np.isnan(p)) and (p < ALPHA_STAT)
            rows.append({
                "metric":           metric,
                "model_A":          m1,
                "model_B":          m2,
                "mean_A":           round(s1.mean(), 4),
                "mean_B":           round(s2.mean(), 4),
                "delta_A_minus_B":  round(delta, 6),
                "wilcoxon_W":       round(w, 2) if not np.isnan(w) else np.nan,
                "p_value":          round(p, 6) if not np.isnan(p) else np.nan,
                "cohens_d":         round(d, 4),
                "effect_size":      effect_label(abs(d)),
                "significant_005":  sig,
                "n_seeds":          len(s1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / f"wilcoxon_{out_name}.csv", index=False)
    print(f"  [saved] wilcoxon_{out_name}.csv")
    return out


# ==========================================================================
# Pooled McNemar test
# ==========================================================================

def run_mcnemar(model_list: list, out_name: str) -> pd.DataFrame:
    print(f"\n=== McNemar Pooled [{out_name}] ===")
    pair_b, pair_c = {}, {}
    for seed in SEEDS:
        preds, y_ref = {}, None
        for m in model_list:
            try:
                yt, yp   = load_cm_as_arrays(m, seed)
                preds[m] = yp
                if y_ref is None:
                    y_ref = yt
            except Exception as e:
                print(f"  ! {m} seed {seed}: {e}")
                continue
        if y_ref is None:
            continue
        for m1, m2 in itertools.combinations(list(preds.keys()), 2):
            _, _, b, c = mcnemar_pair(y_ref, preds[m1], preds[m2])
            key = (m1, m2)
            pair_b[key] = pair_b.get(key, 0) + b
            pair_c[key] = pair_c.get(key, 0) + c

    rows = []
    for (m1, m2) in pair_b:
        b_t, c_t = pair_b[(m1, m2)], pair_c[(m1, m2)]
        if b_t + c_t == 0:
            chi2_v, p = np.nan, np.nan
        else:
            chi2_v = (abs(b_t - c_t) - 1.0) ** 2 / (b_t + c_t)
            p      = 1.0 - chi2.cdf(chi2_v, df=1)
        sig    = (not np.isnan(p)) and (p < ALPHA_STAT)
        winner = m1 if b_t > c_t else (m2 if c_t > b_t else "tie")
        rows.append({
            "model_A":         m1,
            "model_B":         m2,
            "pooled_b":        b_t,
            "pooled_c":        c_t,
            "chi2":            round(chi2_v, 4) if not np.isnan(chi2_v) else np.nan,
            "p_value":         round(p, 6)      if not np.isnan(p)      else np.nan,
            "significant_005": sig,
            "winner":          winner,
            "n_seeds_pooled":  len(SEEDS),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / f"mcnemar_{out_name}.csv", index=False)
    print(f"  [saved] mcnemar_{out_name}.csv")
    return out


# ==========================================================================
# Per-metric individual CSV + PNG
# ==========================================================================

def save_per_metric_files(df: pd.DataFrame, model_list: list,
                          tag: str, short_fn=None):
    if short_fn is None:
        short_fn = lambda m: m

    out_dir = SAVE_DIR / "per_metric" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    order   = [m for m in model_list if m in df["model"].unique()]
    palette = sns.color_palette("tab10", len(order))

    for metric in METRICS:
        label = METRIC_LABELS[metric]

        # Individual CSV for this metric
        sub = df[df["model"].isin(order)][["model", "seed", metric]].copy()
        sub = sub.rename(columns={metric: label})
        sub.to_csv(out_dir / f"{metric}.csv", index=False)

        # Boxplot PNG
        fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.8), 6))
        sns.boxplot(
            x="model", y=metric,
            data=df[df["model"].isin(order)],
            order=order, palette=palette,
            width=0.55, ax=ax, linewidth=1.8,
        )
        sns.stripplot(
            x="model", y=metric,
            data=df[df["model"].isin(order)],
            order=order, color="k", size=5.5, jitter=0.18, ax=ax,
        )
        ax.set_title(f"{label} — {tag}  (n={len(SEEDS)} seeds)")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticklabels(
            [short_fn(m) for m in order],
            rotation=35, ha="right", fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()

    print(f"  [saved] per-metric files ? {out_dir}")


# ==========================================================================
# Aggregated summary builder
# ==========================================================================

def build_aggregate(df: pd.DataFrame, model_list: list) -> pd.DataFrame:
    sub      = df[df["model"].isin(model_list)]
    agg_spec = {m: ["mean", "std", "min", "max"] for m in METRICS}
    agg      = sub.groupby("model").agg(agg_spec).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    valid    = [m for m in model_list if m in agg.index]
    return agg.reindex(valid)


# ==========================================================================
# Main comparison plots
# ==========================================================================

def save_main_comparison_plots(df: pd.DataFrame, agg: pd.DataFrame,
                                model_list: list):
    palette = sns.color_palette("tab10", len(model_list))
    order   = [m for m in model_list if m in df["model"].unique()]

    def short(m):
        return m.replace("BASE_", "B-").replace("TCDA_", "T-")

    # ---- combined boxplot (all metrics side by side) --------------------
    fig, axes = plt.subplots(1, len(METRICS), figsize=(36, 6.5), sharey=False)
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(
            x="model", y=metric,
            data=df[df["model"].isin(order)],
            order=order, palette=palette[:len(order)],
            width=0.55, ax=ax, linewidth=1.8,
        )
        sns.stripplot(
            x="model", y=metric,
            data=df[df["model"].isin(order)],
            order=order, color="k", size=5.5, jitter=0.18, ax=ax,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticklabels([short(m) for m in order],
                           rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"TCDA-Net v2 — All Models  (n={len(SEEDS)} seeds)\n"
        f"B-=Baseline  T-=+TCDA"
    )
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "boxplot_all_metrics.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ---- barplot per metric ---------------------------------------------
    for metric in METRICS:
        mc_col = f"{metric}_mean"
        sc_col = f"{metric}_std"
        if mc_col not in agg.columns:
            continue
        valid  = [m for m in order if m in agg.index]
        means  = agg.loc[valid, mc_col].values
        stds   = agg.loc[valid, sc_col].values
        x      = np.arange(len(valid))

        fig, ax = plt.subplots(figsize=(max(14, len(valid) * 1.4), 6.5))
        bars    = ax.bar(
            x, means, yerr=stds, capsize=7,
            color=palette[:len(valid)], edgecolor="k", linewidth=1.2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([short(m) for m in valid],
                           rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]} ± Std")
        ax.set_title(
            f"{METRIC_LABELS[metric]} — All Models  (n={len(SEEDS)} seeds)"
        )
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
        for bar, m_val, s_val in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s_val + 0.006,
                f"{m_val:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"barplot_{metric}.png",
                    dpi=180, bbox_inches="tight")
        plt.close()

    # ---- TCDA gain per backbone × metric --------------------------------
    gain_rows = []
    for bname in BACKBONE_NAMES:
        base_n = f"BASE_{bname}"
        tcda_n = f"TCDA_{bname}"
        for metric in METRICS:
            mc = f"{metric}_mean"
            if mc not in agg.columns:
                continue
            bm = agg.loc[base_n, mc] if base_n in agg.index else np.nan
            tm = agg.loc[tcda_n, mc] if tcda_n in agg.index else np.nan
            gain_rows.append({
                "backbone": bname,
                "metric":   METRIC_LABELS[metric],
                "gain":     round(float(tm - bm), 4),
            })
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(SAVE_DIR / "tcda_gain.csv", index=False)

    gp     = gain_df.pivot(index="backbone", columns="metric", values="gain")
    col_o  = [METRIC_LABELS[m] for m in METRICS if METRIC_LABELS[m] in gp.columns]
    gp     = gp[col_o]
    x_pos  = np.arange(len(gp))
    w      = 0.14
    pal2   = sns.color_palette("Set2", len(gp.columns))

    fig, ax = plt.subplots(figsize=(14, 6))
    for ci, col in enumerate(gp.columns):
        offset = (ci - len(gp.columns) / 2 + 0.5) * w
        vals   = gp[col].values
        bars   = ax.bar(
            x_pos + offset, vals, w, label=col,
            color=pal2[ci], edgecolor="k", linewidth=0.8,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.001 if v >= 0 else -0.004),
                f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.5, fontweight="bold",
            )
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gp.index, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("TCDA Gain (TCDA_X - BASE_X)")
    ax.set_title(
        f"TCDA v2 Gain per Backbone × Metric  (n={len(SEEDS)} seeds)"
    )
    ax.legend(loc="upper right", prop={"weight": "bold"})
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "tcda_gain_barplot.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ---- McNemar p-value heatmap ----------------------------------------
    _mcnemar_heatmap(order, "main_models")

    print("  [ok] Main comparison plots saved.")


def _mcnemar_heatmap(model_list: list, tag: str):
    csv_path = SAVE_DIR / f"mcnemar_{tag}.csv"
    if not csv_path.exists():
        return
    try:
        mc_df = pd.read_csv(csv_path)
        p_mat = pd.DataFrame(
            np.ones((len(model_list), len(model_list))),
            index=model_list, columns=model_list,
        )
        for _, row in mc_df.iterrows():
            pv = row["p_value"]
            if pd.isna(pv):
                continue
            a, b = row["model_A"], row["model_B"]
            if a in p_mat.index and b in p_mat.columns:
                p_mat.loc[a, b] = pv
                p_mat.loc[b, a] = pv

        def short(m):
            return (m.replace("BASE_", "B-")
                     .replace("TCDA_", "T-")
                     .replace("ABL", "A"))

        sm     = {m: short(m) for m in model_list}
        p_plot = p_mat.rename(index=sm, columns=sm)
        n      = len(model_list)
        sz     = max(10, n * 0.9)
        fig, ax = plt.subplots(figsize=(sz, sz))
        sns.heatmap(
            p_plot.astype(float), annot=True, fmt=".4f",
            cmap="RdYlGn_r", vmin=0, vmax=0.1,
            mask=np.eye(n, dtype=bool), ax=ax,
            linewidths=0.4, linecolor="white",
            annot_kws={"size": 8, "weight": "bold"},
            cbar_kws={"label":
                      f"McNemar p (pooled, n={len(SEEDS)} seeds)"},
        )
        ax.set_title(f"Pairwise McNemar p-values [{tag}]")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35,
                           ha="right", fontweight="bold", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontweight="bold", fontsize=9)
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"mcnemar_heatmap_{tag}.png",
                    dpi=160, bbox_inches="tight")
        plt.close()
        print(f"  [saved] mcnemar_heatmap_{tag}.png")
    except Exception as e:
        print(f"  ! McNemar heatmap [{tag}] skipped: {e}")


# ==========================================================================
# Ablation plots
# ==========================================================================

def save_ablation_plots(df: pd.DataFrame, agg: pd.DataFrame):
    abl_dir = SAVE_DIR / "ablation_plots"
    abl_dir.mkdir(parents=True, exist_ok=True)

    lvl_labels_short = [ABL_LABELS[l] for l in ABL_LEVELS]

    # ---- per backbone: one figure, all metrics side by side -------------
    for bname in BACKBONE_NAMES:
        models_bb = [f"ABL{lvl}_{bname}" for lvl in ABL_LEVELS]
        valid_bb  = [m for m in models_bb if m in df["model"].unique()]
        if not valid_bb:
            continue

        sub_agg = agg.reindex(valid_bb)
        n_col   = len(METRICS)
        pal     = sns.color_palette("Blues_d", len(ABL_LEVELS))

        fig, axes = plt.subplots(1, n_col, figsize=(n_col * 5, 5.5))
        if n_col == 1:
            axes = [axes]

        for ax, metric in zip(axes, METRICS):
            mc_col = f"{metric}_mean"
            sc_col = f"{metric}_std"
            means  = (sub_agg[mc_col].values
                      if mc_col in sub_agg.columns
                      else np.zeros(len(valid_bb)))
            stds   = (sub_agg[sc_col].values
                      if sc_col in sub_agg.columns
                      else np.zeros(len(valid_bb)))
            x      = np.arange(len(valid_bb))
            lbs    = [ABL_LABELS[lvl] for lvl in ABL_LEVELS[:len(valid_bb)]]
            bars   = ax.bar(
                x, means, yerr=stds, capsize=6,
                color=pal[:len(valid_bb)], edgecolor="k", linewidth=1.1,
            )
            for bar, m_val, s_val in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s_val + 0.005,
                    f"{m_val:.4f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(lbs, rotation=30, ha="right", fontsize=8.5)
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(METRIC_LABELS[metric])
            ax.set_ylim(0, 1.12)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Ablation Study — {bname}  (n={len(SEEDS)} seeds)", y=1.02
        )
        plt.tight_layout()
        plt.savefig(abl_dir / f"ablation_{bname}.png",
                    dpi=180, bbox_inches="tight")
        plt.close()

    # ---- cross-backbone: one plot per metric ----------------------------
    pal_bb = sns.color_palette("tab10", len(BACKBONE_NAMES))
    w      = 0.18
    x_pos  = np.arange(len(ABL_LEVELS))

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(14, 6))
        for bi, bname in enumerate(BACKBONE_NAMES):
            means_bb, stds_bb = [], []
            for lvl in ABL_LEVELS:
                mname = f"ABL{lvl}_{bname}"
                sub   = df[df["model"] == mname][metric].values
                means_bb.append(sub.mean() if len(sub) > 0 else 0.0)
                stds_bb.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
            offset = (bi - len(BACKBONE_NAMES) / 2 + 0.5) * w
            ax.bar(
                x_pos + offset, means_bb, w,
                yerr=stds_bb, capsize=4,
                label=bname, color=pal_bb[bi], edgecolor="k", linewidth=0.8,
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lvl_labels_short, rotation=25,
                           ha="right", fontsize=10)
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]} ± Std")
        ax.set_title(
            f"Ablation — {METRIC_LABELS[metric]} across Backbones"
            f"  (n={len(SEEDS)} seeds)"
        )
        ax.legend(loc="lower right", prop={"weight": "bold"})
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(abl_dir / f"ablation_cross_{metric}.png",
                    dpi=180, bbox_inches="tight")
        plt.close()

    print(f"  [ok] Ablation plots saved ? {abl_dir}")


# ==========================================================================
# Superiority table
# ==========================================================================

def build_superiority_table(df: pd.DataFrame,
                             wilc_df: pd.DataFrame,
                             mc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bname in BACKBONE_NAMES:
        proposed = f"TCDA_{bname}"
        baseline = f"BASE_{bname}"
        sp_df    = df[df["model"] == proposed]
        sb_df    = df[df["model"] == baseline]
        if sp_df.empty or sb_df.empty:
            continue
        for metric in METRICS:
            sp    = sp_df[metric].values
            sb    = sb_df[metric].values
            d     = cohens_d(sp, sb)
            delta = sp.mean() - sb.mean()

            wrow  = wilc_df[
                (wilc_df["metric"] == metric) & (
                    ((wilc_df["model_A"] == proposed) &
                     (wilc_df["model_B"] == baseline)) |
                    ((wilc_df["model_A"] == baseline) &
                     (wilc_df["model_B"] == proposed))
                )
            ]
            p_w   = wrow.iloc[0]["p_value"] if not wrow.empty else np.nan

            mcrow = mc_df[
                ((mc_df["model_A"] == proposed) &
                 (mc_df["model_B"] == baseline)) |
                ((mc_df["model_A"] == baseline) &
                 (mc_df["model_B"] == proposed))
            ]
            p_mc  = mcrow.iloc[0]["p_value"] if not mcrow.empty else np.nan

            sig   = (delta > 0) and (
                (not np.isnan(p_mc) and p_mc < ALPHA_STAT) or
                (not np.isnan(p_w)  and p_w  < ALPHA_STAT)
            )
            rows.append({
                "Backbone":      bname,
                "Metric":        METRIC_LABELS[metric],
                "TCDA_mean±std": f"{sp.mean():.4f}±{sp.std(ddof=1):.4f}",
                "BASE_mean±std": f"{sb.mean():.4f}±{sb.std(ddof=1):.4f}",
                "Delta":         f"{delta:+.4f}",
                "Cohens_d":      f"{d:.4f}",
                "Effect":        effect_label(abs(d)),
                "Wilcoxon_p":    f"{p_w:.4f}"  if not np.isnan(p_w)  else "n/a",
                "McNemar_p":     f"{p_mc:.4f}" if not np.isnan(p_mc) else "n/a",
                "TCDA_wins":     ("YES *" if sig
                                  else ("numerically" if delta > 0 else "NO")),
            })

    sup_df = pd.DataFrame(rows)
    sup_df.to_csv(SAVE_DIR / "superiority_table.csv", index=False)
    print("\n=== Superiority Table (TCDA vs Baseline per backbone) ===")
    print(sup_df.to_string(index=False))
    return sup_df


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    print("\n" + "=" * 72)
    print("  TCDA-Net v2 — Brain Tumour Classification  (4-class)")
    print(f"  Backbones : {BACKBONE_NAMES}")
    print(f"  Seeds     : {SEEDS}  (n={len(SEEDS)})")
    print(f"  Device    : {DEVICE}")
    print(f"  Epochs    : {EPOCHS}  |  Batch : {BATCH}  |  DESC_DIM : {DESC_DIM}")
    print(f"  EfficientNetB4 : EXCLUDED from all experiments")
    print(f"  GradCAM        : DISABLED")
    print(f"  Statistical    : Wilcoxon + McNemar ONLY")
    print(f"\n  Ablation levels:")
    for k, v in ABL_LABELS.items():
        print(f"    ABL{k} = {v.replace(chr(10), ' ')}")
    print("=" * 72)

    # =======================================================================
    # PHASE 1 — Train main models (BASE + TCDA, 4 backbones)
    # =======================================================================
    main_results_csv = SAVE_DIR / "main_results.csv"
    main_results     = []

    for model_name in MAIN_MODELS:
        for seed in SEEDS:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res = run_one_seed(seed, model_name, main_results_csv)
            main_results.append(res)
            (pd.DataFrame(main_results)
               .drop_duplicates(subset=["model", "seed"])
               .reset_index(drop=True)
               .to_csv(main_results_csv, index=False))

    df_main = (pd.read_csv(main_results_csv)
                 .drop_duplicates(subset=["model", "seed"])
                 .reset_index(drop=True))

    # =======================================================================
    # PHASE 2 — Train ablation intermediate models (ABL1, ABL2, ABL3)
    #           ABL0 == BASE and ABL4 == TCDA — results reused, not retrained
    # =======================================================================
    abl_interm_csv = SAVE_DIR / "ablation_interm_results.csv"
    abl_interm     = []

    for model_name in ABL_TRAIN_MODELS:
        for seed in SEEDS:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res = run_one_seed(seed, model_name, abl_interm_csv)
            abl_interm.append(res)
            (pd.DataFrame(abl_interm)
               .drop_duplicates(subset=["model", "seed"])
               .reset_index(drop=True)
               .to_csv(abl_interm_csv, index=False))

    df_abl_interm = (pd.read_csv(abl_interm_csv)
                       .drop_duplicates(subset=["model", "seed"])
                       .reset_index(drop=True))

    # Build reuse rows: ABL0 <- BASE results, ABL4 <- TCDA results
    abl_reuse_rows = []
    for bname in BACKBONE_NAMES:
        for seed in SEEDS:
            # ABL0 = BASE
            row = df_main[
                (df_main["model"] == f"BASE_{bname}") &
                (df_main["seed"]  == seed)
            ]
            if not row.empty:
                r = row.iloc[0].to_dict()
                r["model"] = f"ABL0_{bname}"
                abl_reuse_rows.append(r)
            # ABL4 = TCDA
            row = df_main[
                (df_main["model"] == f"TCDA_{bname}") &
                (df_main["seed"]  == seed)
            ]
            if not row.empty:
                r = row.iloc[0].to_dict()
                r["model"] = f"ABL4_{bname}"
                abl_reuse_rows.append(r)

    df_abl_reuse = pd.DataFrame(abl_reuse_rows)

    # Merge all ablation rows
    df_abl_all = (pd.concat([df_abl_reuse, df_abl_interm], ignore_index=True)
                    .drop_duplicates(subset=["model", "seed"])
                    .reset_index(drop=True))
    df_abl_all.to_csv(SAVE_DIR / "ablation_all_results.csv", index=False)

    # =======================================================================
    # PHASE 3 — Aggregate
    # =======================================================================
    agg_main = build_aggregate(df_main, MAIN_MODELS)
    agg_main.to_csv(SAVE_DIR / "summary_main_aggregated.csv")
    print("\n=== Main Aggregated Summary ===")
    print(agg_main.to_string())

    agg_abl = build_aggregate(df_abl_all, ABLATION_MODELS)
    agg_abl.to_csv(SAVE_DIR / "summary_ablation_aggregated.csv")
    print("\n=== Ablation Aggregated Summary ===")
    print(agg_abl.to_string())

    # =======================================================================
    # PHASE 4 — Per-metric individual CSV + PNG
    # =======================================================================
    save_per_metric_files(
        df_main, MAIN_MODELS, tag="main_models",
        short_fn=lambda m: m.replace("BASE_", "B-").replace("TCDA_", "T-"),
    )
    save_per_metric_files(
        df_abl_all, ABLATION_MODELS, tag="ablation_models",
        short_fn=lambda m: m,
    )

    # =======================================================================
    # PHASE 5 — Statistical tests
    # =======================================================================
    # Main: all 8 models pairwise
    wilc_main = run_wilcoxon(df_main, MAIN_MODELS, out_name="main_models")
    mc_main   = run_mcnemar(MAIN_MODELS,            out_name="main_models")

    # Ablation: pairwise within each backbone's 5 conditions
    for bname in BACKBONE_NAMES:
        abl_bb = [f"ABL{lvl}_{bname}" for lvl in ABL_LEVELS]
        valid  = [m for m in abl_bb if m in df_abl_all["model"].unique()]
        if len(valid) >= 2:
            run_wilcoxon(df_abl_all, valid, out_name=f"ablation_{bname}")
            run_mcnemar(valid,              out_name=f"ablation_{bname}")

    # =======================================================================
    # PHASE 6 — Superiority table (TCDA vs BASE per backbone)
    # =======================================================================
    build_superiority_table(df_main, wilc_main, mc_main)

    # =======================================================================
    # PHASE 7 — Plots
    # =======================================================================
    save_main_comparison_plots(df_main, agg_main, MAIN_MODELS)
    save_ablation_plots(df_abl_all, agg_abl)

    # =======================================================================
    # Done
    # =======================================================================
    print(f"\n{'='*72}")
    print(f"  All outputs saved to : {SAVE_DIR.resolve()}")
    print()
    print(f"  Main results         : main_results.csv")
    print(f"  Ablation results     : ablation_all_results.csv")
    print(f"  Aggregates           : summary_main_aggregated.csv")
    print(f"                         summary_ablation_aggregated.csv")
    print(f"  Statistical tests    : wilcoxon_main_models.csv")
    print(f"                         mcnemar_main_models.csv")
    print(f"                         wilcoxon_ablation_<backbone>.csv (x4)")
    print(f"                         mcnemar_ablation_<backbone>.csv  (x4)")
    print(f"  Per-metric files     : per_metric/main_models/   (5 CSV + 5 PNG)")
    print(f"                         per_metric/ablation_models/ (5 CSV + 5 PNG)")
    print(f"  Comparison plots     : boxplot_all_metrics.png")
    print(f"                         barplot_<metric>.png  (x5)")
    print(f"                         tcda_gain_barplot.png")
    print(f"                         mcnemar_heatmap_main_models.png")
    print(f"  Ablation plots       : ablation_plots/ablation_<backbone>.png (x4)")
    print(f"                         ablation_plots/ablation_cross_<metric>.png (x5)")
    print(f"  Superiority table    : superiority_table.csv")
    print("=" * 72)
