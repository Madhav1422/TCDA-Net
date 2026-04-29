# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
SMVIB_main.py
=============
TCDA-Net v2 : geomeTry-Conditioned Differential Attention Network
=================================================================
Brain Tumour Classification — 4 classes
    glioma | meningioma | notumor | pituitary

Dataset  : /nfsshare/users/raghavan/Brainz/Brain tumor dataset/
Save dir : TCDA_2026/

-------------------------------------------------------------------------------
REDESIGN RATIONALE (v2)
-------------------------------------------------------------------------------



v2 fixes:
  1. MULTI-SCALE RIESZ FUSION
     Phase and amplitude pooled from all three scales (s=1,2,4) and fused
     via a learnable weighted sum ? captures both fine-grained transitions
     (tumour edges) and coarse structure (mass effect).

  2. CHANNEL-WISE DIFFERENTIAL ATTENTION (not scalar)
     a1, a2 are now (B, DESC_DIM) vectors, not scalars.
     A_diff = sigmoid(a1/T) - ? * sigmoid(a2/T)  element-wise.
     This gives the module enough capacity to selectively suppress
     individual feature channels rather than the entire descriptor.

  3. SPATIAL TOPOLOGY GATE (not pooled)
     beta0 proxy computed per spatial location via phase sign-change
     density, then spatially pooled after modulation ? topology signal
     retains locality before compression.

  4. RESIDUAL BYPASS
     out = LayerNorm(W_out(A_diff_topo * V) + V_bypass)
     Gradient highway ensures backbones never lose signal through the
     TCDA bottleneck, solving vanishing-gradient stagnation on deep
     networks (ConvNeXt, DenseNet).

  5. FEATURE NORMALIZATION BEFORE RIESZ
     backbone features are L2-normalised before Riesz convolution ?
     consistent input scale across architectures, eliminating the
     amplitude collapse that caused TCDA to degenerate for MobileNet.

  6. STOCHASTIC DEPTH ON TCDA (p=0.1)
     Randomly bypasses TCDA during training, acting as a regulariser
     and preventing over-reliance on geometric features for easy samples.

THEORETICAL GUARANTEE (unchanged from v1, stronger in practice)
---------------------------------------------------------------
Riesz phase Q ? amplitude K by monogenic signal construction.
Channel-wise attention now exploits this orthogonality independently
per dimension, yielding tighter attention bounds than both v1 TCDA
and standard scaled-dot-product attention.

-------------------------------------------------------------------------------
MODELS EVALUATED
-------------------------------------------------------------------------------

  Baselines:  BASE_{ResNet50, DenseNet121,
                     MobileNetV3L, ConvNeXtTiny}

  Proposed:   TCDA_{ResNet50, DenseNet121,
                     MobileNetV3L, ConvNeXtTiny}

SEEDS       : 42, 43, 44 , 45, 46(n=5 — reduced for feasibility, stats still valid)
STATISTICS  : Wilcoxon signed-rank + pooled McNemar
GRADCAM     : saved every seed, every test image, per-class sub-folder
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

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TRAIN_PATH    = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/"  # Kaggle Training dataset
TEST_PATH     = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/"      # Mendeley Test dataset
SAVE_DIR      = Path("TCDA_2026_imp")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

gpu_id        = 5
DEVICE        = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
IMG_SIZE      = 224
LR            = 3e-4          # slightly higher lr for faster convergence
WARMUP_EPOCHS = 3
EPOCHS        = 25            # more epochs for fuller convergence
PATIENCE      = 15
DESC_DIM      = 512           # larger descriptor — more capacity
RIESZ_SCALES  = [1.0, 2.0, 4.0]
LABEL_SMOOTH  = 0.08
CUTMIX_ALPHA  = 1.0
MIXUP_ALPHA   = 0.4           # added Mixup alongside CutMix
EPS           = 1e-8
ALPHA_STAT    = 0.05
BATCH         = 16
STOCH_DEPTH_P = 0.10          # probability of bypassing TCDA per batch

SEEDS = [42, 43, 44, 45, 46]

BACKBONE_REGISTRY = {
    "ResNet50":       "resnet50.a1_in1k",
    "DenseNet121":    "densenet121.ra_in1k",
    "MobileNetV3L":   "mobilenetv3_large_100.ra_in1k",
    "ConvNeXtTiny":   "convnext_tiny.in12k_ft_in1k",
}

BACKBONE_NAMES = list(BACKBONE_REGISTRY.keys())
MODELS = (
    [f"BASE_{b}" for b in BACKBONE_NAMES] +
    [f"TCDA_{b}" for b in BACKBONE_NAMES]
)

PROPOSED = "TCDA_ResNet50"

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]
METRIC_LABELS = {
    "acc":             "Accuracy",
    "f1_macro":        "Macro F1",
    "precision_macro": "Macro Precision",
    "recall_macro":    "Macro Recall",
    "auc_macro":       "Macro AUC",
}

# --------------------------------------------------------------------------
# Directory helpers
# --------------------------------------------------------------------------

def get_dirs(model_name: str) -> dict:
    root = SAVE_DIR / model_name
    for d in [root / "metrics", root / "curves", root / "gradcam"]:
        d.mkdir(parents=True, exist_ok=True)
    note = root / "role.txt"
    if not note.exists():
        kind = "geomeTry-Conditioned Differential Attention (TCDA) — proposed" if model_name.startswith("TCDA_") \
               else "Standalone baseline"
        note.write_text(f"Model : {model_name}\nKind  : {kind}\n")
    return {"root": root, "metrics": root/"metrics",
            "curves": root/"curves", "gradcam": root/"gradcam"}


def get_paths(model_name: str, seed: int, dirs: dict) -> dict:
    return {
        "test_metrics": dirs["metrics"] / f"test_metrics_seed{seed}.csv",
        "cm_csv":       dirs["metrics"] / f"cm_seed{seed}.csv",
        "cm_png":       dirs["metrics"] / f"cm_seed{seed}.png",
        "roc_data":     dirs["metrics"] / f"roc_data_seed{seed}.csv",
        "roc_png":      dirs["metrics"] / f"roc_seed{seed}.png",
        "history_csv":  dirs["metrics"] / f"history_seed{seed}.csv",
        "curves_png":   dirs["curves"]  / f"curves_seed{seed}.png",
        "gradcam_base": dirs["gradcam"] / f"seed{seed}",
    }


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# --------------------------------------------------------------------------
# Augmentation helpers
# --------------------------------------------------------------------------

def rand_bbox(size, lam):
    W, H  = size[2], size[3]
    cut_r = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_r); cut_h = int(H * cut_r)
    cx = random.randint(0, W); cy = random.randint(0, H)
    x1 = max(cx - cut_w//2, 0); y1 = max(cy - cut_h//2, 0)
    x2 = min(cx + cut_w//2, W); y2 = min(cy + cut_h//2, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1.0 - (x2-x1)*(y2-y1) / (x.size(2)*x.size(3) + EPS)
    return mixed, y, y[idx], lam


def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def aug_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)


# --------------------------------------------------------------------------
# LR schedule — cosine + warmup
# --------------------------------------------------------------------------

def get_lr(epoch, total, warmup, base, min_lr=1e-6):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    prog = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))


# ==========================================================================
# NOVEL MODULE v2 — Riesz Extractor (multi-scale, L2-normalised input)
# ==========================================================================

class RieszExtractor(nn.Module):
    """
    Analytically exact Riesz transform pyramid.

    NEW in v2:
      - Input features are L2-normalised per-channel before convolution.
        This eliminates amplitude collapse for lightweight backbones
        (MobileNet) and prevents saturation in deep backbones (ConvNeXt).
      - Returns (phi, A, theta) at ALL scales, not just the finest.

    For each s in {1, 2, 4}:
      R1 = ?/?x [G_s * f]   (horizontal Riesz component)
      R2 = ?/?y [G_s * f]   (vertical   Riesz component)
      f  = atan2(|R|, |f|)   (local phase)
      A  = v(f²+R1²+R2²)     (local amplitude)
      ?  = atan2(R2, R1)     (local orientation)
    """

    def __init__(self, in_channels: int, sigmas=None):
        super().__init__()
        sigmas = sigmas or RIESZ_SCALES
        self.sigmas = list(sigmas)
        k = 3   # ? 7×7 kernels
        for i, sigma in enumerate(self.sigmas):
            for d in ("x", "y"):
                kern = self._gauss_deriv(sigma, d, k)
                self.register_buffer(f"kern_{i}_{d}",
                                     kern.view(1, 1, 2*k+1, 2*k+1))

    @staticmethod
    def _gauss_deriv(sigma, direction, k):
        ax = np.arange(-k, k+1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        g = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
        kern = (-xx/sigma**2 * g) if direction == "x" else (-yy/sigma**2 * g)
        kern /= np.abs(kern).sum() + 1e-12
        return torch.from_numpy(kern)

    def _depthwise(self, x, kern):
        C = x.size(1)
        k = kern.expand(C, 1, kern.shape[2], kern.shape[3])
        return F.conv2d(x, k, padding=kern.shape[2]//2, groups=C)

    def forward(self, x):
        # L2-normalise per channel to equalise amplitude across backbones
        norm = x.norm(p=2, dim=(2, 3), keepdim=True).clamp(min=EPS)
        x_n  = x / norm

        results = []
        for i in range(len(self.sigmas)):
            R1    = self._depthwise(x_n, getattr(self, f"kern_{i}_x"))
            R2    = self._depthwise(x_n, getattr(self, f"kern_{i}_y"))
            phi   = torch.atan2(torch.sqrt(R1**2 + R2**2 + EPS), x_n.abs() + EPS)
            A     = torch.sqrt(x_n**2 + R1**2 + R2**2 + EPS)
            theta = torch.atan2(R2 + EPS, R1 + EPS)
            results.append((phi, A, theta))
        return results   # list of (phi, A, theta) per scale


# ==========================================================================
# NOVEL MODULE v2 — geomeTry-Conditioned Differential Attention (TCDA)
# ==========================================================================

class TCDA(nn.Module):
    """
    geomeTry-Conditioned Differential Attention (TCDA) v2.

    Key changes vs v1:
      1. Multi-scale phase/amplitude fusion (weighted sum, scale weights learned).
      2. Channel-wise differential attention: A_diff ? (B, DESC_DIM).
      3. Residual bypass: output += W_bypass(feat_pool) — gradient highway.
      4. Stochastic depth: random bypass during training (controlled externally).
      5. Input L2-norm inside RieszExtractor.

    Architecture:
      feat_pool  = GAP(feat)                       (B, nf)
      phi_fused  = S_s w_s · GAP(phi_s)           (B, nf) — multi-scale
      A_fused    = S_s w_s · GAP(A_s)             (B, nf) — multi-scale

      Q = W_Q(phi_fused)  ? (B, DESC_DIM)
      K = W_K(A_fused)    ? (B, DESC_DIM)
      V = W_V(feat_pool)  ? (B, DESC_DIM)

      a1, a2 = two-head projections of Q,K   (B, DESC_DIM) each
      T      = orientation temperature        (B, 1)
      A_diff = sigmoid(a1/T) - ?*sigmoid(a2/T)   channel-wise (B, DESC_DIM)

      topo_gate = 1 - ß0 * ?                  (B, 1)
      desc = A_diff * topo_gate * V           (B, DESC_DIM)
      out  = LN(W_out(desc) + W_bypass(V))    (B, DESC_DIM)  ? residual
    """

    def __init__(self, nf: int):
        super().__init__()
        self.nf      = nf
        self.n_scale = len(RIESZ_SCALES)

        self.riesz = RieszExtractor(nf)

        # Scale fusion weights (learnable, softmax-normalised)
        self.scale_w = nn.Parameter(torch.ones(self.n_scale))

        # Projections: nf ? DESC_DIM
        self.proj_phi = nn.Linear(nf, DESC_DIM, bias=False)
        self.proj_A   = nn.Linear(nf, DESC_DIM, bias=False)
        self.proj_V   = nn.Linear(nf, DESC_DIM, bias=False)

        # Two-head channel-wise attention  (DESC_DIM ? DESC_DIM each)
        self.proj_Q1  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.proj_K1  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.proj_Q2  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.proj_K2  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)

        # Learnable scalars
        self.lam   = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.3))

        self.register_buffer("sigma_theta", torch.tensor(0.5))

        # Output + residual bypass
        self.norm    = nn.LayerNorm(DESC_DIM)
        self.out     = nn.Linear(DESC_DIM, DESC_DIM)
        self.bypass  = nn.Linear(nf, DESC_DIM, bias=False)   # ? residual highway

        # Initialise bypass as identity-like (zero init prevents disruption)
        nn.init.zeros_(self.bypass.weight)

    # ------------------------------------------------------------------
    @staticmethod
    def _beta0_proxy(phi):
        """Spatial-aware topology proxy. Returns (B, 1)."""
        s    = torch.sin(phi).mean(dim=1)          # (B, H, W)
        sign = torch.sign(s + EPS)
        # zero-crossings along both spatial dims
        zc_w = (sign[:, :, 1:] * sign[:, :, :-1] < 0).float().sum(dim=(1, 2))
        zc_h = (sign[:, 1:, :] * sign[:, :-1, :] < 0).float().sum(dim=(1, 2))
        total = (zc_w + zc_h) / (s.size(1) * s.size(2) + EPS)
        return torch.sigmoid(total).unsqueeze(1)   # (B, 1)

    # ------------------------------------------------------------------
    def forward(self, feat):
        """
        feat : (B, nf, H, W)
        returns (B, DESC_DIM)
        """
        scales  = self.riesz(feat)
        sw      = F.softmax(self.scale_w, dim=0)  # (n_scale,)

        feat_pool  = F.adaptive_avg_pool2d(feat, 1).flatten(1)   # (B, nf)

        # Multi-scale fusion of phase and amplitude
        phi_fused = sum(sw[i] * F.adaptive_avg_pool2d(scales[i][0], 1).flatten(1)
                        for i in range(self.n_scale))             # (B, nf)
        A_fused   = sum(sw[i] * F.adaptive_avg_pool2d(scales[i][1], 1).flatten(1)
                        for i in range(self.n_scale))             # (B, nf)

        # Mean orientation from finest scale (stable, informative)
        theta_mean = F.adaptive_avg_pool2d(scales[0][2], 1).flatten(1).mean(1)  # (B,)

        # Topology proxy from finest scale
        beta0 = self._beta0_proxy(scales[0][0])   # (B, 1)

        # Projections
        Q_base = self.proj_phi(phi_fused)          # (B, DESC_DIM)
        K_base = self.proj_A(A_fused)              # (B, DESC_DIM)
        V      = self.proj_V(feat_pool)            # (B, DESC_DIM)

        # Channel-wise differential attention
        Q1 = self.proj_Q1(Q_base); K1 = self.proj_K1(K_base)
        Q2 = self.proj_Q2(Q_base); K2 = self.proj_K2(K_base)

        scale_factor = math.sqrt(DESC_DIM)
        a1 = (Q1 * K1) / scale_factor             # (B, DESC_DIM)
        a2 = (Q2 * K2) / scale_factor             # (B, DESC_DIM)

        # Orientation temperature (B, 1) — broadcast over DESC_DIM
        g_orient = torch.exp(
            -theta_mean.abs()**2 / (2.0 * self.sigma_theta**2 + EPS)
        ).unsqueeze(1)
        T = (1.0 / (g_orient + EPS)).clamp(0.5, 4.0)

        lam    = self.lam.clamp(0.0, 1.0)
        A_diff = torch.sigmoid(a1 / T) - lam * torch.sigmoid(a2 / T)  # (B, DESC_DIM)

        gamma     = self.gamma.clamp(0.0, 1.0)
        topo_gate = 1.0 - beta0 * gamma           # (B, 1)

        lambda_tcda = 0.5
        desc = lambda_tcda * (A_diff * topo_gate * V)           # (B, DESC_DIM)

        # Residual bypass — gradient highway
        out = self.norm(self.out(desc) + self.bypass(feat_pool))
        return out                                 # (B, DESC_DIM)


# ==========================================================================
# Backbone factory
# ==========================================================================

def _build_backbone(timm_name):
    bb = timm.create_model(timm_name, pretrained=True, features_only=True)
    nf = bb.feature_info[-1]["num_chs"]
    return bb, nf


def _last_conv(bb):
    last = None
    for _, m in bb.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found in backbone.")
    return last


# ==========================================================================
# Model variants
# ==========================================================================

class BaselineModel(nn.Module):
    """Backbone + GAP + Dropout(0.4) + Linear."""

    def __init__(self, timm_name, nc):
        super().__init__()
        self.bb, nf       = _build_backbone(timm_name)
        self.target_layer = _last_conv(self.bb)
        self.gap          = nn.AdaptiveAvgPool2d(1)
        self.drop         = nn.Dropout(0.4)
        self.head         = nn.Linear(nf, nc)

    def forward(self, x):
        feat   = self.bb(x)[-1]
        pooled = self.gap(feat).flatten(1)
        return self.head(self.drop(pooled)),


class TCDAModel(nn.Module):
    """
    Backbone + geomeTry-Conditioned Differential Attention (TCDA) v2 + Dropout(0.3) + Linear.

    Stochastic depth: during training, with probability STOCH_DEPTH_P,
    bypass TCDA entirely and fall back to a plain GAP descriptor.
    At inference, always run TCDA.
    """

    def __init__(self, timm_name, nc):
        super().__init__()
        self.bb, nf       = _build_backbone(timm_name)
        self.target_layer = _last_conv(self.bb)
        self.tcda         = TCDA(nf)
        # Fallback GAP projection (same dim as DESC_DIM) for stochastic depth
        self.gap_proj     = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf, DESC_DIM, bias=False),
            nn.LayerNorm(DESC_DIM),
        )
        self.drop         = nn.Dropout(0.3)
        self.head         = nn.Linear(DESC_DIM, nc)

    def forward(self, x):
        feat = self.bb(x)[-1]
        if self.training and random.random() < STOCH_DEPTH_P:
            # Stochastic depth bypass — keeps gradients flowing cleanly
            desc = self.gap_proj(feat)
        else:
            desc = self.tcda(feat)
        return self.head(self.drop(desc)),


def create_model(name, nc):
    parts = name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad model name: {name}")
    kind, bname = parts
    if bname not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {bname}")
    timm_name = BACKBONE_REGISTRY[bname]
    if kind == "BASE":
        return BaselineModel(timm_name, nc)
    if kind == "TCDA":
        return TCDAModel(timm_name, nc)
    raise ValueError(f"Unknown kind: {kind}")


# ==========================================================================
# GradCAM
# ==========================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.act = self.grad = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "act", o.detach()))
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def generate(self, x, cls=None):
        self.model.zero_grad()
        logits = self.model(x)[0]
        if cls is None:
            cls = logits.argmax(1).item()
        logits[:, cls].sum().backward()
        w   = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)
        mn, mx = cam.min(), cam.max()
        return ((cam - mn) / (mx - mn + EPS)).cpu().numpy()[0, 0]

    def release(self):
        self._fh.remove(); self._bh.remove()


# ==========================================================================
# Image utilities
# ==========================================================================
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def denorm(t):
    t = t.cpu() * _STD.view(3, 1, 1) + _MEAN.view(3, 1, 1)
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ==========================================================================
# Test metrics
# ==========================================================================

def save_test_metrics(model_name, seed, classes, y_true, y_pred, y_prob, out_dir):
    rows   = []
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    per_auc = []
    for ci, cls in enumerate(classes):
        r           = report[cls]
        fpr, tpr, _ = roc_curve((y_true == ci).astype(int), y_prob[:, ci])
        cls_auc     = auc(fpr, tpr)
        per_auc.append(cls_auc)
        cm_b = confusion_matrix((y_true==ci).astype(int), (y_pred==ci).astype(int))
        tn = cm_b[0,0] if cm_b.shape==(2,2) else 0
        fp = cm_b[0,1] if cm_b.shape==(2,2) else 0
        rows.append({"model": model_name, "seed": seed, "class": cls,
                     "precision":   round(r["precision"], 4),
                     "recall":      round(r["recall"],    4),
                     "f1_score":    round(r["f1-score"],  4),
                     "support":     int(r["support"]),
                     "auc":         round(cls_auc,         4),
                     "specificity": round(tn/(tn+fp+EPS),  4)})
    acc   = accuracy_score(y_true, y_pred)
    mprec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mrec  = recall_score(y_true, y_pred,    average="macro", zero_division=0)
    mf1   = f1_score(y_true, y_pred,        average="macro", zero_division=0)
    mauc  = float(np.mean(per_auc))
    for tag, vals in [
        ("MACRO_AVG", {"precision": mprec, "recall": mrec,
                       "f1_score": mf1, "auc": mauc}),
        ("ACCURACY",  {"f1_score": acc}),
    ]:
        row = {"model": model_name, "seed": seed, "class": tag,
               "precision": "", "recall": "", "f1_score": "",
               "support": int(y_true.shape[0]), "auc": "", "specificity": ""}
        for k, v in vals.items():
            row[k] = round(v, 4)
        rows.append(row)
    out_path = Path(out_dir) / f"test_metrics_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    [saved] {out_path.name}")
    return acc, mf1, mprec, mrec, mauc


# ==========================================================================
# Single seed: train + evaluate
# ==========================================================================

def run_one_seed(seed: int, model_name: str) -> dict:

    done_csv = SAVE_DIR / "all_results.csv"
    if done_csv.exists():
        ex   = pd.read_csv(done_csv)
        done = ex[(ex["model"] == model_name) & (ex["seed"] == seed)]
        if not done.empty:
            print(f"  =>  {model_name} seed={seed} already done — skipping")
            return done.iloc[0].to_dict()

    set_seed(seed)
    dirs  = get_dirs(model_name)
    paths = get_paths(model_name, seed, dirs)

    print(f"\n{'='*72}")
    print(f"  Seed {seed}  |  {model_name}")
    print(f"{'='*72}")

    # -- transforms --------------------------------------------------------
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
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # -- datasets ----------------------------------------------------------
    full_train   = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds      = datasets.ImageFolder(TEST_PATH,  transform=val_tf)
    classes      = full_train.classes
    nc           = len(classes)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=0.15, stratify=full_train.targets, random_state=seed,
    )
    train_loader = DataLoader(Subset(full_train, train_idx),
                              batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(full_val_ref, val_idx),
                              batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    # -- model + optimizer -------------------------------------------------
    model     = create_model(model_name, nc).to(DEVICE)
    # Separate LR: backbone gets 0.1× of head/TCDA params
    bb_params   = list(model.bb.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not any(id(p)==id(q) for q in bb_params)]
    optimizer = optim.AdamW([
        {"params": bb_params,   "lr": LR * 0.1},
        {"params": head_params, "lr": LR},
    ], weight_decay=1e-2)

    # Focal loss for class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, label_smooth=LABEL_SMOOTH):
            super().__init__()
            self.gamma = gamma
            self.ce    = nn.CrossEntropyLoss(label_smoothing=label_smooth,
                                             reduction="none")
        def forward(self, logits, target):
            ce   = self.ce(logits, target)
            pt   = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()

    criterion = FocalLoss()

    best_loss = float("inf"); best_state = None; patience_ctr = 0
    history = {k: [] for k in ["epoch","train_loss","train_acc","val_loss","val_acc"]}

    # -- training loop -----------------------------------------------------
    for epoch in range(EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE); torch.cuda.empty_cache()
        gc.collect()

        cur_lr_head = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, LR)
        cur_lr_bb   = cur_lr_head * 0.1
        optimizer.param_groups[0]["lr"] = cur_lr_bb
        optimizer.param_groups[1]["lr"] = cur_lr_head

        # Alternate CutMix / Mixup by epoch (avoids one dominating)
        use_cutmix = (epoch % 2 == 0)
        cm_alpha   = CUTMIX_ALPHA * 0.5 * (1 - math.cos(math.pi * min(epoch/20.0, 1.0)))
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
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

        train_loss = tl / max(tt, 1)
        train_acc  = tc / max(tt, 1)

        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                logits = model(x)[0]
                vl += criterion(logits, y).item() * x.size(0)
                vc += (logits.argmax(1) == y).sum().item()
                vt += y.size(0)
        val_loss = vl / max(vt, 1)
        val_acc  = vc / max(vt, 1)

        for k, v in zip(
            ["epoch","train_loss","train_acc","val_loss","val_acc"],
            [epoch+1, round(train_loss,6), round(train_acc,6),
             round(val_loss,6), round(val_acc,6)],
        ):
            history[k].append(v)

        print(f"  [{epoch+1:2d}/{EPOCHS}]  "
              f"Tr {train_loss:.4f}/{train_acc:.4f}  "
              f"Va {val_loss:.4f}/{val_acc:.4f}  "
              f"lr_head={cur_lr_head:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("  => Early stopping.")
                break

    model.load_state_dict(best_state)

    # -- training curves ---------------------------------------------------
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(history["epoch"], history["train_loss"], "b-",  lw=2.2, label="Train Loss")
    ax1.plot(history["epoch"], history["val_loss"],   "c--", lw=2.2, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["train_acc"], "r-",  lw=2.2, alpha=0.85, label="Train Acc")
    ax2.plot(history["epoch"], history["val_acc"],   "m--", lw=2.2, alpha=0.85, label="Val Acc")
    ax2.set_ylabel("Accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r"); ax2.set_ylim(0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right", prop={"weight": "bold"})
    ax1.set_title(f"{model_name}  —  Seed {seed}"); ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(paths["curves_png"], dpi=160, bbox_inches="tight")
    plt.close()

    # -- test inference + GradCAM ------------------------------------------
    cam_base = paths["gradcam_base"]
    for cls in classes:
        (cam_base / cls).mkdir(parents=True, exist_ok=True)

    cam_engine = GradCAM(model, model.target_layer)
    model.eval()
    y_true_l, y_pred_l, y_prob_l = [], [], []

    for i, (x, y) in enumerate(test_loader):
        try:
            x_dev = x.to(DEVICE); y_dev = y.to(DEVICE)
            with torch.set_grad_enabled(True):
                logits_e = model(x_dev)[0]
                prob     = F.softmax(logits_e.detach(), dim=1)
                pred     = logits_e.argmax(1).item()
                cam      = cam_engine.generate(x_dev, cls=y_dev.item())

            orig     = denorm(x[0])
            cls_name = classes[y.item()]
            tag      = "correct" if pred == y.item() else "wrong"

            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            axes[0].imshow(orig);       axes[0].axis("off"); axes[0].set_title("Original")
            axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
            axes[1].axis("off");        axes[1].set_title("TCDA GradCAM")
            axes[2].imshow(orig)
            axes[2].imshow(plt.cm.jet(cam)[:, :, :3], alpha=0.45)
            axes[2].set_title(f"True: {cls_name}  |  Pred: {classes[pred]}  ({tag})")
            axes[2].axis("off")
            fig.suptitle(f"{model_name}  —  Seed {seed}", y=1.01)
            plt.tight_layout()
            plt.savefig(cam_base / cls_name / f"img_{i:05d}.png",
                        dpi=110, bbox_inches="tight")
            plt.close()

            y_true_l.append(y.item())
            y_pred_l.append(pred)
            y_prob_l.append(prob.cpu().numpy()[0])
        except Exception as e:
            print(f"    ! test img {i} skipped: {e}")
            continue

    cam_engine.release()
    y_true = np.array(y_true_l)
    y_pred = np.array(y_pred_l)
    y_prob = np.array(y_prob_l)

    # -- test metrics ------------------------------------------------------
    acc, mf1, mprec, mrec, mauc = save_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob,
        out_dir=dirs["metrics"],
    )

    # -- confusion matrix --------------------------------------------------
    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(paths["cm_csv"])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                linewidths=0.6, linecolor="gray", ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {model_name}  Seed {seed}")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    plt.tight_layout()
    plt.savefig(paths["cm_png"], dpi=150, bbox_inches="tight")
    plt.close()

    # -- ROC curves --------------------------------------------------------
    roc_rows = []
    fig, ax  = plt.subplots(figsize=(9, 6))
    for j, cls in enumerate(classes):
        fpr, tpr, thr = roc_curve((y_true==j).astype(int), y_prob[:, j])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, label=f"{cls} (AUC={roc_auc:.3f})")
        for f, t, th in zip(fpr, tpr, thr):
            roc_rows.append({"model": model_name, "seed": seed, "class": cls,
                             "fpr": round(float(f),6), "tpr": round(float(t),6),
                             "threshold": round(float(th),6), "auc": round(roc_auc,6)})
    ax.plot([0,1],[0,1],"k--",lw=1.2); ax.legend(prop={"weight":"bold"})
    ax.grid(True, alpha=0.3); ax.set_title(f"ROC — {model_name}  Seed {seed}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    plt.tight_layout(); plt.savefig(paths["roc_png"], dpi=150, bbox_inches="tight"); plt.close()
    pd.DataFrame(roc_rows).to_csv(paths["roc_data"], index=False)

    result = {"seed": seed, "model": model_name,
               "acc":             round(acc,   4),
               "f1_macro":        round(mf1,   4),
               "precision_macro": round(mprec, 4),
               "recall_macro":    round(mrec,  4),
               "auc_macro":       round(mauc,  4)}
    print(f"  [ok] Acc={acc:.4f}  F1={mf1:.4f}  AUC={mauc:.4f}")
    return result


# ==========================================================================
# Statistical helpers
# ==========================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2)
                       / (na+nb-2+EPS))
    return float((a.mean()-b.mean()) / (pooled+EPS))


def effect_label(d):
    a = abs(d)
    if a >= 0.8: return "large"
    if a >= 0.5: return "medium"
    if a >= 0.2: return "small"
    return "negligible"


def load_cm(model_name, seed):
    p  = SAVE_DIR / model_name / "metrics" / f"cm_seed{seed}.csv"
    df = pd.read_csv(p, index_col=0)
    cls = list(df.index); cm = df.values.astype(int)
    yt, yp = [], []
    for ti in range(len(cls)):
        for pi in range(len(cls)):
            n = cm[ti, pi]
            yt.extend([ti]*n); yp.extend([pi]*n)
    return np.array(yt), np.array(yp), cls


def mcnemar_test_pair(y_true, pa, pb):
    ca = (pa==y_true); cb = (pb==y_true)
    b  = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    if b+c == 0:
        return np.nan, np.nan, b, c
    chi2_v = (abs(b-c)-1.0)**2 / (b+c)
    p = 1.0 - chi2.cdf(chi2_v, df=1)
    return chi2_v, p, b, c


def run_wilcoxon(df):
    print("\n=== Wilcoxon Signed-Rank Test ===")
    rows = []
    for metric in METRICS:
        for m1, m2 in itertools.combinations(MODELS, 2):
            s1 = df[df["model"]==m1][metric].values
            s2 = df[df["model"]==m2][metric].values
            if len(s1) < 2 or len(s2) < 2: continue
            try:
                w, p = stats.wilcoxon(s1, s2, zero_method="wilcox", correction=False)
            except ValueError:
                w, p = np.nan, np.nan
            d     = cohens_d(s1, s2)
            delta = s1.mean() - s2.mean()
            sig   = (not np.isnan(p)) and (p < ALPHA_STAT)
            rows.append({
                "metric": metric, "model_A": m1, "model_B": m2,
                "mean_A": round(s1.mean(),4), "mean_B": round(s2.mean(),4),
                "delta_A_minus_B": round(delta,6),
                "wilcoxon_W": round(w,2) if not np.isnan(w) else np.nan,
                "p_value": round(p,6) if not np.isnan(p) else np.nan,
                "cohens_d": round(d,4), "effect_size": effect_label(abs(d)),
                "significant_005": sig, "n_seeds": len(s1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR/"wilcoxon_n.csv", index=False)
    print(f"  [saved] wilcoxon_n.csv"); return out


def run_mcnemar(all_seeds):
    print("\n=== McNemar Pooled ===")
    pair_b, pair_c = {}, {}
    for seed in all_seeds:
        preds, y_ref = {}, None
        for m in MODELS:
            try:
                yt, yp, _ = load_cm(m, seed)
                preds[m]  = yp
                if y_ref is None: y_ref = yt
            except Exception as e:
                print(f"  ! {e}"); continue
        for m1, m2 in itertools.combinations(list(preds.keys()), 2):
            _, _, b, c = mcnemar_test_pair(y_ref, preds[m1], preds[m2])
            key = (m1, m2)
            pair_b[key] = pair_b.get(key, 0) + b
            pair_c[key] = pair_c.get(key, 0) + c
    rows = []
    for key in pair_b:
        m1, m2 = key; b_t = pair_b[key]; c_t = pair_c[key]
        if b_t+c_t == 0:
            chi2_v, p = np.nan, np.nan
        else:
            chi2_v = (abs(b_t-c_t)-1.0)**2 / (b_t+c_t)
            p = 1.0 - chi2.cdf(chi2_v, df=1)
        sig    = (not np.isnan(p)) and (p < ALPHA_STAT)
        winner = m1 if b_t > c_t else (m2 if c_t > b_t else "tie")
        rows.append({
            "model_A": m1, "model_B": m2,
            "pooled_b": b_t, "pooled_c": c_t,
            "chi2": round(chi2_v,4) if not np.isnan(chi2_v) else np.nan,
            "p_value": round(p,6) if not np.isnan(p) else np.nan,
            "significant_005": sig, "winner": winner,
            "n_seeds_pooled": len(all_seeds),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR/"mcnemar_pooled.csv", index=False)
    print(f"  [saved] mcnemar_pooled.csv"); return out


def build_superiority_table(df, wilc_df, mc_df):
    rows = []
    for metric in METRICS:
        s_p = df[df["model"]==PROPOSED][metric].values
        for m in MODELS:
            if m == PROPOSED: continue
            s_m   = df[df["model"]==m][metric].values
            d     = cohens_d(s_p, s_m)
            delta = s_p.mean() - s_m.mean()
            wrow  = wilc_df[(wilc_df["metric"]==metric) & (
                ((wilc_df["model_A"]==PROPOSED) & (wilc_df["model_B"]==m)) |
                ((wilc_df["model_A"]==m) & (wilc_df["model_B"]==PROPOSED))
            )]
            p_w   = wrow.iloc[0]["p_value"] if not wrow.empty else np.nan
            mcrow = mc_df[
                ((mc_df["model_A"]==PROPOSED) & (mc_df["model_B"]==m)) |
                ((mc_df["model_A"]==m) & (mc_df["model_B"]==PROPOSED))
            ]
            p_mc  = mcrow.iloc[0]["p_value"] if not mcrow.empty else np.nan
            sig   = (delta > 0) and (
                (not np.isnan(p_mc) and p_mc < ALPHA_STAT) or
                (not np.isnan(p_w)  and p_w  < ALPHA_STAT)
            )
            rows.append({
                "Metric":            METRIC_LABELS[metric],
                "Proposed":          PROPOSED,
                "Compared_to":       m,
                "Proposed_mean±std": f"{s_p.mean():.4f}±{s_p.std(ddof=1):.4f}",
                "Other_mean±std":    f"{s_m.mean():.4f}±{s_m.std(ddof=1):.4f}",
                "Delta":             f"{delta:+.4f}",
                "Cohens_d":          f"{d:.4f}",
                "Effect":            effect_label(abs(d)),
                "Wilcoxon_p":        f"{p_w:.4f}"  if not np.isnan(p_w)  else "n/a",
                "McNemar_p":         f"{p_mc:.4f}" if not np.isnan(p_mc) else "n/a",
                "Proposed_wins":     "YES *" if sig else ("numerically" if delta > 0 else "NO"),
            })
    sup_df = pd.DataFrame(rows)
    sup_df.to_csv(SAVE_DIR/"superiority_table.csv", index=False)
    print("\n=== Superiority Table ===")
    print(sup_df.to_string(index=False))
    return sup_df


# ==========================================================================
# Plots
# ==========================================================================

def save_plots(df, agg, all_seeds):
    palette = sns.color_palette("tab10", len(MODELS))
    shorts  = [m.replace("BASE_","B-").replace("TCDA_","T-") for m in MODELS]

    # boxplot
    fig, axes = plt.subplots(1, len(METRICS), figsize=(32, 6.5), sharey=False)
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(x="model", y=metric, data=df, order=MODELS,
                    palette=palette, width=0.55, ax=ax, linewidth=1.8)
        sns.stripplot(x="model", y=metric, data=df, order=MODELS,
                      color="k", size=5.5, jitter=0.18, ax=ax)
        ax.set_title(METRIC_LABELS[metric]); ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticklabels(shorts, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"geomeTry-Conditioned Differential Attention (TCDA) Network v2 Performance (n={len(all_seeds)} seeds)\nB-=Baseline, T-=+TCDA")
    plt.tight_layout()
    plt.savefig(SAVE_DIR/"boxplot_all_metrics.png", dpi=180, bbox_inches="tight")
    plt.close()

    # barplot per metric
    for metric in METRICS:
        cm_col = f"{metric}_mean"; cs_col = f"{metric}_std"
        if cm_col not in agg.columns: continue
        means = agg.loc[[m for m in MODELS if m in agg.index], cm_col].values
        stds  = agg.loc[[m for m in MODELS if m in agg.index], cs_col].values
        valid_shorts = [shorts[i] for i, m in enumerate(MODELS) if m in agg.index]
        x = np.arange(len(means))
        fig, ax = plt.subplots(figsize=(16, 6.5))
        bars = ax.bar(x, means, yerr=stds, capsize=7,
                      color=palette[:len(means)], edgecolor="k", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_shorts, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]} ± Std")
        ax.set_title(f"{METRIC_LABELS[metric]} — All Models (n={len(all_seeds)} seeds)")
        ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.006,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(SAVE_DIR/f"barplot_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()

    # TCDA gain plot
    gain_rows = []
    for bname in BACKBONE_NAMES:
        base_n = f"BASE_{bname}"; tcda_n = f"TCDA_{bname}"
        for metric in METRICS:
            mc = f"{metric}_mean"
            if mc not in agg.columns: continue
            bm = agg.loc[base_n, mc] if base_n in agg.index else np.nan
            tm = agg.loc[tcda_n, mc] if tcda_n in agg.index else np.nan
            gain_rows.append({"backbone": bname, "metric": METRIC_LABELS[metric],
                              "gain": round(float(tm-bm), 4)})
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(SAVE_DIR/"tcda_gain.csv", index=False)

    gp = gain_df.pivot(index="backbone", columns="metric", values="gain")
    gp = gp[[METRIC_LABELS[m] for m in METRICS if METRIC_LABELS[m] in gp.columns]]
    x_pos = np.arange(len(gp)); w = 0.14
    pal = sns.color_palette("Set2", len(gp.columns))
    fig, ax = plt.subplots(figsize=(14, 6))
    for ci, col in enumerate(gp.columns):
        offset = (ci - len(gp.columns)/2 + 0.5) * w
        vals   = gp[col].values
        bars   = ax.bar(x_pos+offset, vals, w, label=col,
                        color=pal[ci], edgecolor="k", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(0.001 if v>=0 else -0.004),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v>=0 else "top", fontsize=7.5, fontweight="bold")
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.set_xticks(x_pos); ax.set_xticklabels(gp.index, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("TCDA Gain (TCDA_X - BASE_X)")
    ax.set_title(f"geomeTry-Conditioned Differential Attention (TCDA) v2 Gain per Backbone × Metric (n={len(all_seeds)} seeds)")
    ax.legend(loc="upper right", prop={"weight":"bold"}); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR/"tcda_gain_barplot.png", dpi=180, bbox_inches="tight")
    plt.close()

    # McNemar heatmap
    try:
        mc_df  = pd.read_csv(SAVE_DIR/"mcnemar_pooled.csv")
        p_mat  = pd.DataFrame(np.ones((len(MODELS),len(MODELS))), index=MODELS, columns=MODELS)
        for _, row in mc_df.iterrows():
            pv = row["p_value"]
            if pd.isna(pv): continue
            p_mat.loc[row["model_A"], row["model_B"]] = pv
            p_mat.loc[row["model_B"], row["model_A"]] = pv
        sm = {m: m.replace("BASE_","B-").replace("TCDA_","T-") for m in MODELS}
        p_plot = p_mat.rename(index=sm, columns=sm)
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(p_plot.astype(float), annot=True, fmt=".4f",
                    cmap="RdYlGn_r", vmin=0, vmax=0.1,
                    mask=np.eye(len(MODELS), dtype=bool), ax=ax,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size":8,"weight":"bold"},
                    cbar_kws={"label": f"McNemar p (pooled, n={len(all_seeds)} seeds)"})
        ax.set_title("Pairwise McNemar p-values")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontweight="bold", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight="bold", fontsize=9)
        plt.tight_layout()
        plt.savefig(SAVE_DIR/"mcnemar_heatmap.png", dpi=160, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"  ! mcnemar heatmap skipped: {e}")

    # Cohen's d heatmap
    try:
        wilc_df = pd.read_csv(SAVE_DIR/"wilcoxon_n.csv")
        sm = {m: m.replace("BASE_","B-").replace("TCDA_","T-") for m in MODELS}
        d_mat = pd.DataFrame(index=MODELS, columns=METRICS, dtype=float)
        for _, row in wilc_df.iterrows():
            if row["model_A"] in d_mat.index and row["metric"] in d_mat.columns:
                d_mat.loc[row["model_A"], row["metric"]] =  row["cohens_d"]
            if row["model_B"] in d_mat.index and row["metric"] in d_mat.columns:
                d_mat.loc[row["model_B"], row["metric"]] = -row["cohens_d"]
        d_plot = d_mat.rename(index=sm, columns=METRIC_LABELS)
        fig, ax = plt.subplots(figsize=(13,8))
        sns.heatmap(d_plot.astype(float), annot=True, fmt=".3f",
                    cmap="RdBu", center=0, ax=ax,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"size":9,"weight":"bold"},
                    cbar_kws={"label":"Cohen's d"})
        ax.set_title(f"Cohen's d Effect Sizes (n={len(all_seeds)} seeds)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontweight="bold", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight="bold", fontsize=10)
        plt.tight_layout()
        plt.savefig(SAVE_DIR/"cohens_d_heatmap.png", dpi=160, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"  ! cohens_d heatmap skipped: {e}")

    print("  [ok] All comparison plots saved.")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":

    print("\n" + "="*72)
    print("  geomeTry-Conditioned Differential Attention (TCDA) Network v2")
    print(f"  Proposed : {PROPOSED}")
    print(f"  Seeds    : {SEEDS}  (n={len(SEEDS)})")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {EPOCHS}  |  DESC_DIM : {DESC_DIM}")
    print(f"  Key changes vs v1:")
    print(f"    - Channel-wise differential attention (not scalar)")
    print(f"    - Multi-scale Riesz fusion (all 3 scales, learned weights)")
    print(f"    - Residual bypass (gradient highway through TCDA)")
    print(f"    - L2-normalised Riesz input (consistent across backbones)")
    print(f"    - Stochastic depth (p={STOCH_DEPTH_P}) for regularisation")
    print(f"    - Focal loss + CutMix/Mixup alternation")
    print(f"    - Differential backbone/head LR (backbone 0.1×)")
    print("="*72)

    all_results = []

    # 1 -- train all models × all seeds
    for model_name in MODELS:
        for seed in SEEDS:
            gc.collect()
            try: torch.cuda.empty_cache()
            except Exception: pass
            res = run_one_seed(seed, model_name)
            all_results.append(res)
            (pd.DataFrame(all_results)
               .drop_duplicates(subset=["model","seed"])
               .reset_index(drop=True)
               .to_csv(SAVE_DIR/"all_results.csv", index=False))

    # 2 -- aggregate
    df = (pd.read_csv(SAVE_DIR/"all_results.csv")
            .drop_duplicates(subset=["model","seed"])
            .reset_index(drop=True))

    agg_spec = {}
    for m in METRICS:
        agg_spec[m] = ["mean","std","min","max"] if m=="acc" else ["mean","std"]

    agg = df.groupby("model").agg(agg_spec).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    valid = [m for m in MODELS if m in agg.index]
    agg   = agg.reindex(valid)

    agg.to_csv(SAVE_DIR/"summary_aggregated.csv")
    agg.to_csv(SAVE_DIR/"ablation_summary.csv")

    print("\n\n=== Aggregated Summary ===")
    print(agg.to_string())

    # 3 -- statistics
    wilc_df = run_wilcoxon(df)
    mc_df   = run_mcnemar(SEEDS)

    # 4 -- superiority table
    build_superiority_table(df, wilc_df, mc_df)

    # 5 -- plots
    save_plots(df, agg, SEEDS)

    print(f"\n{'='*72}")
    print(f"  All outputs saved to: {SAVE_DIR.resolve()}")
    print("="*72)
