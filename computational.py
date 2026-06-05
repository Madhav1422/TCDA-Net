# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
efficiency_benchmark.py
=======================
Standalone computational-cost profiler for TCDA v2 paper.

Reports (per model):
  - Total parameters          (M)
  - Backbone parameters       (M)
  - Head/TCDA parameters      (M)  — parameter overhead of proposed module
  - GFLOPs @ 224×224          (via fvcore)
  - Inference latency         (ms, mean ± std, CPU and GPU)
  - GPU peak memory           (MB, batch=1 and batch=16)
  - TCDA module parameters    (M)  — isolated TCDA overhead

All models: BASE_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}
            TCDA_{ResNet50, DenseNet121, MobileNetV3L, ConvNeXtTiny}

Output files (written to SAVE_DIR):
  efficiency_results.csv         — full per-model table
  efficiency_latency_detail.csv  — per-run latency samples
  efficiency_summary.txt         — human-readable summary printed + saved
"""

import gc
import math
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config (mirror SMVIB_main.py)
# --------------------------------------------------------------------------
SAVE_DIR    = Path("TCDA_2026_imp")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

gpu_id      = 5
DEVICE_GPU  = torch.device(f"cuda:{gpu_id}" if torch.cuda.device_count() > gpu_id else "cpu")
DEVICE_CPU  = torch.device("cpu")

IMG_SIZE    = 224
DESC_DIM    = 512
RIESZ_SCALES = [1.0, 2.0, 4.0]
EPS         = 1e-8
NC          = 4          # number of classes

WARMUP_RUNS  = 50        # GPU/CPU warmup before timing
TIMING_RUNS  = 200       # timed repetitions for latency
BATCH_SIZES  = [1, 16]   # for memory profiling

BACKBONE_REGISTRY = {
    "ResNet50":     "resnet50.a1_in1k",
    "DenseNet121":  "densenet121.ra_in1k",
    "MobileNetV3L": "mobilenetv3_large_100.ra_in1k",
    "ConvNeXtTiny": "convnext_tiny.in12k_ft_in1k",
}
BACKBONE_NAMES = list(BACKBONE_REGISTRY.keys())
MODELS = (
    [f"BASE_{b}" for b in BACKBONE_NAMES] +
    [f"TCDA_{b}" for b in BACKBONE_NAMES]
)

# --------------------------------------------------------------------------
# Paste the exact module definitions from SMVIB_main.py
# (kept self-contained so this script runs independently)
# --------------------------------------------------------------------------

class RieszExtractor(nn.Module):
    def __init__(self, in_channels: int, sigmas=None):
        super().__init__()
        sigmas = sigmas or RIESZ_SCALES
        self.sigmas = list(sigmas)
        k = 3
        for i, sigma in enumerate(self.sigmas):
            for d in ("x", "y"):
                kern = self._gauss_deriv(sigma, d, k)
                self.register_buffer(f"kern_{i}_{d}",
                                     kern.view(1, 1, 2*k+1, 2*k+1))

    @staticmethod
    def _gauss_deriv(sigma, direction, k):
        import numpy as np
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
        return results


class TCDA(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        self.nf      = nf
        self.n_scale = len(RIESZ_SCALES)
        self.riesz   = RieszExtractor(nf)
        self.scale_w = nn.Parameter(torch.ones(self.n_scale))
        self.proj_phi = nn.Linear(nf, DESC_DIM, bias=False)
        self.proj_A   = nn.Linear(nf, DESC_DIM, bias=False)
        self.proj_V   = nn.Linear(nf, DESC_DIM, bias=False)
        self.proj_Q1  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.proj_K1  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.proj_Q2  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.proj_K2  = nn.Linear(DESC_DIM, DESC_DIM, bias=False)
        self.lam      = nn.Parameter(torch.tensor(0.5))
        self.gamma    = nn.Parameter(torch.tensor(0.3))
        self.register_buffer("sigma_theta", torch.tensor(0.5))
        self.norm     = nn.LayerNorm(DESC_DIM)
        self.out      = nn.Linear(DESC_DIM, DESC_DIM)
        self.bypass   = nn.Linear(nf, DESC_DIM, bias=False)
        nn.init.zeros_(self.bypass.weight)

    @staticmethod
    def _beta0_proxy(phi):
        s    = torch.sin(phi).mean(dim=1)
        sign = torch.sign(s + EPS)
        zc_w = (sign[:, :, 1:] * sign[:, :, :-1] < 0).float().sum(dim=(1, 2))
        zc_h = (sign[:, 1:, :] * sign[:, :-1, :] < 0).float().sum(dim=(1, 2))
        total = (zc_w + zc_h) / (s.size(1) * s.size(2) + EPS)
        return torch.sigmoid(total).unsqueeze(1)

    def forward(self, feat):
        scales     = self.riesz(feat)
        sw         = F.softmax(self.scale_w, dim=0)
        feat_pool  = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        phi_fused  = sum(sw[i] * F.adaptive_avg_pool2d(scales[i][0], 1).flatten(1)
                         for i in range(self.n_scale))
        A_fused    = sum(sw[i] * F.adaptive_avg_pool2d(scales[i][1], 1).flatten(1)
                         for i in range(self.n_scale))
        theta_mean = F.adaptive_avg_pool2d(scales[0][2], 1).flatten(1).mean(1)
        beta0      = self._beta0_proxy(scales[0][0])
        Q_base = self.proj_phi(phi_fused)
        K_base = self.proj_A(A_fused)
        V      = self.proj_V(feat_pool)
        Q1 = self.proj_Q1(Q_base); K1 = self.proj_K1(K_base)
        Q2 = self.proj_Q2(Q_base); K2 = self.proj_K2(K_base)
        scale_factor = math.sqrt(DESC_DIM)
        a1 = (Q1 * K1) / scale_factor
        a2 = (Q2 * K2) / scale_factor
        g_orient = torch.exp(
            -theta_mean.abs()**2 / (2.0 * self.sigma_theta**2 + EPS)
        ).unsqueeze(1)
        T = (1.0 / (g_orient + EPS)).clamp(0.5, 4.0)
        lam    = self.lam.clamp(0.0, 1.0)
        A_diff = torch.sigmoid(a1 / T) - lam * torch.sigmoid(a2 / T)
        gamma     = self.gamma.clamp(0.0, 1.0)
        topo_gate = 1.0 - beta0 * gamma
        desc = 0.5 * (A_diff * topo_gate * V)
        out  = self.norm(self.out(desc) + self.bypass(feat_pool))
        return out


def _build_backbone(timm_name):
    bb = timm.create_model(timm_name, pretrained=False, features_only=True)
    nf = bb.feature_info[-1]["num_chs"]
    return bb, nf


def _last_conv(bb):
    last = None
    for _, m in bb.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found.")
    return last


class BaselineModel(nn.Module):
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
    def __init__(self, timm_name, nc):
        super().__init__()
        self.bb, nf       = _build_backbone(timm_name)
        self.target_layer = _last_conv(self.bb)
        self.tcda         = TCDA(nf)
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
        desc = self.tcda(feat)
        return self.head(self.drop(desc)),


def create_model(name, nc):
    parts = name.split("_", 1)
    kind, bname = parts
    timm_name = BACKBONE_REGISTRY[bname]
    if kind == "BASE":
        return BaselineModel(timm_name, nc)
    if kind == "TCDA":
        return TCDAModel(timm_name, nc)
    raise ValueError(f"Unknown kind: {kind}")


# --------------------------------------------------------------------------
# Parameter counting
# --------------------------------------------------------------------------

def count_params(model):
    """Returns total, backbone, and head/TCDA parameter counts (in Millions)."""
    total  = sum(p.numel() for p in model.parameters())
    bb     = sum(p.numel() for p in model.bb.parameters())
    head   = total - bb
    return total / 1e6, bb / 1e6, head / 1e6


def count_tcda_params(model):
    """Isolated TCDA module parameter count (M). 0 for baseline."""
    if hasattr(model, "tcda"):
        return sum(p.numel() for p in model.tcda.parameters()) / 1e6
    return 0.0


# --------------------------------------------------------------------------
# FLOPs via fvcore
# --------------------------------------------------------------------------

def compute_gflops(model, device, img_size=IMG_SIZE):
    """
    Uses fvcore FlopCountAnalysis. Falls back to thop if fvcore is absent.
    Returns GFLOPs (float).
    """
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    model.eval()

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_settings(False)   # silence warnings
        flops.uncalled_modules_settings(False)
        return flops.total() / 1e9

    except ImportError:
        pass

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9   # MACs ? FLOPs

    except ImportError:
        pass

    # Manual estimate via hooks (fallback — less accurate)
    total_flops = [0]

    def conv_hook(m, inp, out):
        b, c_out, h_out, w_out = out.shape
        k_h, k_w = m.kernel_size
        c_in     = m.in_channels // m.groups
        total_flops[0] += 2 * b * c_out * h_out * w_out * c_in * k_h * k_w

    def linear_hook(m, inp, out):
        total_flops[0] += 2 * inp[0].numel() * m.out_features

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return total_flops[0] / 1e9


# --------------------------------------------------------------------------
# Latency measurement
# --------------------------------------------------------------------------

def measure_latency_gpu(model, device, img_size=IMG_SIZE,
                         warmup=WARMUP_RUNS, n=TIMING_RUNS):
    """
    Single-image (batch=1) latency on GPU, measured with CUDA events.
    Returns (mean_ms, std_ms, all_times list).
    """
    model.eval().to(device)
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(n):
                starter.record()
                _ = model(dummy)
                ender.record()
                torch.cuda.synchronize(device)
                times.append(starter.elapsed_time(ender))
    else:
        with torch.no_grad():
            for _ in range(n):
                t0 = time.perf_counter()
                _ = model(dummy)
                times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return float(times.mean()), float(times.std()), times.tolist()


def measure_latency_cpu(model, img_size=IMG_SIZE,
                         warmup=WARMUP_RUNS, n=TIMING_RUNS):
    """
    Single-image latency on CPU.
    """
    model.eval().to(DEVICE_CPU)
    dummy = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return float(times.mean()), float(times.std())


# --------------------------------------------------------------------------
# GPU memory profiling
# --------------------------------------------------------------------------

def measure_gpu_memory(model, device, batch_sizes=BATCH_SIZES, img_size=IMG_SIZE):
    """
    Peak GPU memory (MB) for each batch size via torch.cuda.max_memory_allocated.
    Returns dict {batch_size: peak_mb}.
    """
    if device.type != "cuda":
        return {b: float("nan") for b in batch_sizes}
    results = {}
    model.eval().to(device)
    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        dummy = torch.zeros(bs, 3, img_size, img_size, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        results[bs] = round(peak, 2)
        del dummy
        torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------
# Main benchmark loop
# --------------------------------------------------------------------------

def run_benchmark():
    print("\n" + "="*72)
    print("  TCDA v2 — Computational Cost Benchmark")
    print(f"  GPU : {DEVICE_GPU}  |  IMG_SIZE={IMG_SIZE}  "
          f"WARMUP={WARMUP_RUNS}  TIMING_RUNS={TIMING_RUNS}")
    print("="*72)

    # Check for fvcore/thop
    for lib in ("fvcore", "thop"):
        try:
            __import__(lib)
            print(f"  [ok] {lib} found — FLOPs via {lib}")
            break
        except ImportError:
            print(f"  [warn] {lib} not found")

    rows         = []
    latency_rows = []

    for model_name in MODELS:
        print(f"\n{'-'*60}")
        print(f"  Profiling: {model_name}")
        print(f"{'-'*60}")

        gc.collect()
        if DEVICE_GPU.type == "cuda":
            torch.cuda.empty_cache()

        # Build on CPU first for param/flop counting
        model_cpu = create_model(model_name, NC)
        model_cpu.eval()

        # -- Parameters ------------------------------------------------
        total_M, bb_M, head_M = count_params(model_cpu)
        tcda_M                = count_tcda_params(model_cpu)
        print(f"  Params — total={total_M:.3f}M  backbone={bb_M:.3f}M  "
              f"head/TCDA={head_M:.3f}M  TCDA_only={tcda_M:.3f}M")

        # -- FLOPs -----------------------------------------------------
        try:
            gflops = compute_gflops(model_cpu, DEVICE_CPU)
            print(f"  GFLOPs = {gflops:.4f}")
        except Exception as e:
            gflops = float("nan")
            print(f"  GFLOPs = ERROR ({e})")

        # -- GPU latency -----------------------------------------------
        if DEVICE_GPU.type == "cuda":
            model_gpu = create_model(model_name, NC).to(DEVICE_GPU)
            model_gpu.eval()
            try:
                lat_gpu_mean, lat_gpu_std, lat_samples = measure_latency_gpu(
                    model_gpu, DEVICE_GPU)
                print(f"  GPU latency = {lat_gpu_mean:.3f} ± {lat_gpu_std:.3f} ms  "
                      f"(batch=1, n={TIMING_RUNS})")
                for s in lat_samples:
                    latency_rows.append({"model": model_name, "device": "GPU",
                                         "latency_ms": round(s, 4)})
            except Exception as e:
                lat_gpu_mean = lat_gpu_std = float("nan")
                print(f"  GPU latency = ERROR ({e})")
            # Memory
            mem = measure_gpu_memory(model_gpu, DEVICE_GPU, BATCH_SIZES)
            print(f"  GPU mem (peak) — " +
                  "  ".join(f"batch={b}: {mem[b]:.1f}MB" for b in BATCH_SIZES))
            del model_gpu
            torch.cuda.empty_cache()
        else:
            lat_gpu_mean = lat_gpu_std = float("nan")
            mem = {b: float("nan") for b in BATCH_SIZES}
            print("  GPU not available — skipping GPU latency & memory")

        # -- CPU latency -----------------------------------------------
        try:
            lat_cpu_mean, lat_cpu_std = measure_latency_cpu(model_cpu)
            print(f"  CPU latency = {lat_cpu_mean:.3f} ± {lat_cpu_std:.3f} ms  "
                  f"(batch=1, n={TIMING_RUNS})")
        except Exception as e:
            lat_cpu_mean = lat_cpu_std = float("nan")
            print(f"  CPU latency = ERROR ({e})")

        # Latency rows for CPU
        # (we only store mean/std for CPU to keep file compact)
        latency_rows.append({"model": model_name, "device": "CPU_mean",
                              "latency_ms": round(lat_cpu_mean, 4)})

        del model_cpu
        gc.collect()

        rows.append({
            "model":              model_name,
            "kind":               "TCDA" if model_name.startswith("TCDA_") else "BASE",
            "backbone":           model_name.split("_", 1)[1],
            "total_params_M":     round(total_M, 4),
            "backbone_params_M":  round(bb_M,    4),
            "head_tcda_params_M": round(head_M,  4),
            "tcda_only_params_M": round(tcda_M,  4),
            "gflops":             round(gflops,  4),
            "gpu_lat_mean_ms":    round(lat_gpu_mean, 4) if not math.isnan(lat_gpu_mean) else float("nan"),
            "gpu_lat_std_ms":     round(lat_gpu_std,  4) if not math.isnan(lat_gpu_std)  else float("nan"),
            "cpu_lat_mean_ms":    round(lat_cpu_mean, 4) if not math.isnan(lat_cpu_mean) else float("nan"),
            "cpu_lat_std_ms":     round(lat_cpu_std,  4) if not math.isnan(lat_cpu_std)  else float("nan"),
            **{f"gpu_mem_batch{b}_MB": mem[b] for b in BATCH_SIZES},
        })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(SAVE_DIR / "efficiency_results.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(SAVE_DIR / "efficiency_latency_detail.csv",
                                       index=False)
    print(f"\n  [saved] efficiency_results.csv")
    print(f"  [saved] efficiency_latency_detail.csv")

    # ------------------------------------------------------------------
    # Build human-readable summary with TCDA overhead
    # ------------------------------------------------------------------
    lines = []
    lines.append("\n" + "="*72)
    lines.append("  COMPUTATIONAL COST SUMMARY — TCDA v2  (batch=1, 224×224)")
    lines.append("="*72)

    # Full table
    col_fmt = (
        f"{'Model':<22} {'Params(M)':>10} {'BB(M)':>8} {'H+T(M)':>8} "
        f"{'TCDA(M)':>8} {'GFLOPs':>8} "
        f"{'GPU lat(ms)':>13} {'CPU lat(ms)':>13} "
        f"{'MemB1(MB)':>10} {'MemB16(MB)':>11}"
    )
    lines.append(col_fmt)
    lines.append("-"*120)

    for _, r in df.iterrows():
        gpu_lat_str = (f"{r['gpu_lat_mean_ms']:.2f}±{r['gpu_lat_std_ms']:.2f}"
                       if not math.isnan(r["gpu_lat_mean_ms"]) else "  n/a")
        cpu_lat_str = (f"{r['cpu_lat_mean_ms']:.1f}±{r['cpu_lat_std_ms']:.1f}"
                       if not math.isnan(r["cpu_lat_mean_ms"]) else "  n/a")
        mem1  = f"{r[f'gpu_mem_batch1_MB']:.1f}"  if not math.isnan(r.get(f"gpu_mem_batch1_MB", float("nan")))  else "n/a"
        mem16 = f"{r[f'gpu_mem_batch16_MB']:.1f}" if not math.isnan(r.get(f"gpu_mem_batch16_MB", float("nan"))) else "n/a"
        line = (
            f"{r['model']:<22} {r['total_params_M']:>10.3f} {r['backbone_params_M']:>8.3f} "
            f"{r['head_tcda_params_M']:>8.3f} {r['tcda_only_params_M']:>8.3f} "
            f"{r['gflops']:>8.4f} "
            f"{gpu_lat_str:>13} {cpu_lat_str:>13} "
            f"{mem1:>10} {mem16:>11}"
        )
        lines.append(line)

    lines.append("-"*120)
    lines.append("  BB=Backbone params  H+T=Head+TCDA params  TCDA=isolated TCDA module only")

    # Per-backbone overhead table
    lines.append("\n" + "-"*72)
    lines.append("  TCDA PARAMETER & FLOP OVERHEAD PER BACKBONE")
    lines.append("-"*72)
    lines.append(f"{'Backbone':<18} {'?FLOP(G)':>10} {'?Params(M)':>12} "
                 f"{'?GPUlat(ms)':>14} {'?CPUlat(ms)':>13}")
    lines.append("-"*72)
    for bname in BACKBONE_NAMES:
        base_row = df[df["model"] == f"BASE_{bname}"].iloc[0]
        tcda_row = df[df["model"] == f"TCDA_{bname}"].iloc[0]
        d_flop = tcda_row["gflops"] - base_row["gflops"]
        d_par  = tcda_row["total_params_M"] - base_row["total_params_M"]
        d_gpu  = (tcda_row["gpu_lat_mean_ms"] - base_row["gpu_lat_mean_ms"]
                  if not math.isnan(tcda_row["gpu_lat_mean_ms"]) else float("nan"))
        d_cpu  = (tcda_row["cpu_lat_mean_ms"] - base_row["cpu_lat_mean_ms"]
                  if not math.isnan(tcda_row["cpu_lat_mean_ms"]) else float("nan"))
        gpu_str = f"{d_gpu:+.2f}" if not math.isnan(d_gpu) else "n/a"
        cpu_str = f"{d_cpu:+.2f}" if not math.isnan(d_cpu) else "n/a"
        lines.append(f"{bname:<18} {d_flop:>+10.4f} {d_par:>+12.3f} "
                     f"{gpu_str:>14} {cpu_str:>13}")

    lines.append("-"*72)

    summary = "\n".join(lines)
    print(summary)
    with open(SAVE_DIR / "efficiency_summary.txt", "w") as f:
        f.write(summary)
    print(f"\n  [saved] efficiency_summary.txt")
    print(f"\n  All efficiency outputs ? {SAVE_DIR.resolve()}")
    print("="*72)

    return df


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Install missing profiler if needed
    import subprocess, sys
    for pkg in ("fvcore",):
        try:
            __import__(pkg)
        except ImportError:
            print(f"  Installing {pkg} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg,
                            "--quiet", "--break-system-packages"])

    run_benchmark()
