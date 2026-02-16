"""
FlashAttention Inference Latency Benchmark
==========================================
Compares standard attention vs FlashAttention-style fused kernels,
weight-only INT8 quantization, and CUDA graph replay for LLM inference.

This script demonstrates the exact techniques from:
  "LLM inference latency reduction of 44% using optimized attention kernels,
   quantized tensors, and GPU graph fusion"

Run on: Google Colab (T4 GPU free tier), or any CUDA-capable machine.
Requirements: pip install torch bitsandbytes matplotlib pandas tabulate

Author: Raasikh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from contextlib import contextmanager

# ─────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """All knobs in one place."""
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 4          # stack multiple blocks for realistic model
    mlp_ratio: int = 4
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    warmup_iters: int = 20
    bench_iters: int = 100
    use_amp: bool = True       # FP16 autocast
    device: str = "cuda"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


# ─────────────────────────────────────────────────────────────
# 1. Model Definitions
# ─────────────────────────────────────────────────────────────

class StandardAttentionBlock(nn.Module):
    """
    Vanilla transformer block using nn.MultiheadAttention.
    This is the BASELINE — separate Q, K, V projections, standard
    scaled dot-product attention with separate softmax kernel.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x


class FlashAttentionBlock(nn.Module):
    """
    Optimized transformer block with:
      1. Fused QKV projection (single matmul instead of 3)
      2. F.scaled_dot_product_attention with FlashAttention backend
      3. Proper memory layout for flash kernel compatibility

    This is the KEY optimization — FlashAttention uses tiling and
    recomputation to avoid materializing the full S×S attention matrix,
    reducing memory from O(S²) to O(S) and improving speed via better
    memory access patterns.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV: one [D, 3D] matmul instead of three [D, D] matmuls
        # This reduces kernel launch overhead and improves memory locality
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.ln1 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Fused QKV projection
        qkv = self.qkv(x)                                       # [B, S, 3D]
        qkv = qkv.view(B, S, 3, self.n_heads, self.head_dim)   # [B, S, 3, H, Hd]
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # [3, B, H, S, Hd]
        q, k, v = qkv.unbind(0)                                 # each [B, H, S, Hd]

        # FlashAttention-style fused kernel via PyTorch SDPA
        # is_causal=True for autoregressive LLM inference
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )  # [B, H, S, Hd]

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = self.ln1(x + self.proj(attn_out))
        x = self.ln2(x + self.mlp(x))
        return x


def build_model(block_class, cfg: BenchmarkConfig) -> nn.Module:
    """Stack N transformer blocks into a simple model."""
    blocks = nn.Sequential(*[
        block_class(cfg.d_model, cfg.n_heads, cfg.mlp_ratio)
        for _ in range(cfg.n_layers)
    ])
    return blocks


# ─────────────────────────────────────────────────────────────
# 2. Quantization Wrapper (Weight-Only INT8)
# ─────────────────────────────────────────────────────────────

def try_quantize_model(model: nn.Module) -> tuple:
    """
    Attempt weight-only INT8 quantization using bitsandbytes.
    Falls back gracefully if bitsandbytes is not available.

    In production, this is what "quantized tensors" means:
    - Weights are stored in INT8 (1 byte) instead of FP16 (2 bytes)
    - Per-channel scales map INT8 back to approximate FP16 values
    - Activations stay in FP16 — only weights are quantized
    - Result: ~2x less memory, faster GEMMs on supported hardware
    """
    try:
        import bitsandbytes as bnb

        class QuantizedFlashBlock(nn.Module):
            def __init__(self, d_model, n_heads, mlp_ratio=4):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads

                # INT8 quantized linear layers
                self.qkv = bnb.nn.Linear8bitLt(d_model, 3 * d_model, bias=False)
                self.proj = bnb.nn.Linear8bitLt(d_model, d_model, bias=False)

                self.ln1 = nn.LayerNorm(d_model)
                hidden = d_model * mlp_ratio
                self.mlp = nn.Sequential(
                    bnb.nn.Linear8bitLt(d_model, hidden),
                    nn.GELU(),
                    bnb.nn.Linear8bitLt(hidden, d_model),
                )
                self.ln2 = nn.LayerNorm(d_model)

            def forward(self, x):
                B, S, D = x.shape
                qkv = self.qkv(x)
                qkv = qkv.view(B, S, 3, self.n_heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=True
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
                x = self.ln1(x + self.proj(attn_out))
                x = self.ln2(x + self.mlp(x))
                return x

        return QuantizedFlashBlock, True
    except ImportError:
        print("⚠️  bitsandbytes not found — skipping INT8 quantization benchmark")
        print("   Install with: pip install bitsandbytes")
        return None, False


# ─────────────────────────────────────────────────────────────
# 3. CUDA Graph Capture + Replay
# ─────────────────────────────────────────────────────────────

class CUDAGraphRunner:
    """
    Wraps a model for CUDA graph capture and replay.

    How CUDA graphs work:
    1. RECORD phase: Run the model once while GPU records every kernel launch
    2. REPLAY phase: Instead of CPU telling GPU "launch kernel A, then B, then C..."
       the GPU replays the entire recorded sequence in one shot

    Why this helps:
    - Eliminates CPU→GPU kernel launch overhead (microseconds per kernel add up)
    - Especially impactful when batch size is small but model has many ops
    - Requires FIXED input shapes (can't change B or S after capture)

    Limitation: shapes must be static. In production, you'd capture one graph
    per (batch_size, seq_len) bucket.
    """
    def __init__(self, model: nn.Module, example_input: torch.Tensor,
                 use_amp: bool = True):
        self.model = model
        self.use_amp = use_amp
        self.stream = torch.cuda.Stream()
        self.graph = torch.cuda.CUDAGraph()

        # Static buffers — CUDA graph operates on fixed memory addresses
        self.static_input = example_input.clone()
        self.static_output = torch.empty_like(example_input)

        # Warmup: ensure all kernels are JIT-compiled / initialized
        for _ in range(5):
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        _ = model(self.static_input)
                else:
                    _ = model(self.static_input)
        torch.cuda.synchronize()

        # Capture the graph
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        self.static_output = model(self.static_input)
                else:
                    self.static_output = model(self.static_input)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Copy new data into the static buffer, replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()


# ─────────────────────────────────────────────────────────────
# 4. Benchmarking Engine
# ─────────────────────────────────────────────────────────────

@contextmanager
def cuda_timer():
    """Precise GPU timing using CUDA events (not wall clock)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    # Store elapsed time in ms on the context manager
    cuda_timer.elapsed_ms = start.elapsed_time(end)


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    cfg: BenchmarkConfig,
    use_cuda_graph: bool = False,
    label: str = "",
) -> Dict:
    """
    Run warmup + timed iterations, collect p50/p95/p99 latencies.

    Returns dict with timing stats in milliseconds.
    """
    device = cfg.device
    x = torch.randn(batch_size, seq_len, cfg.d_model, device=device)

    # Optionally wrap in CUDA graph
    runner = model
    if use_cuda_graph:
        try:
            runner = CUDAGraphRunner(model, x, use_amp=cfg.use_amp)
        except Exception as e:
            print(f"  ⚠️  CUDA graph capture failed: {e}")
            return None

    # Warmup
    with torch.no_grad():
        for _ in range(cfg.warmup_iters):
            if cfg.use_amp and not use_cuda_graph:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = runner(x)
            else:
                _ = runner(x)
    torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(cfg.bench_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            if cfg.use_amp and not use_cuda_graph:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = runner(x)
            else:
                _ = runner(x)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))  # ms

    latencies.sort()
    n = len(latencies)

    return {
        "label": label,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "mean_ms": sum(latencies) / n,
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "std_ms": (sum((l - sum(latencies)/n)**2 for l in latencies) / n) ** 0.5,
    }


# ─────────────────────────────────────────────────────────────
# 5. Main Benchmark Runner
# ─────────────────────────────────────────────────────────────

def print_header():
    print("=" * 70)
    print("  FlashAttention Inference Latency Benchmark")
    print("  Comparing: Standard Attention → Flash + Fused QKV")
    print("             → + INT8 Quantization → + CUDA Graph Replay")
    print("=" * 70)


def get_gpu_info() -> dict:
    """Collect GPU metadata for the report."""
    if not torch.cuda.is_available():
        return {"error": "No CUDA GPU available"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_mem / 1e9, 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "cudnn_version": str(torch.backends.cudnn.version()),
    }


def run_full_benchmark(cfg: Optional[BenchmarkConfig] = None):
    """Execute the complete benchmark suite."""
    if cfg is None:
        cfg = BenchmarkConfig()

    print_header()

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n❌ No CUDA GPU detected. This benchmark requires a GPU.")
        print("   Run on Google Colab with a T4 GPU (free tier).")
        sys.exit(1)

    gpu_info = get_gpu_info()
    print(f"\n🖥️  GPU: {gpu_info['name']}")
    print(f"   Memory: {gpu_info['total_memory_gb']} GB")
    print(f"   CUDA: {gpu_info['cuda_version']}  |  PyTorch: {gpu_info['pytorch_version']}")
    print(f"   Compute Capability: {gpu_info['compute_capability']}")

    # Check FlashAttention support
    flash_available = hasattr(torch.backends.cuda, 'sdp_kernel')
    print(f"\n   FlashAttention SDPA: {'✅ Available' if flash_available else '⚠️  Not available (PyTorch < 2.0)'}")

    # Check bitsandbytes
    QuantizedBlock, has_bnb = try_quantize_model(None)

    print(f"\n📋 Config: {cfg.n_layers} layers, d_model={cfg.d_model}, "
          f"n_heads={cfg.n_heads}, AMP={'on' if cfg.use_amp else 'off'}")
    print(f"   Warmup: {cfg.warmup_iters} iters | Bench: {cfg.bench_iters} iters")

    all_results = []

    # ── Sweep over batch sizes and sequence lengths ──
    for batch_size in cfg.batch_sizes:
        for seq_len in cfg.seq_lengths:
            print(f"\n{'─' * 60}")
            print(f"  Batch={batch_size}, SeqLen={seq_len}")
            print(f"{'─' * 60}")

            # 1) Standard Attention (baseline)
            print("  [1/4] Standard Attention (baseline)...", end=" ", flush=True)
            model_std = build_model(StandardAttentionBlock, cfg).to(cfg.device).eval()
            result = benchmark_model(model_std, batch_size, seq_len, cfg,
                                     label="Standard Attention")
            if result:
                all_results.append(result)
                baseline_p95 = result["p95_ms"]
                print(f"p95={result['p95_ms']:.2f} ms")
            del model_std
            torch.cuda.empty_cache()
            gc.collect()

            # 2) FlashAttention + Fused QKV
            print("  [2/4] FlashAttention + Fused QKV...", end=" ", flush=True)
            model_flash = build_model(FlashAttentionBlock, cfg).to(cfg.device).eval()
            result = benchmark_model(model_flash, batch_size, seq_len, cfg,
                                     label="Flash + Fused QKV")
            if result:
                all_results.append(result)
                pct = ((baseline_p95 - result["p95_ms"]) / baseline_p95) * 100
                print(f"p95={result['p95_ms']:.2f} ms  ({pct:+.1f}% vs baseline)")

            # 3) Flash + CUDA Graph
            print("  [3/4] Flash + CUDA Graph...", end=" ", flush=True)
            try:
                result = benchmark_model(model_flash, batch_size, seq_len, cfg,
                                         use_cuda_graph=True,
                                         label="Flash + CUDA Graph")
                if result:
                    all_results.append(result)
                    pct = ((baseline_p95 - result["p95_ms"]) / baseline_p95) * 100
                    print(f"p95={result['p95_ms']:.2f} ms  ({pct:+.1f}% vs baseline)")
            except Exception as e:
                print(f"⚠️  Skipped ({e})")
            del model_flash
            torch.cuda.empty_cache()
            gc.collect()

            # 4) Flash + INT8 Quantization (if bitsandbytes available)
            if has_bnb and QuantizedBlock is not None:
                print("  [4/4] Flash + INT8 Quantized...", end=" ", flush=True)
                try:
                    model_q = nn.Sequential(*[
                        QuantizedBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio)
                        for _ in range(cfg.n_layers)
                    ]).to(cfg.device).eval()
                    result = benchmark_model(model_q, batch_size, seq_len, cfg,
                                             label="Flash + INT8")
                    if result:
                        all_results.append(result)
                        pct = ((baseline_p95 - result["p95_ms"]) / baseline_p95) * 100
                        print(f"p95={result['p95_ms']:.2f} ms  ({pct:+.1f}% vs baseline)")
                    del model_q
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"⚠️  Skipped ({e})")
            else:
                print("  [4/4] Flash + INT8 Quantized... ⏭️  Skipped (no bitsandbytes)")

    return all_results, gpu_info, cfg


# ─────────────────────────────────────────────────────────────
# 6. Results Table + Summary
# ─────────────────────────────────────────────────────────────

def print_results_table(results: List[Dict]):
    """Print a clean results table."""
    try:
        from tabulate import tabulate
        headers = ["Method", "B", "S", "P50 (ms)", "P95 (ms)", "P99 (ms)",
                    "Mean (ms)", "Std (ms)"]
        rows = []
        for r in results:
            rows.append([
                r["label"], r["batch_size"], r["seq_len"],
                f"{r['p50_ms']:.2f}", f"{r['p95_ms']:.2f}", f"{r['p99_ms']:.2f}",
                f"{r['mean_ms']:.2f}", f"{r['std_ms']:.2f}",
            ])
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        # Fallback without tabulate
        print(f"\n{'Method':<25} {'B':>3} {'S':>5} {'P50':>8} {'P95':>8} {'P99':>8} {'Mean':>8}")
        print("-" * 75)
        for r in results:
            print(f"{r['label']:<25} {r['batch_size']:>3} {r['seq_len']:>5} "
                  f"{r['p50_ms']:>7.2f}  {r['p95_ms']:>7.2f}  {r['p99_ms']:>7.2f}  "
                  f"{r['mean_ms']:>7.2f}")


def compute_speedup_summary(results: List[Dict]) -> List[Dict]:
    """Compute speedup of each method vs Standard Attention baseline."""
    summary = []
    # Group by (batch_size, seq_len)
    configs = set((r["batch_size"], r["seq_len"]) for r in results)
    for bs, sl in sorted(configs):
        group = [r for r in results if r["batch_size"] == bs and r["seq_len"] == sl]
        baseline = next((r for r in group if r["label"] == "Standard Attention"), None)
        if baseline is None:
            continue
        for r in group:
            if r["label"] == "Standard Attention":
                continue
            reduction = ((baseline["p95_ms"] - r["p95_ms"]) / baseline["p95_ms"]) * 100
            summary.append({
                "method": r["label"],
                "batch_size": bs,
                "seq_len": sl,
                "baseline_p95": baseline["p95_ms"],
                "optimized_p95": r["p95_ms"],
                "reduction_pct": reduction,
            })
    return summary


def print_speedup_summary(summary: List[Dict]):
    """Print the speedup comparison."""
    print("\n" + "=" * 70)
    print("  SPEEDUP SUMMARY (P95 latency reduction vs Standard Attention)")
    print("=" * 70)

    for s in summary:
        bar = "█" * max(1, int(s["reduction_pct"] / 2))
        print(f"  B={s['batch_size']:>2}, S={s['seq_len']:>4} | "
              f"{s['method']:<22} | "
              f"{s['baseline_p95']:>7.2f} → {s['optimized_p95']:>7.2f} ms  "
              f"({s['reduction_pct']:>+6.1f}%) {bar}")

    # Overall average
    if summary:
        avg_reduction = sum(s["reduction_pct"] for s in summary) / len(summary)
        best = max(summary, key=lambda s: s["reduction_pct"])
        print(f"\n  📊 Average P95 reduction: {avg_reduction:.1f}%")
        print(f"  🏆 Best: {best['method']} at B={best['batch_size']}, "
              f"S={best['seq_len']} → {best['reduction_pct']:.1f}% reduction")


# ─────────────────────────────────────────────────────────────
# 7. Visualization (Matplotlib)
# ─────────────────────────────────────────────────────────────

def generate_plots(results: List[Dict], gpu_info: dict, save_dir: str = "."):
    """Generate publication-quality benchmark plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("⚠️  matplotlib not found — skipping plots. Install: pip install matplotlib")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"FlashAttention Inference Latency Benchmark\n"
        f"GPU: {gpu_info.get('name', 'Unknown')} | "
        f"PyTorch {gpu_info.get('pytorch_version', '?')} | "
        f"CUDA {gpu_info.get('cuda_version', '?')}",
        fontsize=13, fontweight="bold"
    )

    colors = {
        "Standard Attention": "#e74c3c",
        "Flash + Fused QKV": "#3498db",
        "Flash + CUDA Graph": "#2ecc71",
        "Flash + INT8": "#f39c12",
    }

    # ── Plot 1: P95 Latency vs Sequence Length (fixed batch=8 or first available) ──
    ax1 = axes[0]
    target_bs = 8 if 8 in set(r["batch_size"] for r in results) else \
                results[0]["batch_size"] if results else 1

    for label in colors:
        data = [(r["seq_len"], r["p95_ms"]) for r in results
                if r["label"] == label and r["batch_size"] == target_bs]
        if data:
            data.sort()
            xs, ys = zip(*data)
            ax1.plot(xs, ys, "o-", color=colors[label], label=label,
                     linewidth=2, markersize=6)

    ax1.set_xlabel("Sequence Length", fontsize=11)
    ax1.set_ylabel("P95 Latency (ms)", fontsize=11)
    ax1.set_title(f"P95 Latency vs Seq Length (batch={target_bs})", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Speedup % vs Baseline ──
    ax2 = axes[1]
    summary = compute_speedup_summary(results)

    for label in ["Flash + Fused QKV", "Flash + CUDA Graph", "Flash + INT8"]:
        data = [(s["seq_len"], s["reduction_pct"]) for s in summary
                if s["method"] == label and s["batch_size"] == target_bs]
        if data:
            data.sort()
            xs, ys = zip(*data)
            ax2.bar([x + list(colors.keys()).index(label) * 15 - 15 for x in xs],
                    ys, width=12, color=colors[label], label=label, alpha=0.85)

    ax2.set_xlabel("Sequence Length", fontsize=11)
    ax2.set_ylabel("P95 Latency Reduction (%)", fontsize=11)
    ax2.set_title(f"Speedup vs Standard Attention (batch={target_bs})", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "benchmark_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Plot saved to: {plot_path}")
    plt.close()
    return plot_path


# ─────────────────────────────────────────────────────────────
# 8. Save Results to JSON
# ─────────────────────────────────────────────────────────────

def save_results(results: List[Dict], gpu_info: dict, cfg: BenchmarkConfig,
                 save_dir: str = "."):
    """Save all results to JSON for reproducibility."""
    output = {
        "gpu_info": gpu_info,
        "config": asdict(cfg),
        "results": results,
        "speedup_summary": compute_speedup_summary(results),
    }
    path = os.path.join(save_dir, "benchmark_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"💾 Results saved to: {path}")


# ─────────────────────────────────────────────────────────────
# 9. Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # You can customize the config here
    cfg = BenchmarkConfig(
        d_model=1024,
        n_heads=16,
        n_layers=4,
        batch_sizes=[1, 4, 8],
        seq_lengths=[128, 256, 512, 1024],
        warmup_iters=20,
        bench_iters=100,
        use_amp=True,
    )

    results, gpu_info, cfg = run_full_benchmark(cfg)

    print_results_table(results)

    summary = compute_speedup_summary(results)
    print_speedup_summary(summary)

    generate_plots(results, gpu_info)
    save_results(results, gpu_info, cfg)

    print("\n✅ Benchmark complete!")
    print("   You can now screenshot the output for your portfolio.")
    print("   Results + plot saved in current directory.")
