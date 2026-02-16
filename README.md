# FlashAttention Inference Latency Benchmark

## What This Is

A hands-on benchmark comparing LLM inference optimization techniques:
- **Standard Attention** → **FlashAttention + Fused QKV** → **+ CUDA Graphs** → **+ INT8 Quantization**


## Quick Start

### Google Colab (Easiest)
1. Upload `FlashAttention_Benchmark.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells (~5 minutes)
4. Screenshot the results table and plots

### Local Machine
```bash
pip install torch bitsandbytes matplotlib tabulate pandas
python benchmark.py
```

## What Each Optimization Does

### 1. FlashAttention + Fused QKV
**Standard attention** computes Q·Kᵀ → softmax → ×V as **separate GPU kernels**, materializing a full S×S attention matrix in slow global memory.

**FlashAttention** fuses everything into one kernel using **tiling**: processes small blocks that fit in fast on-chip SRAM, never materializes the full matrix.

**Fused QKV** replaces 3 separate [D, D] matmuls with a single [D, 3D] matmul — fewer kernel launches, better memory locality.

### 2. CUDA Graph Capture + Replay
Each GPU operation normally requires the CPU to "tell" the GPU what to do (kernel launch). With many small ops, this overhead adds up.

CUDA graphs **record** the entire forward pass once, then **replay** it as a single action. The CPU just says "replay" instead of issuing hundreds of individual commands.

**Trade-off:** Requires fixed input shapes.

### 3. Weight-Only INT8 Quantization
Stores model weights in INT8 (1 byte) instead of FP16 (2 bytes), with per-channel scale factors for accurate dequantization. Activations stay in FP16.

**Result:** ~2x less memory, faster matrix multiplications on supported hardware.

## Project Structure

```
flash_attention_benchmark/
├── benchmark.py                     # Full benchmark script (run locally)
├── FlashAttention_Benchmark.ipynb   # Colab notebook version
└── README.md                        # This file
```

## Expected Results (T4 GPU)

Typical P95 latency reductions vs Standard Attention:
- **Flash + Fused QKV:** 15–35% reduction
- **Flash + CUDA Graph:** 25–50% reduction (especially at small batch sizes)
- **Flash + INT8:** Additional 10–20% on top of Flash

Exact numbers depend on GPU, batch size, and sequence length. The benchmark sweeps all combinations so you can see the full picture.

## Key Concepts for Interviews

| Concept | What It Is | Why It Helps |
|---------|-----------|--------------|
| FlashAttention | Tiled, fused attention kernel | O(S) memory vs O(S²), better bandwidth |
| Fused QKV | Single [D,3D] matmul | Fewer kernel launches |
| CUDA Graphs | Record+replay GPU execution | Eliminates CPU→GPU launch overhead |
| INT8 Quantization | Weight-only, per-channel scales | Smaller weights = faster GEMMs |
| P95 Latency | 95th percentile response time | What users actually feel |
| CUDA Events | GPU-side timing | More accurate than wall-clock |

## How to Talk About This

**Problem → Analysis → Action → Result:**

> "Our baseline LLM inference P95 was too high. I profiled and found attention blocks and kernel launch overhead dominating. I replaced standard attention with FlashAttention via SDPA and fused the QKV projections, applied weight-only INT8 with per-channel scales, and captured the forward pass as a CUDA graph for replay. Measured with CUDA events under realistic load — P95 dropped by X% without meaningful quality degradation."
