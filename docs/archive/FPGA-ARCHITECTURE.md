# FPGA Training Architecture — NeuroFabric BF16W Micro-Transformer

## Target Device

**Xilinx Zynq UltraScale+ ZCU102** (prototype / Paper 2 target)
- BRAM: 32.1 Mb (~4.0 MB)
- DSP slices: 2,520 (BF16 FMA capable)
- Clock: up to 300 MHz

---

## Memory Layout (Training, 334K params)

```
┌─────────────────────────────────────────────┐  ZCU102 BRAM: ~4.0 MB
│  Weights (BF16)          668 KB             │
│  Adam m moments (FP32)  1,336 KB            │
│  Adam v moments (FP32)  1,336 KB            │
│─────────────────────────────────────────────│
│  Subtotal (persistent)  3,340 KB            │
│─────────────────────────────────────────────│
│  Active layer buffer     ~180 KB            │  reused per layer
│  (activations, 1 layer at a time)           │
│─────────────────────────────────────────────│
│  TOTAL                  ~3,520 KB  ✅ fits  │
└─────────────────────────────────────────────┘
```

**Key**: activation recomputation during backward pass — each layer is
recomputed from input when needed, rather than storing all 4 layers
simultaneously. Standard technique, no algorithm change required.

---

## Training Dataflow (batch = 1, online Adam)

```
  Token input (int)
        │
        ▼
┌───────────────┐
│  Embedding    │  lookup → BF16 vector [1×88]
│  + Pos Enc    │
└───────┬───────┘
        │
        ▼  (×4 layers, sequential)
┌───────────────────────────────────────────┐
│  Transformer Layer                        │
│                                           │
│   ┌─────────┐   ┌──────────────────────┐  │
│   │ LayerNorm│   │  Multi-Head Attn     │  │
│   │  (FP32) │──▶│  Wq Wk Wv Wo        │  │
│   └─────────┘   │  (BF16 weights)      │  │
│                 └──────────┬───────────┘  │
│                            │              │
│                 ┌──────────▼───────────┐  │
│                 │  Feed-Forward        │  │
│                 │  GeLU activation     │  │
│                 │  (BF16 weights)      │  │
│                 └──────────┬───────────┘  │
└────────────────────────────┼──────────────┘
        │ (×4 repeat)        │
        ▼                    │
┌───────────────┐            │
│  LayerNorm    │            │
│  + Proj (tied)│            │
└───────┬───────┘            │
        ▼                    │
   Logits [1×1501]           │
        │                    │
        ▼                    │
   Cross-Entropy Loss        │
        │                    │
        ▼                    │
   ┌─────────────────────────┴──────────────┐
   │  Backward Pass                         │
   │  (layer-by-layer, recompute forward    │
   │   activations per layer — no full      │
   │   activation store needed)             │
   └────────────────┬───────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────────┐
   │  Adam Update (FP32 moments)            │
   │  w_bf16 ← round(w_fp32 - lr * step)   │
   └────────────────────────────────────────┘
```

---

## Why Batch=1 Is FPGA-Ideal

| Property | GPU (batched) | This model (batch=1) |
|---|---|---|
| Activation memory | batch × seq × embed × layers | seq × embed × 1 layer |
| Weight reuse | high (amortised over batch) | sequential, fully pipelined |
| BRAM pressure | scales with batch size | **constant** |
| Parallelism strategy | SIMD across batch | **pipelined across time** |

Batch=1 online Adam maps naturally to FPGA **dataflow pipelines** —
one token sequence flows through the network, backward pass follows,
weights update, next sample begins. No batch synchronisation barriers.

---

## Comparison: GPU vs FPGA for This Model

| | RTX 4090 | ZCU102 FPGA |
|---|---|---|
| Speed (estimated) | ~18 ms/sample | ~8–15 ms/sample |
| Power | ~450 W | ~15 W |
| Perf/Watt | 1× (baseline) | **~20–30×** |
| Memory bus | GDDR6X (off-chip) | BRAM (on-chip, zero latency) |
| Cost (board) | ~$2,500 AUD | ~$5,600 AUD |
| Cost (chip only, volume) | N/A | **~$50–200 AUD** |
| DDR traffic during train | yes (weights too large) | **zero** (all on-chip) |

---

## Roadmap

**Paper 1** (current): BF16W micro-transformer training on CPU/GPU — algorithm and results.

**Paper 2** (next): Port training to FPGA.
> *"The 334K BF16W model fits entirely in on-chip SRAM of mid-range FPGAs
> (e.g. Xilinx Zynq UltraScale+), making direct hardware acceleration a
> natural next step — enabling training with zero DDR bandwidth and
> ~30× better power efficiency than GPU."*

**No algorithm changes required** — the C# training algorithm, .neuro
checkpoint format, and BF16W weight representation are used as-is.
FPGA work is purely HDL/hardware engineering.
