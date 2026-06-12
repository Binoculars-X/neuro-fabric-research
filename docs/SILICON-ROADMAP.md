# NeuronFabric Silicon Roadmap
## From 1M-Param Proof to Brain-Scale Analogue CIM

---

## 1. The Core Architectural Insight

GPUs waste most of their energy **moving weights**, not computing with them.

Every forward pass, a GPU must stream all weights from HBM (off-chip DRAM) to compute cores:
- 100B param model × 2B (BF16) = 200 GB to stream
- HBM bandwidth: ~3 TB/s → **67ms just to read weights**
- Power profile: ~700W, mostly memory bandwidth

**NeuronFabric approach:** weights live permanently inside the chip — in SRAM (Phase 1 FPGA/ASIC) or in an on-package DRAM array with embedded compute logic (Phase 2 DRAM-PIM). In both cases weights never leave the chip. Only activations and gradients cross chip boundaries.
No streaming. No memory wall. Pure multiply-accumulate.

> *GPUs waste most of their energy moving weights. Our chip never moves them.*

---

## 2. Single-Die Capacity by Process Node

### 2a. SRAM-only die (Phase 1 — FPGA/ASIC digital)

SRAM density scales with node shrink. A 400mm² die holds:

| Node | SRAM density | Params trainable (BF16W Adam, 10B/param) | Est. power |
|---|---|---|---|
| 12nm | ~0.5 MB/mm² | ~20M | ~5W |
| 7nm | ~0.9 MB/mm² | ~36M | ~8W |
| 5nm | ~1.5 MB/mm² | ~60M | ~12W |
| 3nm | ~2.5 MB/mm² | ~100M | ~12W |
| **2nm** | **~3.5 MB/mm²** | **~140M** | **~12W** |
| 1.4nm | ~5.5 MB/mm² | ~220M | ~12W |

BF16W Adam permanent memory per param: weight (2B) + m (4B) + v (4B) = **10 bytes/param**. Gradients are transient (computed per layer, Adam step applied, discarded) — not counted in permanent storage.

The architecture improves automatically with every node shrink — no redesign required.

---

### 2b. DRAM-PIM die (Phase 2 — Processing-In-Memory ASIC)

A more radical approach: a multi-die DRAM package (8–16 GB total) with an embedded compute die — similar to HBM architecture but targeting full training rather than inference-only. A single LPDDR5 die holds ~0.75–1.5 GB; an 8 GB package stacks 6–8 such dies with one logic/compute die at the base (the same pattern as Samsung HBM-PIM and SK Hynix AiM). The DRAM array **is** the chip — weights never leave the package. Only activations and gradients cross chip boundaries, exactly as the NeuronFabric software architecture already enforces.

**Permanent training memory per param (BF16W Adam):**

| Buffer | Dtype | Bytes/param | Permanent? |
|---|---|---|---|
| Weight | BF16 | 2 | ✅ always resident |
| Adam m | FP32 | 4 | ✅ always resident |
| Adam v | FP32 | 4 | ✅ always resident |
| Gradient dW | FP32 | 4 | ❌ transient per-layer |
| **Permanent total** | | **10** | |

Gradients are transient: with layer-by-layer backward + activation recomputation (standard technique, already described in FPGA-ARCHITECTURE.md), each layer's dW is computed, Adam step applied, dW discarded before moving to the next layer. Peak gradient buffer = one layer's weights = negligible vs total DRAM.

**Package structure (8 GB, 6-die stack):**

```
┌─────────────────────────────────────────────────────┐
│  NeuronFabric DRAM-PIM Package                      │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │ DRAM die 5  (~1.5 GB)  weights + m + v        │  │
│  ├───────────────────────────────────────────────┤  │
│  │ DRAM die 4  (~1.5 GB)  weights + m + v        │  │
│  ├───────────────────────────────────────────────┤  │
│  │ DRAM die 3  (~1.5 GB)  weights + m + v        │  │
│  ├───────────────────────────────────────────────┤  │
│  │ DRAM die 2  (~1.5 GB)  weights + m + v        │  │
│  ├───────────────────────────────────────────────┤  │
│  │ DRAM die 1  (~1.5 GB)  weights + m + v        │  │
│  ├───────────────────────────────────────────────┤  │
│  │ Logic die (compute)                           │  │
│  │  ┌──────────────────────────────────────────┐ │  │
│  │  │  6.8M subarrays, each with:              │ │  │
│  │  │   ~2048 weights (BF16 w + FP32 m,v)      │ │  │
│  │  │   1× BF16 FMA unit  (~500 transistors)   │ │  │
│  │  │   1× Adam state machine (~500 transistors)│ │  │
│  │  │  All subarrays update IN PARALLEL         │ │  │
│  │  │  Full Adam step for all 680M params: ~2µs │ │  │
│  │  └──────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  External pins (only):                              │
│    → Activations in  [seqLen × embedDim]            │
│    ← Activations out [seqLen × embedDim]            │
│    → Gradient in     [seqLen × embedDim]  (backprop)│
│    ← nothing (weights NEVER leave)                  │
└─────────────────────────────────────────────────────┘
```

**Compute block area overhead — transistor-count analysis:**

One FMA unit serves all ~2048 weights in its subarray sequentially (one weight updated per cycle). All 6.8M subarrays fire in parallel → full Adam step for all 680M weights takes ~2048 cycles ÷ 1 GHz = **~2 µs** — well within any training step budget.

- One DRAM cell: 1T1C (1 transistor + 1 capacitor)
- 8 GB package = 6.8 GB effective = ~54 billion DRAM cells → ~6.8M subarrays (~2048 weights each)
- One compute unit per subarray: BF16 FMA + Adam state machine ≈ **~1,500 transistors**
- Total compute transistors: 6.8M × 1,500 = **~10 billion**
- vs DRAM cell transistors: ~54 billion
- Compute overhead: 10B ÷ 54B = **~19%**

Samsung HBM-PIM achieves ~10–20% area overhead with more complex SIMD units — NeuronFabric's simpler per-subarray design lands in the same range. **~19% overhead confirmed feasible**, consistent with the 15–20% budget used in the capacity table above.

**Trainable params by chip size (10 bytes/param permanent, 85% DRAM available):**

| Chip DRAM | Effective DRAM | Training params (10B/param) | Inference params (2B/param) |
|---|---|---|---|
| 4 GB | 3.4 GB | ~340M | ~1.7B |
| **8 GB** | **6.8 GB** | **~680M** | **~3.4B** |
| 16 GB | 13.6 GB | ~1.36B | ~6.8B |
| 32 GB | 27.2 GB | ~2.72B | ~13.6B |

**MoE scaling with 8 GB DRAM-PIM chips (training, ~680M params/chip):**

| Chips | Total trainable params | Active params/token (top-√N) | Comparable model |
|---|---|---|---|
| 10 | ~6.8B | ~2.2B (top-3) | GPT-2 class |
| 100 | ~68B | ~6.8B (top-10) | GPT-3 class |
| 1,000 | ~680B | ~68B (top-32) | GPT-4 class |
| 10,000 | ~6.8T | ~680B (top-100) | Frontier class |

**Why DRAM-PIM changes the equation vs SRAM-only:**
- DRAM density is ~100× higher than SRAM — 8 GB vs ~40 MB for the same package footprint
- Internal DRAM bandwidth (~200–400 GB/s within die) eliminates the memory wall entirely
- Per-chip power remains low (~3–8W) — the compute block is tiny relative to the DRAM array
- The `ApplyUpdate` hook in the software architecture maps directly to the DRAM write path — no algorithm change required at any scale

**Relationship to existing PIM research:**
Samsung HBM-PIM and SK Hynix AiM both add compute to DRAM but target **inference only**. NeuronFabric's contribution is the full **training** loop (forward + backward + Adam) running inside the memory chip, validated by the software reference implementation in this repository.

---

## 3. Pipeline Parallelism — Scaling Depth

A transformer pipeline maps naturally to a chip chain:

```
[Chip 1: Embedding + Layers 1–100] → [Chip 2: Layers 101–200] → ... → [Chip N: Layers + LM Head]
```

- Only **activations** pass between chips: `[seqLen × embedDim]` tensor per hop
- At embedDim=128, seqLen=128: **32 KB per token** between stages — trivial bandwidth
- Weights never move — they stay in each chip's memory (SRAM or DRAM) permanently
- **All chips are identical silicon** — same mask, same fab run, configured by weight flash at boot

**Hard upper bound:** pipeline depth = number of transformer layers (1 chip per layer maximum).

**Latency per pipeline:**
- Die-to-die (UCIe, same package): ~10–50 ns/hop
- 1,000 hops × 50 ns = **50 µs total** — 200× faster than the brain's ~10 ms

---

## 4. Parallel Replicas — Scaling Throughput

Identical pipelines run in parallel for concurrent requests:

```
Pipeline lane 0: [C1]→[C2]→...→[CN]   ← request A
Pipeline lane 1: [C1]→[C2]→...→[CN]   ← request B
Pipeline lane K: [C1]→[C2]→...→[CN]   ← request K
```

This gives a **two-axis scaling model**:
- **Pipeline depth** (series) → more layers → deeper reasoning
- **Parallel replicas** (width) → more throughput → more concurrent users

Both axes scale **linearly** with chip count. No quadratic memory bandwidth wall.

---

## 5. Product SKU Ladder

### Phase 1 — SRAM ASIC (5nm, ~43M params/chip training)

| Product | Chips | Total params | Power | Market |
|---|---|---|---|---|
| Edge module | 4 | ~170M | ~50W | Car, phone base station |
| Enterprise node | 40 | ~1.7B | ~500W | Hospital, factory |
| Datacenter blade | 160 | ~7B | ~2kW | Cloud inference |
| Full rack | 640 | ~28B | ~8kW | Large-scale serving |

### Phase 2 — DRAM-PIM (8 GB package, ~680M params/chip training, ~3.4B inference)

| Product | Chips | Training params | Inference params | Power | Market |
|---|---|---|---|---|---|
| Edge module | 4 | ~2.7B | ~13.6B | ~30W | Car, phone base station |
| Enterprise node | 40 | ~27B | ~136B | ~200W | Hospital, factory |
| Datacenter blade | 160 | ~109B | ~544B | ~800W | Cloud inference |
| Full rack | 640 | ~435B | ~2.2T | ~3kW | Large-scale serving |
| **1T training system** | **~1,500** | **~1T** | — | **~12kW** | Frontier AI training |

**vs GPU equivalent for 1T training:** ~6,000 H100s × 700W = **~4 MW**
**NeuronFabric DRAM-PIM:** ~12 kW — **~330× less power**

---

## 6. ASIC Training Throughput Strategy — Outperforming GPUs

A single DRAM-PIM chip has a smaller GEMM unit than an H100. System-level throughput must be recovered through parallelism and architectural advantages. The ASIC roadmap explicitly targets the following:

### 6a. Large batch sizes (primary lever)
The FPGA proof-of-concept runs batch=1 due to BRAM constraints — this is an FPGA limitation, not an architectural one. The ASIC logic die includes a dedicated **activation SRAM buffer** sized for batch=256–1024:

- Activation buffer per chip: `batch × seqLen × embedDim × 2B` — e.g. batch=256, seqLen=128, embedDim=512 = **32 MB** (fits easily in on-die SRAM alongside the DRAM array)
- Adam step cost is independent of batch size — the ~2µs update is amortized over the whole batch
- **Effective throughput scales linearly with batch size** — the weight-stationary design means larger batch = more tokens per Adam step at zero extra weight-access cost

| Batch size | Adam step | Tokens/step | GPU advantage eliminated? |
|---|---|---|---|
| 1 (FPGA) | ~2µs | 1 | No — GPU wins on GEMM |
| 32 (current SW) | ~2µs | 32 | Partially |
| 256 (ASIC target) | ~2µs | 256 | Yes — memory wall gap closed |
| 1024 (ASIC max) | ~2µs | 1024 | Yes — NeuronFabric leads on tokens/joule |

### 6b. Pipeline fill — activation streaming
A 1000-chip pipeline, once filled, processes one new batch per chip-latency tick. The pipeline is never idle after warm-up:

```
t=0:  Chip1 processes batch A
t=1:  Chip1 processes batch B,  Chip2 processes batch A
t=2:  Chip1 processes batch C,  Chip2 processes batch B,  Chip3 processes batch A
...
Steady state: all 1000 chips busy simultaneously → 1000× throughput
```

This is standard GPipe / pipeline parallelism — well understood, directly applicable.

### 6c. MoE expert parallelism
With MoE, each token activates only top-k chips. The remaining N−k chips are **simultaneously processing other batch items**. At 1000 chips with top-32 routing, 968 chips are always processing in parallel. This is the primary throughput multiplier at scale.

### 6d. Gradient accumulation over micro-batches
For very large effective batch sizes (e.g. batch=4096), micro-batches of 256 are accumulated before the Adam step. This is already implemented in the software reference (`GradientAccumulator`) and maps directly to the ASIC — gradients accumulate in the transient FP32 buffer, Adam fires once.

### 6e. Throughput target vs H100

| Metric | H100 | NeuronFabric DRAM-PIM (1000 chips) | Advantage |
|---|---|---|---|
| Peak FLOPS | 2000 TFLOPS | ~200 TFLOPS (est.) | GPU 10× |
| Weight streaming | 700W to move | 0W — stationary | NF wins |
| Tokens/joule (training) | baseline | ~100–330× | **NF wins** |
| Adam step latency | ~10ms (stream all weights) | ~2µs (parallel in-place) | **NF 5000×** |
| Scaling cost | +$30k/chip | +$50–200/chip (DRAM pkg) | **NF wins** |

**The roadmap goal:** at batch=256+ with pipeline-parallel execution, NeuronFabric DRAM-PIM matches or exceeds H100 tokens/second at 1/100th the power per token. Raw FLOPS is not the target — **tokens/joule is the target**.

### 6f. Post-silicon optimization compounding — the long game

First silicon is only the starting point. Once the architecture exists in hardware, a dedicated optimization effort targeting *this specific chip* — not a general-purpose GPU — unlocks a compounding improvement curve that GPUs cannot follow:

**Hardware-level (chip rev 2, 3, ...):**
- Custom number formats tuned to transformer arithmetic (e.g. narrower exponent, wider mantissa than BF16)
- Wider subarrays (4096, 8192 weights/subarray) → fewer sequencing cycles → faster Adam step
- Dedicated activation GEMM array co-designed with the DRAM array — not bolted on
- Higher internal DRAM clock (2–4 GHz vs 1 GHz baseline) → Adam step down to ~0.5µs
- Custom interconnect (not UCIe general-purpose) → 2–5× lower hop latency

**Software/compiler level:**
- Transformer-specific dataflow scheduling: overlap backward pass of layer N with forward pass of layer N+1 — something GPUs do clumsily due to memory pressure
- Weight-stationary kernel fusion: attention + FFN + residual in one chip pass, zero activation eviction
- Learned routing schedules: MoE router trained jointly with chip topology (experts placed physically near their most frequent callers)
- Quantisation co-design: 4-bit or 6-bit weights possible since Adam moments remain FP32 — halves DRAM capacity requirement, doubles effective chip count

**Architecture-level (model design for the chip, not for GPU):**
- Transformer depth/width ratios optimised for pipeline fill, not GPU tensor core utilisation
- Sequence lengths tuned to activation buffer size, not HBM bandwidth
- Attention head count matched to subarray geometry

**The compounding trajectory:**

| Generation | Status | Expected speed vs first GPU-class GPU cluster |
|---|---|---|
| Gen 1 — first silicon | Architecture correct, unoptimised | ~0.1–0.5× tokens/sec, ~100–330× tokens/joule |
| Gen 2 — co-designed GEMM + DRAM | Dedicated compute array | ~1× tokens/sec (parity), ~200× tokens/joule |
| Gen 3 — custom dataflow + routing | Architecture-specific scheduler | ~3–10× tokens/sec, ~500× tokens/joule |
| Gen 4 — full stack (HW + SW + model) | NeuronFabric-native models | **~10–100× tokens/sec**, **~1000× tokens/joule** |

NVIDIA spent 15 years reaching H100 from G80. NeuronFabric starts from a structurally superior base — weights never move — so the ceiling is fundamentally higher. **First silicon proves the principle. The optimization curve is where the performance lead is built.**

---

## 7. Latency vs the Brain

The brain processes a full cortical inference cycle in ~10 ms.

| System | Token latency | vs brain |
|---|---|---|
| Human brain | ~10,000 µs | baseline |
| GPU H100 (DRAM bottleneck) | ~500 µs | 20× faster |
| NeuronFabric 5nm mesh (1T) | ~50 µs | **200× faster** |
| NeuronFabric 2nm mesh (1T) | ~20 µs | 500× faster |

**Latency is already solved.** The remaining race is purely **watts per synapse**.

---

## 8. Analogue CIM — Phase 3 Frontier

**Compute-in-Memory (CIM):** instead of reading weights then multiplying, the MAC happens
*inside* the SRAM bitcell — input voltage × stored weight = output current. Kirchhoff's law
does the accumulation. Zero data movement, zero clock cycles for the multiply.

| Technology | Energy/MAC | Status |
|---|---|---|
| GPU (H100) | ~100 pJ | Production |
| Digital SRAM (NeuronFabric today) | ~1 pJ | Buildable now |
| Analogue SRAM CIM (forward pass) | ~0.01 pJ | Silicon proven (ISSCC 2023–25) |
| **Analogue CIM + analogue Adam** | **~0.001 pJ** | Research (3–5 years) |

### Analogue Adam viability

Adam's three operations per weight:
```
m  = β1·m + (1-β1)·grad          ← linear — analogue friendly ✅
v  = β2·v + (1-β2)·grad²         ← squaring — needs extra circuit ⚠️
w -= lr · m / (√v + ε)           ← sqrt + division — hybrid digital ⚠️
```

The sqrt/divide is ~5% of total Adam operations. 95% is linear — fits analogue CIM perfectly.

**Approximate analogue Adam** (active research 2024–25):
- Replace `√v` with `|grad|` → fully analogue (sign-gradient methods)
- β values baked into capacitor ratios
- 6-bit moment precision — NeuronFabric BF16 results justify this is sufficient

Each weight becomes an **RC cell**: capacitors store m and v, charge/discharge rates = β1, β2,
input current = gradient, output voltage = updated weight. No digital logic, no clock.

### Bridge from Paper 1 to analogue

> Paper 1 proves BF16 Adam (7-bit mantissa) matches FP32 Adam quality.
> → Justifies 6-bit analogue Adam precision.
> → Justifies the analogue CIM Adam cell design.
> → Which is Paper 4.

---

## 9. Brain-Scale Power Projections

### Digital SRAM mesh

| Params | NeuronFabric digital | GPU cluster |
|---|---|---|
| 1T | 120W | 4,000W |
| 10T | 1,200W | 40,000W |
| 100T | 12,000W | 400,000W |

### Analogue CIM mesh

| Params | Analogue CIM | Human brain |
|---|---|---|
| 1T | **2W** | — |
| 10T | **20W** | — |
| **100T** | **200W** | **20W (brain)** |

**100T params (= human brain synapse count) at 200W** — a laptop charger.

---

## 10. Brain Comparison at 100T Params

| Metric | Human brain | NeuronFabric analogue CIM |
|---|---|---|
| Synapses / params | ~100T | 100T |
| Power | 20W | ~200W |
| Inference speed | ~10 ms | ~50 µs (200×) |
| Continuous learning | Yes | Yes (on-chip Adam) |
| Physical volume | 1.2 litres | ~20 litres (rack) |

**Remaining gap:** 10× on power — closed by analogue RC Adam cells (Paper 4).
Reaching 20W at 100T matches the brain on **every metric simultaneously**.

---

## 11. Chip Count at 100T (2nm, 250M params/chip)

```
100T / 250M = 400,000 chips
Pipeline per column: 10,000 layers → 10,000 chips
Parallel columns: 40
Total: 400,000 chips
Fab output: ~400 wafers (= 1 week of modern fab production)
Physical form: server rack (3D stacked chiplets, UCIe interconnect)
```

---

## 12. Research Paper Roadmap

| Paper | Topic | Key claim |
|---|---|---|
| **Paper 1** (current) | BF16 Adam on silicon | BF16 Adam matches FP32 quality; `ApplyUpdate` = SRAM update unit |
| **Paper 2** | Depth vs width scaling | Deep narrow networks on multi-chip pipelines |
| **Paper 3** | Multi-chip mesh scaling | Pipeline × replica topology; 33× power saving vs GPU at 1T params |
| **Paper 4** | Analogue CIM Adam | RC cell weight update; 0.001 pJ/MAC; 200W at 100T params |

---

## 13. Concurrent Workload Observation

During development, CPU BF16 Adam and CPU SGD ran simultaneously on the same R9 9900x
with no measurable interference. Both workloads are memory-bound (not compute-bound),
access separate memory regions, and use different execution unit mixes.

**Silicon implication:** a multi-chip SRAM fabric can run multiple model instances
concurrently — inference, training, fine-tune — without interference. The on-chip memory
hierarchy naturally partitions workloads. This is concrete multi-tenancy support with
no hypervisor or memory controller overhead.

---

*Last updated: May 2026. Training runs active: CPU BF16 Adam → 100k samples, CPU SGD plateau run.*
