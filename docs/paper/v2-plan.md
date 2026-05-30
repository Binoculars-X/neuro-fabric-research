# NeuronFabric — Paper v2 Plan

## Market Positioning — The Gap We Own

**Neuromorphic chips** (Intel Loihi, SpiNNaker, BrainChip) learn on-chip but use STDP —
a Hebbian rule that cannot reproduce gradient-descent results. They are incompatible
with modern transformer training. Target: neuroscience research, not LLMs.

**Inference ASICs** (Groq, Apple ANE, IBM NorthPole, 3–5nm) are efficient and well-funded
but do inference only — no weight update path exists on chip. Training is offloaded to GPU clusters.

**NeuronFabric owns the gap between both:**

| | Neuromorphic | Inference ASIC | **NeuronFabric** |
|---|---|---|---|
| On-chip weight update | ✅ | ❌ | ✅ |
| Gradient-descent compatible | ❌ | — | ✅ |
| Transformer / LLM compatible | ❌ | inference only | ✅ |
| Local Adam | ❌ | ❌ | ✅ |

**One-line positioning (abstract candidate):**

> *Neuromorphic chips learn the wrong way. Inference ASICs don't learn at all.
> NeuronFabric is the first architecture that learns the right way —
> gradient descent, Adam, transformer-compatible — entirely on chip.*

**The practical niche:**
Not "biological spikes". Not giant distributed GPU clusters.
**Practical local-learning MoE training fabric, compatible with modern transformer-style learning.**
Distributed, low-power, edge-deployable — one chip per expert, no host CPU in the update loop.

---

## The Opening (first paragraph of the paper)

> A human brain learns continuously at **20 W**.
> Training GPT-4 required approximately **30 MW sustained over months** — a factor of
> 1.5 million in energy per unit of capability gained.
> The gap is not primarily about transistor count or clock speed.
> It is about *where the weight update happens*.
> The brain updates synaptic weights locally, in place, with no external optimizer and
> no weight traffic across a bus. Every neural accelerator we are aware of does the
> opposite: it separates the compute from the update, ships gradients off-chip, and
> runs an optimizer on a host CPU or GPU cluster.
> NeuronFabric is an attempt to build hardware that learns the way the brain does.

This should be the first thing a reader sees. It sets the stakes and makes the
vocabulary constraint and Adam gap land as *steps toward a real question*, not
just benchmarks.

---

## The Thesis (one sentence)

> A transformer with full backpropagation can run entirely on a fixed-budget chip
> with weights permanently in local SRAM — and the two binding design constraints
> are the **vocabulary-to-parameter ratio** and the **choice of on-chip weight update rule**.

---

## What the Paper Must Show

### 1. The goal of the project
Every existing neural accelerator either does inference only (Groq, Apple ANE, IBM NorthPole)
or offloads the weight update to a host CPU (Cerebras, Tenstorrent).
NeuronFabric's goal is to close that gap: **one chip, weights in SRAM, full forward +
backward + weight update, no host involvement, no off-chip weight traffic.**

This is not incremental — it is a different architecture class.

### 2. What we have already proved

| Claim | Evidence |
|---|---|
| Transformer backprop can run entirely in C# with no ML framework | 61 unit + gradient-check tests passing |
| SRAM-local weight update is numerically identical to GPU SGD | CPU SGD = GPU SGD loss curve (within float32 tolerance) |
| Vocabulary tax is the dominant constraint at 100K params | Three-domain table: vocab=49 → loss 0.42; vocab=302 → loss 2.05; vocab=1501 (100K params) → loss 2.90 |
| On-chip Adam can be implemented in pure C# as silicon reference | `CpuAdamTransformerBus` — 7 tests passing, 2.95 loss @8k vs GPU Adam 2.71 |
| BF16 moments reduce SRAM from 1.2 MB → 800 KB with negligible loss | `CpuAdamBF16TransformerBus` — within 10% of float32 Adam at step 50 |
| 1M param TinyStories is a meaningful single-chip benchmark | Adam ceiling ~1.67 loss; SGD ceiling ~2.4–2.5 (run in progress) |

### 3. Why Adam instead of SGD — the motivation

At 8k samples, GPU Adam reaches loss 2.71; CPU SGD reaches 5.47.
At 100k samples, Adam plateaus at **1.67**. SGD at 1M samples is at **2.59 and still falling**.

This gap (~0.6–0.8 loss units at plateau) is the **quantified hardware cost of not having
an adaptive optimizer on chip**. It is not a software concern — it is a silicon design decision:

- **v1 chip (SGD only)**: simplest, no extra SRAM, ships first. Ceiling ~2.4.
- **v2 chip (BF16 Adam SRAM unit)**: +400 KB on-chip SRAM for moment storage. Ceiling ~1.67.

The paper's new contribution is measuring this gap empirically and expressing it as a
concrete silicon trade-off: **400 KB of SRAM buys you 0.7 loss units on the 1M param benchmark**.

### 4. What we still need to prove

| Open question | Experiment needed |
|---|---|
| Exact CPU SGD plateau | Current run to finish (~2–3M samples) |
| BF16 Adam loss ceiling (not just step-50) | BF16 run to 80–100k samples |
| AVX2 Adam speedup vs scalar (296ms/sample → ?) | Build + benchmark `OptimizedCpuAdamTransformerBus` |
| Multi-chip MoE beats single-chip Adam ceiling | MoE training run (not yet implemented) |
| BF16 loss is within ε of float32 Adam at ceiling | Compare BF16 plateau vs float32 plateau |

### 5. What we need to think about in future

- **SRAM sizing for Adam on SKY130**: 400 KB BF16 moments — does it fit in one tile?
  Current estimate: ~3.2 Mbit. SKY130 OpenMPW limit ~4–8 Mbit. Feasible, needs exact layout.
- **Gradient accumulation across chips in MoE**: the gating chip owns the loss; gradients
  must flow back through the inter-chip SPI link. Bandwidth and latency budget unknown.
- **Integer quantization of moments**: can we go INT8 moments (200 KB) with acceptable drift?
  BF16 is the first step; INT8 is the next if SRAM is the binding constraint.
- **Lion optimizer**: uses only sign(m) — one bit per weight for momentum, no second moment.
  Memory cost: ~100 KB for 1M param model. Convergence vs Adam on this benchmark: unknown.
- **Continuous/online learning**: the ASIC can update weights from live input without retraining
  cycle. Stability guarantees for streaming Adam on-chip are an open research question.

---

## v2 Paper Structure (5 sections, ~8–10 pages)

### 1. Introduction (~1.5 pages)
- **Opening**: human brain 20 W continuous learning vs 30 MW GPT-4 training — the gap is architectural, not computational
- The root cause: every existing chip separates the weight update from the weight storage
- NeuronFabric's claim: SRAM-local weights, full backprop, no host involvement
- Three contributions: (1) vocabulary-budget constraint, (2) adaptive optimizer gap quantified,
  (3) multi-chip scaling via shared-embedding MoE
- Related work folded in (3 short paragraphs: inference ASICs, neuromorphic, training accelerators)

### 2. Architecture (~2 pages)
- NeuronCore → AttentionCore → TransformerBus hierarchy
- The `ApplyUpdate` hook: single override point = the silicon SRAM update unit
- SGD default (v1 silicon) → Adam override (v2 silicon): same hardware path, different SRAM
- Hardware mapping table (software abstraction → BRAM/DSP/bus)

### 3. The Vocabulary-Budget Constraint (~1.5 pages)
- Equation 1: P_reason = P − |V|·d
- Four-row table: 100K/1501, 100K/302, 100K/49, 1M/1501
- One paragraph: threshold at P_reason ~ 80K for coherent patterns
- Implication: shared-embedding MoE pays the vocabulary tax once for the entire pipeline

### 4. The Adaptive Optimizer Gap (~2 pages) ← NEW in v2
- Benchmark: 1M param TinyStories, four variants compared
- Table: CPU SGD, CPU Adam float32, CPU Adam BF16, GPU Adam (oracle)
- The key number: ~0.7 loss units gap between SGD ceiling and Adam ceiling
- Hardware translation: 400 KB BF16 SRAM = the price of closing that gap on silicon
- BF16 vs float32 Adam: within X% — validates the tapeout SRAM specification

### 5. Multi-Chip Scaling and Roadmap (~2 pages)
- Shared-embedding MoE topology (from v1, keep compressed)
- Inter-chip bandwidth equation: fixed at T×d×4 bytes regardless of depth
- v1 silicon (SKY130, SGD): specification and open tasks
- v2 silicon (SKY130, BF16 Adam): 400 KB moment SRAM, specification and open tasks
- Future: INT8 moments, Lion optimizer, online learning stability

---

## What to Cut from v1

| v1 content | Reason to cut |
|---|---|
| Shakespeare failure narrative | Lab notebook, not thesis support |
| MultiWOZ step-by-step table (20k/50k/75k/100k) | Replace with one final-result row |
| Batch-size benchmarking (batch=8/12/16/24) | Implementation detail, not a claim |
| AVX2 optimisation stage table | Journal material; mention speedup number only |
| Multiple sample dialogue outputs | One example per domain maximum |
| Step-by-step CPU optimisation history | Keep only the final ms/sample numbers |

---

## The Core Validation Experiment (arXiv minimum bar)

This is the single most important experiment in the paper.
It answers the question a reviewer will immediately ask:
*"Does your C# BF16 Adam actually converge to the same loss as a real optimizer?"*

### Setup — three curves, same everything

| Property | Must be identical across all three |
|---|---|
| Dataset | TinyStories |
| Model size | ~1M params (embedDim=128, heads=4, ffDim=384, layers=4, vocab=1501, seqLen=128) |
| Tokenizer | Same `TinyStoriesLoader` word-level tokenizer |
| Seed | Same `Random(42)` initialisation |
| Train/val split | Same loader, same shuffle |
| Hyperparameters | lr=1e-3, β₁=0.9, β₂=0.999, ε=1e-8, batch=4 |

### The three curves

1. **GPU FP32 Adam** — TorchSharp `AdamTransformerBus`, RTX 4090 — the oracle
2. **GPU BF16/FP16 Adam** — TorchSharp with `torch.set_default_dtype(torch.bfloat16)` — intermediate reference
3. **C# BF16 Adam** — `CpuAdamBF16TransformerBus` — the hardware silicon reference

### Success criteria

| Criterion | Target |
|---|---|
| Final loss | C# BF16 within **1–3%** of GPU FP32 Adam at 100k samples |
| Accuracy | Within ~2% of GPU |
| Convergence | No divergence, no NaN, smooth curve |
| Loss range | C# BF16 reaches **≤ 1.8** (1.6–1.8 is sufficient for arXiv) |

### The money graph

X-axis: samples seen (0 → 100k).
Y-axis: eval loss.
Three lines: GPU FP32, GPU BF16, C# BF16.

If the C# BF16 line tracks within the GPU BF16 band — the paper is done.
The claim is not "better than GPU". The claim is:

> **BF16 local Adam is viable for tiny language-model training.**
> **This enables local-weight FPGA/ASIC training fabric.**

That is a clear, falsifiable, hardware-relevant engineering result.
It is publishable on arXiv as an engineering proof without further claims.

### What comes after (roadmap in the paper)

Once viability is shown:
- This enables v2 silicon: BF16 Adam SRAM unit on SKY130 tapeout
- The 400 KB SRAM cost is justified by the loss gap vs SGD (Section 4)
- Future: INT8 moments, Lion optimizer, streaming online learning

---

## Tomorrow's Agenda

1. Check BF16 train terminal — get eval loss @8k, start run to 100k
2. Build + benchmark `OptimizedCpuAdamTransformerBus` — ms/sample vs scalar 296ms
3. Run GPU BF16 curve (`--adam-gpu-bf16` flag — needs implementation)
4. Compare three curves → paper Section 4 graph
5. Tag `v0.1-paper` release once three curves are plotted

---

## On Tests as Scientific Instrument (add to paper Introduction or README)

> *AI-assisted development without tests amplifies uncertainty.
> AI-assisted development with deterministic tests converts belief into measurable evidence.*

This is not a software engineering point — it is a scientific validity point.

The NeuronFabric codebase was developed with AI assistance throughout.
Every architectural claim is backed not by "I think it works" but by a test that
either passes or fails:

| Claim | Test / evidence |
|---|---|
| Backpropagation is correct | Numerical gradient checks pass (finite difference vs analytic) |
| CPU SGD matches GPU SGD | Identical loss curves within float32 tolerance — 10 tests |
| Adam converges | Loss decreasing test, synthetic convergence @50 steps |
| BF16 moments don't diverge | BF16 within 10% of float32 Adam at step 50 — unit test |
| BF16 matches float32 at scale | TinyStories 8k run: loss 2.9435 vs 2.95 — 0.2% difference |

The gradient check is the hardest to fake: it computes $\frac{\partial L}{\partial w}$
numerically via finite differences and compares to the analytic backward pass.
If backprop is wrong, this test fails — no amount of "it looks reasonable" substitutes for it.

**For the paper:** this methodology is worth one paragraph in the Experiments section.
The reproducibility of AI-assisted research is an open question in the field.
Deterministic unit tests are the answer — they make the results falsifiable regardless
of how the code was written.

---

## Data still needed before writing v2

| Number | Source |
|---|---|
| CPU SGD final plateau loss + sample count | Background terminal (finish today or stop it) |
| CPU Adam BF16 eval loss @8k and @100k | Terminal `63f3832e` |
| OptimizedCpuAdam ms/sample | Build `OptimizedCpuAdamTransformerBus`, run benchmark |
| BF16 Adam plateau vs float32 Adam plateau | Both runs to ~100k samples |
