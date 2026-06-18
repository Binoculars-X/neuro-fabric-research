# NeuroFabric — Multi-Phase Commercialisation Roadmap

---

## Bootstrap Strategy

**Principle:** delay external investment as long as possible. Every phase through first silicon is self-fundable.

| Phase | Cash out | How |
|---|---|---|
| Phase 1 — FPGA | $500–$2K | Buy one dev board |
| Phase 2a — C# Attention | $0 | Own time + AI tooling |
| Phase 2b — FPGA Attention | $0 | Same board already owned |
| Phase 3 — 130nm silicon | $0–$5K | Google OpenMPW (free shuttle) + $50 PCB |
| Phase 3b — 10-chip PCB | $150–$250 | Same OpenMPW chips + JLCPCB board |
| **Total to multi-chip silicon demo** | **< $10K** | **No investors needed** |
| Phase 4 — 5nm production | $5–20M | First raise — only after silicon proof |
| Phase 5 — 1M units | $50–200M | Series B/C — after commercial traction |

We will invest **up to $50K of own capital** if needed for chip production (PCB, test equipment, MOSIS shuttle if OpenMPW slot is unavailable). External investors are approached only when working silicon with measured benchmarks is in hand.

---

## Architecture Advantage Summary

Unlike systolic arrays (TPU, ANE, Groq) that are **inference-only** and require a separate GPU cluster for training, NeuroFabric keeps weights permanently in local SRAM next to each compute unit and supports native backpropagation on-chip. Training and inference happen on the same hardware.

| Property | GPU cluster (H100) | Inference-only ASICs (Groq/ANE) | NeuroFabric |
|---|---|---|---|
| On-chip training (backprop) | Yes, but ~1 MW/cluster | **No** | **Yes** |
| Training power | ~1 MW (data centre) | N/A | **~watts (on-chip)** |
| Weight traffic per forward pass | ~14 GB (HBM reads) | SRAM, low | **0 GB** (weights static) |
| Continuous learning (live updates) | No — requires retraining cycle | No | **Yes** |
| Multi-chip scaling cost | Grows with parameter count | Grows with parameter count | **Grows with layer width only** |

**Primary claim:** train a model at milliwatts on a chip that fits in your hand. No data centre. No retraining pipeline. No off-chip weight traffic.

---

## Phase 1 — Proof of Concept (Software + FPGA)

**Goal:** prove correctness and viability of the architecture  
**Investment:** $500–$2K (one FPGA dev board; everything else free)  
**Timeline:** 4–8 weeks (C# prototype complete; FPGA port + benchmarks remaining)

### Milestones
- [x] C# prototype: full forward + backprop pipeline
- [x] MNIST 97%+ accuracy (784 → 128 → 64 → 10, online SGD + mini-batch)
- [x] IRIS 99%+ accuracy
- [ ] FPGA implementation (Kintex-7 or Pynq-Z2)
- [ ] Benchmark: inference latency vs CPU/GPU on same model
- [ ] Benchmark: power consumption (target: <3W vs 320W GPU)

### Key Claims to Prove
- **Native backprop on FPGA** — training pass measured in watts, not megawatts
- **Training power vs GPU**: target <10W for MNIST-scale training vs ~300W GPU
- **Inference power**: <3W vs 320W GPU (secondary claim)
- **Sub-microsecond inference latency** per sample (single pipeline, no memory fetch)

---

## Phase 2 — Neuro.Attention + FPGA Validation

**Goal:** prove transformer attention in software, then validate on real FPGA hardware for power and latency benchmarks  
**Investment:** $0 (in-house development + AI tooling; same FPGA board from Phase 1)  
**Timeline:** 8–16 weeks post Phase 1

### Phase 2a — C# Neuro.Attention (software)
- [ ] `AttentionSignal` — 3-bit bus enum (Encode/Forward/Backward/WeightRead/WeightWrite)
- [ ] `EmbeddingLayer` — token index → dense vector (256 vocab, byte-level)
- [ ] `PositionalEncoding` — sinusoidal, stateless, zero BRAM
- [ ] `AttentionCore` — Q/K/V projections + scaled dot-product attention + backprop
- [ ] `AttentionLayer` — N heads parallel + output projection W_O
- [ ] `TransformerBus` — extends NeuralBus with Encode pass
- [ ] Dataset: Shakespeare corpus (~1M chars, byte-level, embedded resource)
- [ ] Dataset (fallback): TinyStories word-level loader — if Shakespeare accuracy plateaus below 30% at 100K params, switch to TinyStories (~1,500 word vocab, simple sentence structure, 100K model produces readable output; same loader interface as ShakespeareLoader)
- [ ] Benchmark: perplexity < 5.0 after 20 epochs; throughput vs PyTorch nanoGPT

### Phase 2b — FPGA attention + backprop
- [ ] Port `TransformerBus` to FPGA — sequence BRAM broadcast, fixed-point softmax
- [ ] Run Shakespeare next-character **inference** on hardware — verify output matches Phase 2a software
- [ ] Run Shakespeare **backprop pass** on hardware — measure training power (target: <10W)
- [ ] Measure weight update correctness — weights after FPGA backprop match C# reference within fp tolerance
- [ ] **Benchmark: training power on FPGA vs GPU** — this is the primary investor claim

### FPGA Implementation Risks
| Risk | Mitigation |
|---|---|
| `exp()` for softmax — no native FPGA unit | 256-entry LUT in BRAM; covers attention score range [-8, 0] |
| Fixed-point precision (Q8.8) — gradients may underflow | Use Q12.4 or Q16.0 wider accumulators for gradient registers |
| Timing closure at 100+ MHz | Target 50 MHz for Phase 2b proof; optimise clock later |
| BRAM port contention (2 heads reading same sequence) | Replicate sequence BRAM — 2 copies, one per head; 8 KB each |

All risks have standard solutions. Fixed-point convergence is the only empirical unknown — validated by comparing FPGA weight updates against C# reference output.

### Prototype Parameters
| Parameter | Value |
|---|---|
| Sequence length | 32 tokens (fixed) |
| Embedding dim | 64 |
| Attention heads | 2 (d_head = 32 each) |
| Feedforward dim | 128 |
| Vocabulary | 256 (byte-level, no tokeniser) |
| Transformer layers | 2 |
| Total parameters | ~100K |

### Gate to Phase 3
Phase 1 FPGA power measurement alone is sufficient to start angel conversations.  
Full gate: **Phase 2b FPGA attention backprop power measurement + Phase 2a perplexity benchmark** → submit OpenMPW shuttle application.

---

## Phase 3 — First Silicon (OpenMPW Shuttle, 130nm)

**Goal:** prove transformer attention runs on real silicon — correct output, measured power, measured latency. Perceptron is solved technology; attention on hardware with native backprop is the novel claim.  
**Investment:** $0–$50K self-funded (OpenMPW shuttle is free; budget covers PCB, test equipment, or MOSIS fallback if shuttle slot unavailable)  
**Timeline:** 6–12 months post Phase 2

### What runs on the chip
- **TransformerBus** — full attention forward pass (not MLP)
- Shakespeare next-character inference: 32-token context → next byte prediction
- Backprop pass to prove on-chip training (unique claim vs all competitors)
- Weight read/write via bus (model export/import in silicon)

### Chip Specs (target)
| Parameter | Value |
|---|---|
| Process | 130nm (Google OpenMPW / SKY130) or 180nm (MOSIS) |
| Die area | ~1–5 mm² (shuttle slot) |
| Attention heads | 2 (d_head = 32) |
| Sequence length | 32 tokens |
| Total parameters | ~100K (same as Phase 2a software model) |
| Clock | 50–100 MHz |
| Power (inference) | ~10–50 mW |
| Purpose | Architecture proof, not performance competition |
| Flash (weight persistence) | External SPI NOR Flash on PCB (~$1 component) |

### Cost Profile
| Item | Cost |
|---|---|
| OpenMPW shuttle slot | **$0** (Google-sponsored) |
| EDA tools (OpenROAD, Yosys, KLayout) | **$0** (open source) |
| PCB + test setup | ~$2–5K |
| **Total cash out** | **~$2K–5K** |

### Gate to Phase 4
Silicon produces correct next-character predictions matching Phase 2a software output + **measured training power < 50 mW** (backprop pass on-chip) → Series A raise for 5nm production chip.

### Phase 3b — 10-Chip Multi-Chip PCB: TinyStories 1M ($150–$250)
The 100K param Shakespeare model fits comfortably on a single chip. Phase 3b uses the full 10-chip capacity (~1M params) to train a **word-level TinyStories model** — a task where the output is legible English sentences, immediately impressive to a non-technical investor.

**Why TinyStories:** Microsoft Research (2023) showed that 1M params on this constrained domain (simple vocabulary, short children's stories, ~1,500 word types) produces coherent grammatically correct English — matching early GPT quality *on this domain*. The output is readable. An investor can see it working.

**10-chip model spec:**
| Parameter | Value |
|---|---|
| Dataset | TinyStories (~2M stories, ~475M tokens) |
| Vocabulary | ~1,500 word-level tokens (no BPE needed) |
| Sequence length | 64 tokens |
| Embedding dim | 128 |
| Attention heads | 4 (d_head = 32) |
| Feedforward dim | 512 |
| Transformer layers | 4 |
| Total parameters | ~1M (10× single chip) |
| Chip assignment | Embedding chip 1 · Layers 1–2 chips 2–5 · Layers 3–4 chips 6–9 · Output chip 10 |

**Milestones:**
- [ ] PCB design (KiCad): 10 chips in pipeline, SPI inter-chip links, shared power rail
- [ ] Port 1M param TinyStories model to chip layout (same HDL as Phase 3, wider d)
- [ ] Run **live training** across all 10 chips on TinyStories: forward pass → backprop in reverse → weights update on-chip simultaneously — model improves over epochs on hardware
- [ ] **Measure inter-chip traffic during training**: logic analyser shows activations forward + gradients backward = 512 bytes/sample (128 floats × 4 bytes) — not 14 GB
- [ ] **Measure total 10-chip training power**: target ~400 mW–1 W
- [ ] **Record perplexity drop over time on hardware** — same curve as C# software reference, proving silicon correctness
- [ ] Generate sample story text live from PCB — readable English output on hardware
- [ ] Confirm: inter-chip bandwidth fixed at 512 bytes/sample regardless of layer depth

| Item | Cost |
|---|---|
| PCB fabrication (JLCPCB, 4-layer, 5 copies) | ~$100–200 |
| Passives + connectors + power regulators | ~$30–50 |
| Logic analyser (Saleae clone) | ~$20 |
| **Total** | **~$150–$250** |

### Phase 3b Risks and Pre-conditions

| Risk | Reality | Mitigation |
|---|---|---|
| **Bare die packaging** | OpenMPW delivers unpackaged dies — cannot solder directly to PCB | Wire-bond to QFN carrier at packaging service lab (~$500–2K for 10 units); or die-on-board with wire-bond lab access |
| **Actual cost** | "$150–$250" assumes packaged chips; real cost is ~$700–2.5K including packaging | Budget $2–3K; use self-funded buffer if needed |
| **Inter-chip protocol must be in RTL at tapeout** | Cannot add inter-chip bus after Phase 3 chip is fabricated — it must be designed into the Phase 3 RTL | Design activation-passing bus (clock + data-valid + byte stream + ready/ack) as part of Phase 2b/3 RTL, before tapeout |
| **PCB design skills** | Someone must know KiCad or pay a designer | Self-learn KiCad (~2–4 weeks for a simple SPI daisy-chain board) or hire Upwork designer (~$500–1K) |
| **Single-chip validation first** | 10-chip PCB cannot be debugged if single chip is broken | Phase 3 must fully validate one chip in isolation (probe station or breakout board) before Phase 3b begins |
| **Phase 3b is optional** | If Phase 3 single-chip demo is already convincing to investors, Phase 3b may not be needed before Series A | Decide after Phase 3 results — do not commit to Phase 3b until single-chip is proven |

**Phase 3b go/no-go decision:** made after Phase 3 silicon validation. If single-chip demo (100K params, Shakespeare inference + backprop at <50 mW) is sufficient for Series A conversations, skip Phase 3b and go directly to Phase 4 raise. Phase 3b is the stronger demo but carries more execution risk.

**This is the Series A demo:** a $250 PCB with 10 chips **training a 1M param language model** at under 1 W — generating readable English stories — with a logic analyser showing 512 bytes inter-chip traffic vs 14 GB for an equivalent GPU. No cloud. No data centre. An LLM learning on a board you can hold in your hand.

### The 5-Minute Investor Demo Script

**Demo Step 1 — Train a language model from scratch at 1 W**

Setup: 10-chip PCB, power meter, laptop showing perplexity graph. Weights randomly initialised.

1. Start training on TinyStories corpus. Perplexity drops live: ~7.0 → ~3.5 over 2 minutes.
2. Total board power on the meter: **~0.8 W**.
3. Sample output appears on screen — readable English short stories generated by the hardware.
4. Point to the logic analyser: **512 bytes per sample** crossing each chip boundary. *"An H100 moves 14 GB per inference. We move 512 bytes. That number stays flat at 10 chips or 10,000."*

**Claim proven:** on-chip training at milliwatts. A GPU cluster needs ~1 MW for this. This board needs less than a USB port.

---

**Demo Step 2 — Real-time personalisation, no cloud, weights survive power-off**

**Pre-demo setup (before investors arrive):**
- Base model pre-trained on TinyStories (offline, on GPU) — gives grammatical English
- Fine-tuned on ~500 meeting-memory examples: *"My name is [X]"* → *"Nice to meet you, [X]"* and *"Who is in the meeting?"* → *"[X], [Y] and [Z]"* — teaches the recall pattern
- Pre-trained weights saved to Flash on the PCB. Board is ready.

**Live demo (investors present):**
1. Each investor types their name: *"My name is Alex"* / *"My name is Dima"* / *"My name is Petr"* — weights update on-chip after each exchange
2. **Unplug the board.** Everyone watches.
3. **Plug it back in.** Ask: *"Who is in this meeting?"*
4. Board answers: *"Alex, Dima and Petr."*
5. *"No data centre. No cloud. A $250 board. It learned your names and remembered them through a power cut."*

**Claim proven:** privacy-preserving continuous learning with Flash persistence. No inference-only ASIC (Groq, ANE, TPU) can do this — they have no backprop. No GPU cluster can do this without a data centre.

---

## Phase 4 — Production Node (5nm)

**Goal:** competitive inference chip, commercial sales  
**Investment:** $5–20M — **first external raise, only after Phase 3 silicon proof**  
**Timeline:** 12–24 months post Phase 3

### Chip Specs (target)
| Parameter | Value |
|---|---|
| Process | 5nm (TSMC N5) |
| Die area | ~300–400 mm² |
| Synapses | ~1B (fp16) |
| Clock | 800 MHz–1.2 GHz |
| Power (inference) | ~1–3W |
| HBM required | **No** — weights on-die |
| Flash (weight persistence) | Embedded eFlash on-die — weights survive power-off |

### Unit Economics at 100K Units
| Cost | NeuroFabric | H100 equiv. |
|---|---|---|
| Silicon | ~$55 | ~$280 |
| Packaging | ~$30 (no HBM) | ~$1,000 (CoWoS + 6× HBM) |
| **Total BOM** | **~$85–100** | **~$1,500–2,000** |
| **Retail (60% margin)** | **~$250** | **~$30,000** |

---

## Phase 5 — Multi-Chip Scale (1M Unit Production)

**Goal:** 1–2T synapse clusters, compete with GPU data centres  
**Investment:** $50–200M (Series B/C; volume manufacturing contracts)  
**Timeline:** 18–36 months post Phase 4

### Multi-Chip Architecture

```
┌──────────┐  activations  ┌──────────┐  activations  ┌──────────┐
│  Chip A  │ ────────────► │  Chip B  │ ────────────► │  Chip C  │
│  ~1B w   │  (4 MB/sample)│  ~1B w   │  (4 MB/sample)│  ~1B w   │
└──────────┘               └──────────┘               └──────────┘
     ◄──── backprop gradients (same bandwidth) ────────────────────
```

Inter-chip traffic = **activations only** (layer width × 4 bytes).  
Weight traffic across chip boundaries = **zero**.  
Scales linearly: 1000 chips = 1000× parameters, same inter-chip bandwidth.

### Scale Targets

| Config | Chips | Synapses | Equiv. Model | Power (est.) |
|---|---|---|---|---|
| Single chip | 1 | ~1B | LLaMA 1B | ~2–3W |
| Small cluster | 8 | ~8B | LLaMA 7B | ~20W |
| Mid cluster | 128 | ~128B | GPT-3 class | ~320W |
| Large cluster | 1,000 | ~1T | GPT-4 class | ~2.5 kW |
| Max cluster | 2,000 | ~2T | Beyond GPT-4 | ~5 kW |

Compare: GPT-4 inference on H100 cluster = **~1 MW**.  
NeuroFabric 1T synapse cluster = **~2.5 kW** — **400× lower power**.

### Unit Economics at 1M Chips Produced
| Cost component | Per chip |
|---|---|
| Silicon (5nm wafer + yield) | $50–60 |
| Packaging | $20–30 |
| Test | $5–10 |
| NRE amortised | $30–50 |
| **Total** | **~$110–150** |

1000-chip cluster cost: **~$110K–150K** vs **$30M for 1000× H100**.

---

## Competitive Landscape

| Company | Product | On-chip training | Weight traffic | Approx. cost/chip |
|---|---|---|---|---|
| Nvidia | H100 | No | ~14 GB/inference | $30,000 retail |
| Google | TPU v5 | No | High (systolic) | Not sold publicly |
| Apple | ANE (A17) | No | On-chip only, small models | Embedded |
| Cerebras | WSE-3 | No | Wafer-scale SRAM | ~$2–5M/system |
| Groq | LPU | No | SRAM, no HBM | ~$20K/card |
| Tenstorrent | Wormhole | Partial | SRAM + DRAM | ~$1,500/card |
| **NeuroFabric** | **Phase 5 ASIC** | **Yes** | **Zero (weights static)** | **~$250 target** |

**Unique differentiator:** NeuroFabric is the only architecture with **native on-chip backpropagation** — train a model at milliwatts on-chip. No competitor (Groq, Cerebras, ANE, TPU) supports on-chip training.

---

## Investment Thesis

| Phase | Raise | Milestone unlocked |
|---|---|---|
| Bootstrapped (Phases 1–3) | < $10K own capital | Working silicon, measured power + latency benchmarks |
| Self-funded buffer | up to $50K own capital | Phase 3 MOSIS fallback, test equipment |
| Series A | $5–20M | Phase 4: 5nm production chip, first commercial customers |
| Series B/C | $50–200M | Phase 5: 1M unit manufacturing, 1T synapse cluster |

**Comparables:** Groq ($300M raised), Cerebras ($720M), Tenstorrent ($700M, backed by Samsung + Hyundai).

The NeuroFabric differentiation over all three: **native backprop + zero weight traffic + linear multi-chip scaling**.
