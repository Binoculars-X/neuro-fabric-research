# NeuroFabric — Research Roadmap & Publication Strategy

---

## The Core Thesis

> A human brain learns continuously at 20 W.
> Training GPT-4 required ~30 MW sustained for months.
> The gap is architectural: every chip today separates the weight update from the weight storage.
> NeuroFabric closes that gap.

**The gap nobody owns:**

| | Neuromorphic (Loihi, SpiNNaker) | Inference ASIC (Groq, Apple ANE) | **NeuroFabric** |
|---|---|---|---|
| On-chip weight update | ✅ STDP only | ❌ | ✅ Adam / SGD |
| Gradient-descent compatible | ❌ | — | ✅ |
| Transformer / LLM compatible | ❌ | inference only | ✅ |
| Local BF16 Adam | ❌ | ❌ | ✅ |

**One-line positioning:**
> *Neuromorphic chips learn the wrong way. Inference ASICs don't learn at all.
> NeuroFabric is the first architecture that learns the right way —
> gradient descent, Adam, transformer-compatible — entirely on chip.*

---

## Three Independent Papers

### Paper 1 — Algorithm *(submit now)*

**Title:** *NeuroFabric: BF16 Adam for On-Chip Transformer Training*

**Core claim:**
> BF16 local Adam converges identically to GPU FP32 Adam on a 1M parameter
> transformer. The vocabulary-budget constraint quantified. Software silicon reference
> for FPGA/ASIC implementation.

**Key results:**
- BF16 Adam loss 2.9435 vs FP32 Adam 2.95 @8k samples — 0.2% difference
- CPU SGD plateau: ~1.97 @ 2.24M samples vs Adam plateau 1.67 @ 80k samples
- Optimizer gap: **+0.30 loss, 28× more samples** — quantified cost of no Adam unit
- Vocabulary-budget constraint: P_reason = P_total − |V|×d — proven across 3 domains

**What it does NOT claim:** faster than GPU, better than PyTorch, production hardware

**Venues:** arXiv → MLSys, ICLR (systems track), IEEE MICRO

**Status:** experiments running — BF16 to 100k samples needed, then write

---

### Paper 2 — Architecture *(after MoE software is built and trained)*

**Title:** *Shared-Embedding MoE Fabric for Fixed-Budget On-Chip LLM Training*

**Core claim:**
> Multi-chip MoE with shared embedding pays the vocabulary tax once,
> achieving X loss vs single-chip Y loss. Inter-chip bandwidth fixed at T×d×4 bytes
> regardless of model depth. Proven in C# software reference.

**Key results (pending):**
- MoE loss vs single-chip loss on TinyStories
- Expert load balance convergence
- Inter-chip bandwidth: 32 KB per sample (fixed) vs 14 GB for equivalent GPU inference

**Venues:** arXiv → ISCA, ASPLOS, MLSys

**Status:** design complete, implementation not started

---

### Paper 3 — Implementation *(after FPGA contractor delivers)*

**Title:** *NeuroFabric FPGA: 1M Parameter Transformer Training at <10W*

**Core claim:**
> First FPGA implementation of a 1M parameter transformer with full BF16 Adam
> backpropagation on-chip. Weights physically resident in BRAM. Adam update is
> a BRAM write. Power measured. Fidelity verified against C# silicon reference.

**Key results (pending):**
- J/sample: FPGA vs GPU vs CPU
- Power: target <12W during training loop
- Fidelity: loss curve matches C# BF16 reference within 1%
- Resource utilisation: LUTs, BRAMs, DSPs on ZCU102

**Venues:** arXiv → FPL, FCCM, DATE

**Status:** requires FPGA contractor (see below)

---

## Publication Timeline

```
Now (May 2026):     BF16 Adam run to 100k samples
                    AVX2 benchmark
                    GPU BF16 reference curve
June 2026:          Write Paper 1, submit to arXiv
July–Aug 2026:      Build MoE software, train, benchmark
Sep 2026:           Write Paper 2, submit to arXiv
                    Begin seed round conversations
Oct 2026:           Contract FPGA developer (parallel track)
Dec 2026:           FPGA results in hand
Jan 2027:           Write Paper 3, submit to arXiv
Q1 2027:            Series A conversations with all 3 papers
```

---

## FPGA Developer Contract

### What you need

One experiment, well-scoped:

> Implement BF16 Adam weight update on ZCU102 FPGA (EK-U1-ZCU102-G).
> Weights in BRAM. Adam update is a BRAM write.
> Measure power during training loop.
> Verify output matches C# reference within BF16 tolerance.

### Deliverables to specify in contract

- SystemVerilog/VHDL source: Adam update unit, BRAM controller, SPI interface
- Vivado project (synthesisable, with timing reports)
- Resource utilisation report (LUTs, BRAMs, DSPs) — goes directly in paper
- Power report (Vivado estimator + measured on-board) — goes directly in paper
- Testbench — verifies output matches C# reference
- **All work-for-hire, MIT license, published as open hardware**

### What to give the contractor

| File | Purpose |
|---|---|
| `ARCHITECTURE.md` | Hardware mapping table, SRAM budget, BF16 encoding |
| `REPRODUCE.md` | Algorithm spec and exact test criteria |
| `src/Neuro.Attention/Adam/BF16/CpuAdamBF16TransformerBus.cs` | Reference implementation to match |
| `src/Neuro.Attention.Tests/Unit/CpuAdamBF16BusTests.cs` | Correctness tests to pass |
| `results/benchmarks.md` | Expected loss curve to match |

### Success criterion

> Loss curve on first 1000 samples matches C# BF16 within 1%.
> Power consumption measured and reported.

### Budget

~2–3K AUD. Reasonable for 2–4 weeks FPGA work on a well-specified contract.

### Contract clause (include explicitly)

> All RTL source, testbenches, and Vivado project files are work-for-hire and will
> be released under MIT license as part of an open research publication.

---

## Open Hardware Repository Structure (after Paper 3)

```
neuro-fabric/
  src/           ← C# software reference (already open)
  rtl/           ← SystemVerilog from FPGA contractor
    adam_unit.sv
    bram_controller.sv
    spi_interface.sv
    tb_adam_unit.sv
  vivado/        ← Vivado project files
  results/       ← timing, power, utilisation reports, loss curves
  docs/          ← papers, journal, roadmap
```

---

## Why FPGA Speed Doesn't Matter

The ZCU102 will be slower than your CPU (ms/sample). That is not the point.

| Metric | RTX 4090 | ZCU102 FPGA | 12nm ASIC (est.) |
|---|---|---|---|
| ms/sample | 18ms | ~1000ms | ~20ms |
| Power | 450W | 12W | 1W |
| J/sample | 8.1 J | ~12 J | **0.02 J** |
| Weight traffic | HBM off-chip | **BRAM on-chip** | **SRAM on-chip** |
| Edge deployment | ❌ | ✅ | ✅ |

The paper claim for Paper 3 is not speed. It is:
> *Same convergence quality, weights never leave SRAM, power measured at <12W.*

The 12nm ASIC (post-funding) is where it becomes competitive with GPU on speed
while remaining 400× more energy efficient.

---

## Investment Story

### What investors see after 3 papers

| Question | Answer |
|---|---|
| Does the algorithm work? | ✅ Paper 1 — BF16 Adam matches GPU |
| Does the architecture scale? | ✅ Paper 2 — MoE multi-chip proven |
| Can you build real hardware? | ✅ Paper 3 — ZCU102 FPGA running at <12W |
| What's the power story? | ✅ FPGA measured J/sample vs GPU |
| What's the moat? | ✅ Open source + 3 publications + first mover |

### When to talk to investors

- **After Paper 2** — start seed round conversations. Software proof of architecture is sufficient.
- **After Paper 3** — Series A. Hardware demo closes the credibility gap.

### Why Sky130 is optional

Sky130 (130nm, free shuttle) proves you can tape out — not that the chip is competitive.
With seed funding, go directly to TSMC 28nm or GF 12nm where the chip is actually fast.
Sky130 is a side experiment for credibility, not a prerequisite.

### The 12nm silicon argument (post-funding)

At 12nm custom silicon with local SRAM Adam:
- ~20ms/sample — competitive with RTX 4090 at 18ms
- ~1W power vs 450W for 4090
- **0.02 J/sample vs 8.1 J/sample — 400× energy efficiency**
- Zero off-chip weight traffic
- Edge-deployable, continuous learning, no retraining cycle

The pitch is not *"faster than 4090 in a datacenter."*
The pitch is: **"same convergence, at the edge, at 1W, where a 4090 cannot go."**

---

## Why the Project is Sound

1. **The gap is real** — no chip does gradient-descent weight updates on the same silicon that stores the weights. This has not changed in 5 years of the AI boom.

2. **The timing is right** — edge AI inference is solved; edge AI *training* is the next unsolved problem. Data centre power limits are a physical constraint. Demand for efficient on-device learning is growing.

3. **The proof chain is solid** — GPU oracle → CPU float32 → CPU BF16 → FPGA → silicon. Each step is falsifiable, reproducible, and open source.

4. **The moat compounds** — 3 papers + open RTL + open software = a body of work that is hard to replicate quickly. First-mover in a specific niche.

5. **The risk is manageable** — Papers 1 and 2 cost nothing. Paper 3 costs 2–3K AUD contracted out. Seed round conversations start before any significant capital outlay.
