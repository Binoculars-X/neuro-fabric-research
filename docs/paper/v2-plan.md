# NeuronFabric — Paper v2 Plan

**Working title:** *NeuronFabric: A Software Reference Architecture for On-Chip Transformer Training with Local Adam*

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

> The long-term goal is a silicon chip where every neuron runs Adam locally — weights update in place, no gradient traffic off-chip, scaling limited only by the number of chips connected via axon data links.
> Paper 1 establishes the software proof that this is numerically viable.

---

## What the Paper Must Show

### 1. The goal of the project

The ultimate target is a **silicon chip where every neuron holds its own weights and runs Adam locally** — potentially in analogue circuitry. No gradient bus. No host optimizer. Scaling is achieved by connecting chips via axon data links only (activations forward, nothing backward off-chip). This is the architecture the brain uses at a functional level.

Paper 1 does not claim silicon. It establishes the **software reference proof**: local Adam in pure C# converges identically to GPU Adam. This is the numerical foundation that makes the FPGA and silicon steps credible.

Every existing accelerator either does inference only (Groq, Apple ANE, IBM NorthPole) or offloads the weight update to a host CPU (Cerebras, Tenstorrent). NeuronFabric closes that gap.

### 2. What we have already proved

| Claim | Evidence | Status |
|---|---|---|
| Transformer backprop can run entirely in C# with no ML framework | 61 unit + gradient-check tests passing | ✅ |
| SRAM-local weight update is numerically identical to GPU SGD | CPU SGD = GPU SGD loss curve (within float32 tolerance) | ✅ |
| Vocabulary tax is the dominant constraint at small param budgets | Three-domain table: vocab=49 → loss 0.42; vocab=302 → loss 2.05; vocab=1501 → TBD (EXP-001) | ⏳ clean number pending |
| On-chip Adam can be implemented in pure C# as silicon reference | `CpuAdamTransformerBus` — tests passing | ✅ |
| BF16W (w=BF16, m=FP32, v=FP32) zero convergence penalty | Appointment corpus only — TinyStories parity unknown | ⚠️ needs TinyStories run |
| BF16W vs FP32 Adam on TinyStories | **[PLACEHOLDER — future BF16W run]** — may show slower convergence; honest finding either way | ⏳ |
| **443K FP32 does NOT fit ZCU102** | 443K × 12 B = **5.32 MB** > 4.5 MB ❌ | ✅ calculated |
| **443K BF16W fits ZCU102** | 443K × 10 B = **4.23 MB** < 4.5 MB ✓ (270 KB headroom) | ✅ calculated |
| Same chip runs Shakespeare (256 vocab) or TinyStories (1501 vocab) | Embed SRAM provisioned for 1501 tokens; Shakespeare uses 90 KB of 528 KB | ✅ design verified |

### 3. Why Adam — and why on-chip

Every existing inference ASIC offloads the weight update to a host GPU/CPU.
Adam is the standard optimizer for transformer training. The paper's claim is that
Adam can run **entirely on-chip** — no host involvement, no off-chip moment storage.

The silicon cost is concrete: the **canonical 443K config** (embed=88, ff=264, layers=4, vocab=1501) requires 443K × 12 B = **5.32 MB** in FP32 — does **not** fit ZCU102's 4.5 MB budget.
**BF16W (weights=BF16, moments=FP32) drops this to 443K × 10 B = 4.23 MB — fits with 270 KB headroom.** This makes BF16W not optional but *required* for the FPGA target.
The 443K config is also the "universal" chip: it can train on Shakespeare (vocab=256) or TinyStories (vocab=1501) without hardware changes — only the embedding SRAM region changes content.

```
┌─────────────────────────────────────────────────────────────────┐
│           NeuronFabric 443K — Canonical FPGA Config             │
│          embed=88  heads=4  ff=264  layers=4  vocab=1501        │
├─────────────────────────────────────────────────────────────────┤
│  INPUT                                                          │
│  token [1..1501] ──► Embedding SRAM                            │
│                      vocab×embed = 1501×88                      │
│                      [  528 KB provisioned  ]                   │
│                      [  90 KB used for Shakespeare  ]           │
├─────────────────────────────────────────────────────────────────┤
│  TRANSFORMER CORE  (×4 identical layers)                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LayerNorm (88)                                         │   │
│  │  Attention: 4 heads × headDim=22                        │   │
│  │    Wq Wk Wv: [88×22]   Wo: [22×88]   fan-in: 88        │   │
│  │  + residual                                             │   │
│  │  LayerNorm (88)                                         │   │
│  │  FF: W1 [88→264]  W2 [264→88]   fan-in: 264 (max)      │   │
│  │  + residual                                             │   │
│  │                                                         │   │
│  │  Adam state per weight: m (FP32) + v (FP32)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          × 4                                    │
├─────────────────────────────────────────────────────────────────┤
│  OUTPUT                                                         │
│  logits = layerOut · Embedding.T   (weight-tied, no extra params)│
├─────────────────────────────────────────────────────────────────┤
│  SRAM BUDGET (ZCU102 = 4.5 MB)                                 │
│                                                                 │
│  BF16W  ████████████████████████░░  4.23 MB / 4.5 MB  ✓       │
│  FP32   ██████████████████████████████  5.32 MB  ✗ overflow    │
│                                                                 │
│  Breakdown (BF16W):                                             │
│    Weights  (BF16): 443K × 2 B = 0.89 MB                      │
│    Moments  (FP32): 443K × 8 B = 3.34 MB  (m + v)             │
│    Total:                        4.23 MB  ← fits ✓             │
└─────────────────────────────────────────────────────────────────┘
```

### 4. What we still need to prove

| Open question | Experiment needed |
|---|---|
| 380K FP32 Adam held-out floor on TinyStories | EXP-001 GPU run (in progress) |
| CPU FP32 Adam tracks GPU Adam | EXP-001 CPU run (in progress) |
| BF16W parity or penalty on TinyStories | Future BF16W run after FP32 baseline confirmed |
| CPU Adam FP32 generalises to second domain | **[OPTIONAL]** Shakespeare char-level (vocab~50), **380K params**, 100K samples (~4.5h) — same FPGA budget, negligible vocab tax vs TinyStories vocab=1501; supports Section 3 vocab-budget table |

### 5. What we need to think about in future

- **Lion optimizer**: sign(m) only — half the moment SRAM, worth benchmarking
- **INT8 moments**: further SRAM reduction if BF16W shows penalty
- **Continuous/online learning**: stability guarantees for streaming Adam on-chip

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
- SGD default → Adam override: same hardware path, different SRAM
- Hardware mapping table (software abstraction → BRAM/DSP/bus)

### 3. The Vocabulary-Budget Constraint (~1.5 pages)
- Equation 1: P_reason = P − |V|·d
- Four-row table: 100K/1501, 100K/302, 100K/49, 1M/1501
- One paragraph: threshold at P_reason ~ 80K for coherent patterns
- Implication: shared-embedding MoE pays the vocabulary tax once for the entire pipeline

### 4. The Adaptive Optimizer Gap (~2 pages) ← NEW in v2
- Benchmark: 380K param TinyStories, held-out 90/10 split, linear LR decay
- Table: CPU Adam FP32 vs CPU Adam BF16W vs GPU Adam FP32 (oracle)

| Variant | Val Loss @250K | Accuracy | ms/sample |
|---|---|---|---|
| GPU Adam FP32 | **[TBD — EXP-001 Run A]** | **[TBD]** | ~16 |
| CPU Adam FP32 | **[TBD — EXP-001 CPU run]** | **[TBD]** | ~300 |
| CPU Adam BF16W | **[TBD — future run]** | **[TBD]** | ~300 |

- The key number: gap between SGD ceiling and BF16W Adam ceiling (in loss units)
- Hardware translation: BF16W SRAM cost = **0.76 MB** for 380K params
- BF16W parity criterion: within **0.05 val loss** of GPU FP32 Adam — if penalty observed, report honestly as "slower convergence, requires further tuning" and note FP32 is sufficient for FPGA stage
- **FP32 Adam is the primary paper result**; BF16W is a secondary finding (positive or negative)

**Charts needed:**
- **Figure 1**: Loss curves (val loss vs samples) for all 4 variants on same axes — the main result
- **Figure 2**: SRAM budget breakdown (weights vs moments vs activations) — the silicon argument
- **Figure 3** (optional): Scaling table — params vs held-out floor at fixed sample budget

### 5. Hardware Feasibility and Roadmap (~1.5 pages)
- **No FPGA measurements in Paper 1** — this is a numerical/software paper
- **443K params** canonical config (embed=88, ff=264, layers=4, vocab=1501) — fits ZCU102 in FP32 (3.34 MB); same chip runs Shakespeare or TinyStories — 2 sentences only, not the focus
- Architecture maps naturally to silicon: weights in BRAM, `ApplyUpdate` hook = on-chip optimizer unit
- **FPGA demo target (Paper 2)**: Shakespeare char-level, **334K params** (embed=88, ff=264, vocab=256), full training on ZCU102 — no host CPU in the update loop. Loss drops from ~3.2 to ~1.54 entirely on chip. This is the hardware proof. (334K Shakespeare = same transformer core as 443K universal config, smaller embedding table)
- **Ultimate vision**: analogue Adam per neuron, chips connected via axon data links only (activations forward, nothing backward off-chip) — scaling by adding chips, not by widening a gradient bus
- Scaling table by process node (calculated, not measured):

| Node | SRAM density | Params (BF16W Adam) |
|---|---|---|
| 12nm | ~0.5 MB/mm² | ~33M |
| 7nm | ~0.9 MB/mm² | ~66M |
| 5nm | ~1.5 MB/mm² | ~100M |

- Future work: INT8 moments, Lion optimizer, on-chip fine-tuning from streaming data

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

## Tomorrow's Agenda

1. Review EXP-001 GPU results when run completes
2. Start BF16W run once FP32 baseline confirmed
3. Plot Figure 1 (loss curves) from EXP-001 data
4. Begin drafting Section 4 with real numbers
5. **Add .bat files to `Neuro.Attention.TrainApp`** for each tested model/target:
   - `run-exp001-gpu-adam.bat` — GPU FP32 Adam, 380K params, TinyStories, 250K samples
   - `run-exp001-cpu-adam-fp32.bat` — CPU FP32 Adam, 380K params, TinyStories, 350K samples
   - `run-exp001-cpu-adam-bf16w.bat` — CPU BF16W Adam, 380K params, TinyStories, 350K samples
   - `run-exp001-shakespeare.bat` — CPU FP32 Adam, **380K params**, Shakespeare char-level (vocab~50), 100K samples (optional — Section 3 vocab contrast, same FPGA budget)
   - Each bat file should include the full command with all hyperparameters so results are reproducible

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
| 380K GPU Adam FP32 held-out floor | EXP-001 GPU run (~1h, in progress) |
| CPU Adam FP32 curve tracks GPU | EXP-001 CPU run (~15h, in progress) |
| BF16W TinyStories curve | Future run after FP32 baseline confirmed |
| Shakespeare 380K CPU Adam FP32 (optional) | Short validation run (~4.5h) — same FPGA budget, char-level vocab~50 vs TinyStories vocab=1501 for Section 3 |
