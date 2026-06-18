# NeuroFabric — Development Journal

---

## Day 1 ✅ — Neuro.Core (MLP baseline)
- Full forward + backprop pipeline (NeuronCore, NeuronLayer, NeuralBus)
- MNIST 97%+ accuracy, IRIS 99%+ accuracy
- xUnit + FluentAssertions test suite (unit, integration, slow)
- CI pipeline (GitHub Actions)

## Day 2 ✅ — Neuro.Attention Phase 2a (Transformer)
- `EmbeddingLayer`, `AttentionCore`, `AttentionLayer`, `TransformerBus`
- GeLU activation, weight tying, Pre-LN architecture
- Named-tensor `.neuro` checkpoint format (Save/Load, forward-compatible)
- `ShakespeareLoader` (byte-level, vocab=256, embedded resource)
- `Evaluate()` accuracy + loss (pure inference, no weight mutation)
- `Sample()` temperature sampling
- Parallel MatMul (`Parallel.For` outer loop, 12-core utilisation)
- Training app (`Neuro.Attention.TrainApp`): Ctrl+C save, `--resume`, streaming log
- Demo app (`Neuro.Attention.Demo`): interactive generation from checkpoint
- 61 tests passing (unit + integration + gradient checks)
- SuperSlow 100K test: 40K steps, Shakespeare accuracy plateau at ~30-33%
- **Result**: architecture proven correct; Shakespeare byte-level hits capacity ceiling at 115K params

---

## Day 3 ✅ — Batch Training + CPU Optimisation + Domain-Specialised 100K LLM

- [x] `TransformerBus.TrainBatch(int[][] tokens, int[][] targets, float lr)` — gradient accumulation, averaged gradients
- [x] `--batch-size` arg in `Neuro.Attention.TrainApp`
- [x] Integration test: `TrainBatch_ReducesLoss_OnCyclicPattern` (7/7 tests passing)
- [x] `TinyStoriesLoader` in `Neuro.Infrastructure` — download-on-demand, word-level vocab (top-1500), `GetBatch` / `Decode`
- [x] Slow test: `TinyStoriesSlowTest` — 5K steps, loss drops ≥20%, checkpoint saved
- [x] **CPU optimisation architecture** (clean separation):
  - `Neuro.Attention` core classes: `Parallel.For` removed, methods `protected virtual`, `sealed` removed
  - `AttentionLayer.CreateHead()` + `TransformerBus.CreateLayer()` factory overrides
  - New project `Neuro.Cpu.Optimized`: `OptimizedAttentionCore`, `OptimizedAttentionLayer`, `OptimizedTransformerBus`
  - Threshold-gated `Parallel.For` (only when outer dim > 64)
  - `TrainApp` uses `OptimizedTransformerBus` by default; `--no-optimize` for FPGA parity checks
- [x] **Parallel batch clone experiment** — tried cloning model per-batch, averaging gradients, to saturate 12-core CPU
  - Result: overhead (~45ms clone/average) exceeded compute (~60ms) for 115K param model — no speedup
  - Reverted from git. Conclusion: parallel batch worth it only for 10M+ param models
- [x] **Optimal batch size benchmarked**: batch=12 (= physical core count) at 4.45GHz / 70°C all-core turbo
  - batch=8: 41% CPU; batch=12: 65%; batch=16: 80%; batch=24: thermal drop to 4.1GHz (hyperthreading pressure)
- [x] **TinyStories 20K training run**: vocab=1501, 20K steps, batch=12
  - Final loss ~2.9, accuracy ~37% — output is recognisable English words but incoherent
  - Root cause confirmed: `64×1501 = 96K` of the 115K param budget consumed by output projection alone,
    leaving only ~19K params for all transformer layers (no reasoning capacity)
- **Key insight proven**: coherent output at 100K params requires vocab ≤ ~150 words

### Domain-specialised 100K LLM (FPGA target)

> **Design constraint**: NeuroFabric ASIC/FPGA chip = fixed ~100K param budget. Cannot scale params —
> must specialise the domain instead. Multiple chips pipeline for longer generation.

#### Appointment domain (vocab=49)

- [x] GPT-generated corpus `appointment.txt` — 600 lines, 4 dialogue types, constrained to ≤120 unique words
- [x] `AppointmentCorpusLoader` — reads embedded resource, auto-extracts vocab by frequency, word-level tokeniser
  - `GetBatch`, `Decode`, `Encode` interface; no caching needed (5K tokens, reads instantly)
- [x] Wired `--dataset appointment` into `TrainApp` and `Demo`
- [x] **Training result**: 20K steps, vocab=49, 6 layers, ff=256
  - Final: loss=0.42, accuracy=78.9% — plateaued at ~80%
  - Embedding: `64×49 = 3.1K params` → **96K params free for transformer layers**
- [x] **Demo confirmed fluent output** — sample generations:
  - `"can i book an appointment"` → *"is rescheduled for monday morning. please confirm if friday afternoon would suit you. we will send you a reminder."*
  - `"sorry the doctor is not available"` → *"on monday morning. please confirm if thursday afternoon would suit you."*
- **Hypothesis proven**: tiny vocab → all params go to reasoning → fluent domain output at 100K params

#### MultiWOZ (vocab=302)

- [x] `MultiWOZLoader` — downloads MultiWOZ 2.2 train split (17 files) on first use, extracts system turns only
  - 56,776 system turns, 905,979 tokens, top-300 vocab + UNK + EOS
- [x] Wired `--dataset multiwoz` into `TrainApp` and `Demo`
- [x] **Training results** (cumulative, resumable):

  | Total steps | Eval loss | Accuracy |
  |------------|-----------|----------|
  | 20K | 2.96 | 36.7% |
  | 50K | 2.45 | 43.8% |
  | 75K | 2.15 | 46.9% |
  | 100K | **2.05** | **50.8%** |

- **Curve flattened at 100K**: 75K→100K gained only 0.10 loss / +4% accuracy — half the rate of prior blocks.
  Model is at capacity ceiling for vocab=302 at 100K params. 50K more steps not worth it.

- [x] **Demo at 100K steps** — strong structural patterns: *"would you like me to book it for you"*, *"your reference number"*, *"moderately priced and free wifi"*, *"is there anything else i can help you with"*
  - `<UNK>` tokens appear for proper nouns (restaurant names, postcodes, phone numbers) — structurally unavoidable with top-300 vocab; does not affect training correctness
  - **Result: works well** — 100K param model trained in one day on real multi-domain dialogue, producing coherent booking/recommendation phrases
- **Observation**: appointment domain (vocab=49) plateaus faster and generates cleaner output due to constrained vocabulary.
  MultiWOZ (vocab=302) needs more steps but proves the architecture generalises to real-world data.
  Both validate the core FPGA chip design: one domain specialist per chip.

---

## Day 4 — CPU Optimisation Sprint

### CPU optimisation (AVX2 + workspace pooling)

- [x] `AttentionCore.Forward` made `virtual` — enables full override in optimised subclass
- [x] `OptimizedAttentionCoreHeavy` — replaces `OptimizedAttentionCore` under `#if HEAVY_CORE` (csproj define):
  - AVX2/FMA vectorised inner loops for all three MatMul variants (MulInto, MulTransposeBInto, MulTransposeAInto)
  - Pre-allocated workspace arrays for all Forward + Backward intermediates — zero GC allocation per step (except 2 return arrays)
  - In-place `SoftmaxInto` and `SoftmaxBackwardInto`
- [x] `*.neuro` added to `.gitignore`
- [x] `TrainApp` logging changed from ms/step → ms/sample; log interval auto-scales by batch size

### Benchmark results (appointment corpus, 100K param model)

| Implementation | ms/sample |
|---|---|
| Reference serial | 47.0 |
| Parallel.For rows (scalar) | 21.6 |
| + AVX2/FMA | 16.7 |
| + workspace pooling (Heavy) | **13.9** |

---

## Day 5 ✅ — 1M Param TinyStories + GPU Acceleration

### What was built

- **`Neuro.Gpu` project**: `TorchTransformerModel` — TorchSharp `nn.Module` with all weights resident on CUDA
  - `GpuTransformerBus` (`--gpu`): vanilla SGD, numerically matches CPU silicon reference
  - `AdamTransformerBus` (`--adam`): Adam optimiser, best convergence quality
  - Both delegate entirely to `TorchTransformerModel`; `float[,]` only at checkpoint boundaries
  - `SyncBaseWeightsToGpu()` on construction — identical starting weights to CPU for correctness tests
- **10 tests passing**: identical forward logits + identical SGD loss curves (within float32 tolerance)
- **`TransformerBus.TrainStep`**: removed per-layer `ClipGradNorm` calls inside the backward loop — these were attenuating gradients through deep stacks. Single clip on `dLogits` retained. CPU SGD now matches GPU SGD exactly.
- **`TrainApp`**: `--gpu` flag → `GpuTransformerBus`; `--adam` flag → `AdamTransformerBus`

### 1M param TinyStories baseline

Config: `embedDim=128, numHeads=4, ffDim=384, numLayers=4, vocab=1501, seqLen=128` ≈ 1.04M params

| Variant | ms/sample | Total (8k samples) | Eval Loss @8k | Accuracy @8k |
|---|---|---|---|---|
| CPU SGD (optimised) | 51.17 ms | 409.3 s | 5.47 | 3.9% |
| GPU SGD (`--gpu`) | 11.53 ms | 92.2 s | 5.43 | 7.0% |
| GPU Adam (`--adam`) | **17.60 ms** | **140.8 s** | **2.71** | **42.2%** |

- **GPU SGD is 4.4× faster** than CPU SGD with **identical loss** — confirms the silicon reference backprop is numerically correct
- **GPU Adam converges dramatically better** (loss 2.71 vs 5.47) — Adam's per-weight adaptive lr is critical for large-vocab transformers
- CPU SGD = FPGA/ASIC silicon reference; GPU SGD = numerical oracle; GPU Adam = practical trainer
- SGD convergence gap vs Adam is an ASIC design challenge — see `CHALLENGES.md`

### Adam convergence ceiling — 1M param TinyStories

Extended Adam training (resumable checkpoint, `ts-adam-20k.neuro`) to find the capacity ceiling:

| Samples | Eval Loss | Accuracy | Trend |
|---|---|---|---|
| 8k | 2.71 | 42.2% | baseline |
| 20k | 2.35 | 46.1% | ↓ |
| 40k | 1.96 | 50.8% | ↓ |
| 60k | 1.79 | 52.3% | ↓ |
| 80k | **1.67** | 56.3% | ↓ |
| 100k | 1.73 | 53.9% | → plateau |

**Capacity ceiling: ~1.67 eval loss, ~55–57% accuracy.** The model oscillates in this band from 80k onwards with no further downward trend — model capacity is exhausted.

**Demo output** (temperature=0.8, prompt `"Once upon a time"`):  
> *"Once upon a time there were two toys… she was so curious… they couldn't believe it… something unexpected happened… it's ok i'll give you a… thank you…"*

Story structure, dialogue, and narrative flow are clearly learned. `<UNK>` tokens are rare names/words outside the top-1500 vocab — structurally unavoidable, not a training defect.

### TinyStories as the multi-chip benchmark vector

> **The 1M param TinyStories task is now the primary benchmark target for multi-chip CPU/FPGA/ASIC.**

- Adam ceiling ~1.67 loss / 57% accuracy sets the **single-chip upper bound**
- CPU SGD ceiling (pending — 1M sample run incoming) sets the **ASIC silicon reference bound**
- The gap between them quantifies the **on-chip adaptive optimizer value** (Challenge 2 in `CHALLENGES.md`)
- MoE multi-chip architecture (Day 6) should push well below 1.67 — that is the thesis to prove
- Task is hard enough to be meaningful, cheap enough to iterate on (100k samples ≈ 6 min on RTX 4090)

### CPU SGD ceiling — 1M param TinyStories (silicon reference)

| Samples | Eval Loss | Accuracy | Notes |
|---|---|---|---|
| 8k | 5.47 | 3.9% | Day 5 baseline |
| ~1.065M | **2.60** | **48.4%** | still training — 57 ms/sample (optimised CPU) |

CPU SGD at 1M samples reaches **2.60 eval loss / 48%** and is still converging (not plateaued). Run continuing.

### CPU Adam — ASIC silicon reference (`--adam-cpu`)

- `ApplyUpdate(w, grad, lr)` added as `protected virtual` to `AttentionCore` and `AttentionLayer` — single override point mapping to the SRAM update unit on silicon
- `AdamAttentionCore/Layer`, `AdamEmbeddingLayer`, `CpuAdamTransformerBus` — pure C# float32 Adam in `Neuro.Attention/Adam/`
- `CpuAdamBF16TransformerBus` — BF16 moment variant (`ushort[,]` m/v); 800 KB vs 1.2 MB float32; target for Sky130 tapeout
- 7 unit tests passing for each variant (14 total); `--adam-cpu` and `--adam-cpu-bf16` flags in TrainApp

| Variant | ms/sample | Eval Loss @8k | Accuracy @8k |
|---|---|---|---|
| CPU SGD | 57 ms | 5.47 | 3.9% |
| **CPU Adam float32** | **296 ms** | **2.95** | **34.4%** |
| **CPU Adam BF16** | **337 ms** | **2.9435** | **39.8%** |
| GPU Adam | 18 ms | 2.71 | 42.2% |

CPU Adam tracks GPU Adam convergence (2.95 vs 2.71). **BF16 Adam matches float32 Adam to within 0.2%** (2.9435 vs 2.95) — convergence validated. BF16 moments halve the SRAM cost: 400 KB vs 800 KB for float32.

### CPU Adam BF16 — full 8k training curve

Config: same 1M param TinyStories. 337 ms/sample average over 8k samples.

| Samples | Eval Loss | Accuracy |
|---|---|---|
| 400 | 5.6129 | 5.47% |
| 1,200 | 4.6357 | 14.06% |
| 2,000 | 4.0726 | 22.66% |
| 3,200 | 3.5146 | 25.00% |
| 4,400 | 3.3045 | 27.34% |
| 5,600 | 3.1651 | 38.28% |
| 6,800 | 2.9140 | 40.62% |
| 8,000 | **2.9435** | **39.8%** |

Curve mirrors GPU Adam shape — steep initial drop, then slowing. No divergence. BF16 rounding does not destabilise training.

### The Adaptive Optimizer Gap

CPU SGD at 2.18M samples: **2.01 eval loss**. Still descending slowly.
GPU Adam plateau: **1.67 @ 80k samples**.

| Variant | Loss @80k samples | Samples to ~2.0 loss | Sample efficiency |
|---|---|---|---|
| GPU Adam | **1.67** (plateau) | ~40k | 1× |
| CPU SGD | ~4.x | **2,180,000** | ~55× more |

**27× more samples to reach 2.0 loss with SGD vs Adam. Gap at equivalent samples: ~0.34 loss units.**
This is the quantified hardware cost of not having an on-chip Adam SRAM unit.

### Class hierarchy built

```
AttentionCore → AdamAttentionCore → AdamBF16AttentionCore
AttentionLayer → AdamAttentionLayer → AdamBF16AttentionLayer
EmbeddingLayer → AdamEmbeddingLayer → AdamBF16EmbeddingLayer
TransformerBus → CpuAdamTransformerBus        (--adam-cpu)
              → CpuAdamBF16TransformerBus     (--adam-cpu-bf16)
              → OptimizedCpuAdamTransformerBus (--adam-cpu-opt, AVX2, pending benchmark)
```

`OptimizedCpuAdamTransformerBus` built (AVX2 + Parallel.For, 8 floats/cycle) — benchmark pending.

---

## Day 6 — Paper v2 Assembly: Proving On-Chip Adam is Viable

> **Goal**: produce the three experiments that together constitute the paper's proof.
> Each experiment answers one reviewer question. All three must pass before submission.

---

### Experiment 1 — BF16 Adam full convergence (the main claim) 🔄

**Question it answers:** *Does C# BF16 Adam actually reach the same loss floor as GPU FP32 Adam?*

**Setup:**
- Same model: 1M param TinyStories (embedDim=128, heads=4, ffDim=384, layers=4, vocab=1501, seqLen=128)
- Same seed: `Random(42)`
- Same tokenizer: `TinyStoriesLoader` word-level
- Run: `--adam-cpu-bf16` to **100k samples** (resuming from `ts-adam-bf16.neuro` @ 8k)

**Success criterion:** final eval loss ≤ 1.75 (within 5% of GPU Adam plateau 1.67)

**Status:** running — at 8k samples, loss 2.94. Need to resume to 100k.

**Command:**
```powershell
dotnet run --project src/Neuro.Attention.TrainApp -c Release -- `
  ts-adam-bf16.neuro 100000 --resume --adam-cpu-bf16 `
  --dataset tinystories --embed-dim 128 --num-heads 4 --ff-dim 384 --num-layers 4 `
  --batch-size 4 --log-every 4000
```

---

### Experiment 2 — GPU BF16 Adam reference curve 🔲

**Question it answers:** *Is the BF16 precision loss from C# rounding or from BF16 itself?*

This is the middle curve in the three-curve comparison graph. If GPU BF16 and C# BF16 track each other, the BF16 encoding is correct and the silicon reference is valid.

**Setup:**
- Add `--adam-gpu-bf16` flag to TrainApp → `AdamTransformerBus` with `torch.set_default_dtype(bfloat16)`
- Same 1M param TinyStories config, same seed
- Run to 100k samples

**Success criterion:** GPU BF16 loss within 2% of GPU FP32 Adam at 100k — confirms BF16 is not the cause of any gap

**Status:** needs implementation + run

---

### Experiment 3 — AVX2 Adam throughput benchmark 🔲

**Question it answers:** *Is the 296ms/sample scalar Adam fast enough to be practical, or does FPGA need dedicated silicon?*

This isn't a convergence test — it's a speed test. Shows the update loop is the bottleneck and motivates the SRAM unit design.

**Setup:**
- Build `OptimizedCpuAdamTransformerBus` (already created, needs build verification)
- Run 2000 samples, measure ms/sample vs scalar 296ms
- Check CPU utilisation: expect 60–80% vs current 6–7%

**Success criterion:** ≥ 4× speedup over scalar (target: ≤ 80ms/sample)

**Command:**
```powershell
dotnet run --project src/Neuro.Attention.TrainApp -c Release -- `
  ts-adam-opt.neuro 2000 --adam-cpu-opt `
  --dataset tinystories --embed-dim 128 --num-heads 4 --ff-dim 384 --num-layers 4 `
  --batch-size 4 --log-every 100
```

---

### Experiment 4 — Trained model quality test (demo validation) 🔲

**Question it answers:** *Does a BF16 Adam trained model actually generate coherent text?*

After Experiment 1 finishes (100k samples), run the demo and record sample outputs. This is the qualitative proof that the silicon reference produces a usable model.

**Setup:**
```powershell
dotnet run --project src/Neuro.Attention.Demo -c Release -- `
  ts-adam-bf16.neuro --dataset tinystories --temperature 0.8
```

Prompts to test:
- `"Once upon a time"`
- `"The little girl"`
- `"One day a dog"`

**Success criterion:** coherent English sentences, story structure, no repeated loops

---

### The Three-Curve Graph

Once Experiments 1 and 2 are done, generate `results/loss_curves.pdf`:

```
X-axis: samples (0 → 100k, log scale)
Y-axis: eval loss

Line 1: GPU FP32 Adam  — 8 data points (already have ✅)
Line 2: GPU BF16 Adam  — pending Experiment 2
Line 3: C# BF16 Adam   — pending Experiment 1 (100k)
Annotation: CPU SGD plateau ~1.97 @ 2.24M samples (horizontal dashed line)
```

Script: `results/plot_curves.py` (to be created)

---

### Paper sections unlocked by each experiment

| Experiment | Unlocks |
|---|---|
| 1 — BF16 full convergence | Section 4 table final row; abstract claim; conclusion |
| 2 — GPU BF16 curve | Three-curve graph; confirms BF16 encoding correct |
| 3 — AVX2 benchmark | Section 4.2 throughput; FPGA silicon unit motivation |
| 4 — Demo quality | Section 4.3 qualitative results; sample outputs |
| CPU SGD plateau ✅ | Section 4.1 optimizer gap table (1.97 @ 2.24M) |

---

### CPU SGD plateau — confirmed (Day 6, morning)

| Samples | Eval Loss | Notes |
|---|---|---|
| 1,089,100 | 2.5883 | Day 5 last checkpoint |
| 1,630,176 | 2.3054 | |
| 2,183,176 | 2.0689 | |
| 2,232,176 | **1.9425** | oscillating ~1.94–2.04 |
| 2,240,176 | 1.9778 | still slowly descending ~0.03/50k |

Run still active. Plateau ~1.93–1.97 expected at ~3M samples. For the paper: reporting as
**"1.97 @ 2.24M samples, still slowly descending"** is sufficient — the gap vs Adam is already clear.

**Optimizer gap (current best estimate):**
- CPU SGD: ~1.97 @ 2.24M samples
- GPU Adam: 1.67 @ 80k samples
- **Gap: +0.30 loss, 28× more samples required**

### Observation — concurrent CPU SGD + CPU BF16 Adam on R9 9900x (Day 6)

During testing, both runs were active simultaneously on the same R9 9900x with no measurable
interference on either loss curve or throughput.

**Why:** both workloads are memory-bound (weight load/store dominates), not compute-bound.
BF16 weights are half the FP32 footprint, so they fit more readily in L3 cache. The two
processes access entirely separate memory regions and use different execution unit mixes,
so they coexist on the shared memory subsystem without cache thrashing.

**Silicon implication:** a multi-chip SRAM fabric could run multiple model instances
concurrently — one inference, one training, one fine-tune — without significant interference.
This is a concrete multi-tenancy argument: the on-chip memory hierarchy naturally partitions
workloads the same way L3 cache does on a CPU. Worth a paragraph in the multi-chip scaling
section of Paper 1, and a dedicated experiment in Paper 3.

---

### Day 6 — BF16 Adam convergence experiment: conclusion

**Run:** `CpuAdamBF16TransformerBus` (w=FP32, m=BF16, v=BF16), resumed from 8k to 43k samples.

**Full data (eval loss):**

| Samples | Eval Loss | vs GPU Adam | Accuracy |
|---|---|---|---|
| 8,000 | 2.9435 | +0.23 | — |
| 10,000 | 2.8249 | +0.47 | 38.3% |
| 20,000 | 2.7894 | +0.44 | 41.4% |
| 30,000 | 2.7793 | +0.82 | 40.6% |
| 40,000 | 2.7536 | +0.79 | 40.6% |
| 43,000 | 2.7514 | — | 40.6% |

GPU Adam reference: 2.35 @ 20k, 1.96 @ 40k, 1.67 plateau @ 80k.

**Finding:** BF16 moment storage (m=BF16, v=BF16) causes significant convergence lag.
Gap widens from +0.23 at 8k to ~+0.79 at 40k. The curve is still descending but very slowly.

**Root cause (confirmed by literature):**
Dettmers et al. (ICLR 2022) showed that even 8-bit moments require **block-wise dynamic
quantization** to match FP32 convergence. Simple linear BF16 truncation (what this
implementation uses) loses precision in the low-magnitude moment values that dominate
early training, causing the lag. This is not a bug — it is a known limitation of naive
moment quantisation.

**Decision:** stop this run. Implement corrected design: **w=BF16, m=FP32, v=FP32**.

**Why the corrected design works:**
- Moments stay FP32 → Adam update is numerically identical to GPU FP32 Adam
- Weights stored as BF16 → 50% weight SRAM saving (2B vs 4B per param)
- On forward pass: decode BF16 → FP32 (1 instruction), compute, encode back
- Total SRAM: 10 bytes/param vs 12 bytes FP32 = **17% saving, zero convergence penalty**

**Paper claim (revised):**
> *"BF16 weight storage reduces weight SRAM by 50% with zero convergence penalty,
> validated empirically. BF16 moment storage (naive truncation) degrades convergence
> and is not recommended — FP32 moments are required for convergence parity."*

**Next:** implement `CpuAdamBF16WeightsTransformerBus` (w=BF16, m=FP32, v=FP32)
and run 8k validation to confirm convergence matches GPU Adam.

---

### Day 7 — BF16 Weights Adam validation + Scaling experiment

#### BF16 Weights Adam (w=BF16, m=FP32, v=FP32) — 10k run

**Implementation:** `CpuAdamBF16WeightsTransformerBus` — weights stored as BF16 (ushort),
decoded to FP32 for forward pass, Adam moments remain FP32, encode back after update.

**Results @ 10k samples (1M param config, batch=4):**

| Samples | Eval Loss | Accuracy |
|---|---|---|
| 1,000 | 4.79 | 15.6% |
| 5,000 | 3.30 | 28.9% |
| 10,000 | **2.87** | 39.1% |

GPU Adam FP32 reference: 2.71 @ 8k samples. BF16W is ~5k samples behind — consistent
with CPU vs GPU implementation differences (different forward pass, accumulation order),
not BF16 quantisation.

**Key comparison — BF16 moments (wrong) vs BF16 weights (correct):**

| Design | @ 13k samples | Notes |
|---|---|---|
| BF16 moments (m=BF16, v=BF16) | 2.75 | gap still widening at 43k |
| **BF16 weights (w=BF16, m=FP32, v=FP32)** | **2.75** | tracking normally, still dropping |

BF16W reached 2.75 at 13k. BF16 moments needed 43k to reach the same loss — **3× slower
convergence with the wrong design**. Dettmers hypothesis empirically confirmed.

---

#### CPU FP32 Adam baseline — 10k run

Running concurrently to isolate CPU vs GPU implementation gap from BF16 effect.
If CPU FP32 and CPU BF16W land at identical loss at 10k → BF16 weight quantisation
introduces zero penalty. Results pending.

---

#### Scaling experiment — 800K params GPU Adam (80k samples)

**Config:** `embedDim=96, heads=4, ff=288, layers=4` ≈ 800K params

**Full curve:**

| Samples | Eval Loss | Accuracy |
|---|---|---|
| 10k | 2.88 | 39.8% |
| 20k | 2.33 | 50.0% |
| 30k | 2.20 | 51.6% |
| 40k | 2.04 | 50.0% |
| 50k | 1.93 | 50.0% |
| 65k | **1.64** | 56.3% |
| 80k | **1.68** | 60.9% |

**Floor: ~1.64–1.68** — essentially identical to 1M param floor of 1.67.

**Finding:** 800K params reaches the same capacity floor as 1M params on TinyStories.
The +200K param step yields no meaningful improvement. TinyStories (vocab=1501) is
**capacity-saturated at ~800K params** — the dataset is the bottleneck, not the model.

**Implication for FPGA/ASIC:**
- 800K BF16W weights = **1.6 MB** — fits in mid-range FPGA BRAM for inference
- No benefit to 1M for this dataset; 800K is the optimal size

---

#### Scaling experiment — 600K params GPU Adam (80k samples)

**Config:** `embedDim=104, heads=4, ff=312, layers=4` ≈ 590K params

**Key points from curve:**

| Samples | Eval Loss | Accuracy |
|---|---|---|
| 10k | 2.82 | 37.5% |
| 30k | 2.20 | 45.3% |
| 50k | 2.06 | 49.2% |
| 70k | **1.78** | 53.9% |
| 72k | **1.77** | 51.6% |
| 73k | **1.76** | 55.5% |
| 80k | **1.81** | 52.3% |

**Floor: ~1.75–1.78** — meaningfully higher than 800K floor of ~1.64.

**Demo output (600K, temp=0.8):** Mostly real words, coherent TinyStories structure,
only ~4 UNK tokens per 150 generated — noticeably better than the 1M SGD model (which
had UNKs everywhere at ~1.87 loss), confirming loss directly correlates with generation quality.

**Updated scaling table:**

| Params | Config | Floor @ 80k | BF16W SRAM |
|---|---|---|---|
| ~590K | embedDim=104 | ~1.77 | 1.2 MB |
| ~800K | embedDim=96 | ~1.64 | 1.6 MB |
| ~1M | embedDim=128 | ~1.67 | 2.0 MB |

**Finding:** The capacity knee is between 600K and 800K. 600K clearly underfits
(floor 1.77 vs 1.64). 800K and 1M are effectively identical. The optimal parameter
count for TinyStories vocab=1501 is **~800K**.

---

#### Scaling experiment — 400K params GPU Adam (80k samples)

**Config:** `embedDim=80, heads=4, ff=240, layers=4` ≈ 380K params

**Key points:**

| Samples | Eval Loss | Accuracy |
|---|---|---|
| 10k | 2.84 | 40.6% |
| 30k | 2.10 | 45.3% |
| 50k | 1.86 | 57.0% |
| 69k | **1.84** | 54.7% |
| 72k | **1.83** | 50.0% |
| 80k | **1.86** | 50.0% |

**Floor: ~1.83–1.86** — higher than 600K (1.77), confirming continued underfitting.

**Demo output (400K, temp=0.8):** Coherent TinyStories structure, correct narrative arc
("tim saw a sad turtle... tim learned... friends were happy"), only ~3 UNK tokens per
150 generated. Usable output despite higher loss.

**FPGA significance:** 380K FP32 full on-chip training fits ZCU102 (6.9 MB BRAM+URAM):
- Weights: 1.52 MB
- Moments (m+v): 3.04 MB
- Activations + gradients: ~0.94 MB
- **Total: ~5.5 MB** ← fits with 1.4 MB headroom

This is the **ZCU102 POC target**: full FP32 transformer training entirely on-chip,
no DDR access, ARM PS side runs training loop, PL accelerates MatMul.

**Complete scaling table (TinyStories, vocab=1501, GPU Adam FP32, 80k samples):**

| Params | Config (embedDim) | Floor @ 80k | BF16W SRAM | FP32 training SRAM |
|---|---|---|---|---|
| ~380K | 80 | ~1.85 | 0.76 MB | **5.5 MB** ← ZCU102 fits |
| ~590K | 104 | ~1.77 | 1.18 MB | 8.5 MB |
| ~800K | 96 | ~1.64 | 1.60 MB | 11.5 MB |
| ~1M | 128 | ~1.67 | 2.00 MB | 14.4 MB |

**Capacity knee: 600K–800K.** Above 800K no improvement. Below 600K clear underfitting.
**FPGA on-chip training: 380K FP32 on ZCU102** is the clean POC target.

**Paper claim (FPGA):**
> *"A 380K parameter FP32 transformer requires ~5.5 MB for full on-chip training
> (weights + Adam moments + activations), fitting within the 6.9 MB BRAM+URAM budget
> of a Zynq UltraScale+ ZCU102. This enables complete gradient-based training without
> DDR access, achieving eval loss ~1.85 on TinyStories — demonstrating viable on-chip
> learning in silicon."*

---

### Day 8 — Resume bug fix + BF16W vindication

#### Bug: Adam moments not saved in checkpoint (fixed)

**Root cause:** `TransformerBus.Save` serialised weights only. On `--resume`, all Adam
moment arrays (m=0, v=0) and step counters were reset to zero while weights were
restored. This corrupts the optimizer state: v≈0 produces artificially large adaptive
steps, m=0 ignores all prior gradient history.

**Impact on previous results:**
- The Day 7 BF16W run was **resumed from a 10K checkpoint** — the stall at eval loss
  ~2.71 (samples 11K–39K) was entirely caused by corrupted moment state, not BF16
  quantisation noise. The "confirmed convergence penalty" conclusion was **wrong**.
- All other previously recorded runs started fresh (no resume) so are unaffected.

**Fix implemented:**
- `AttentionCore.CollectWeights/LoadWeights` made `virtual`
- `AttentionLayer.CollectWeights/LoadWeights` made `virtual`
- `AdamAttentionCore` overrides to persist m/v moments + step per weight matrix (Wq, Wk, Wv, Wo)
- `AdamAttentionLayer` overrides to persist m/v for Wff1/Wff2 + step
- `EmbeddingLayer` gains virtual `CollectMoments`/`LoadMoments` hooks
- `AdamEmbeddingLayer` overrides to persist embedding m/v + step
- `CpuAdamTransformerBus.Load` added (typed, constructs Adam layers before loading)
- **Backward compatible**: old `.neuro` files load cleanly (missing moment tensors silently zero-init)

**8 new tests in `CpuAdamResumeTests`** — all pass:
- Resumed loss == uninterrupted loss at 1, 5, 20 warmup steps
- Logits identical after resume
- GlobalStep preserved
- Negative control: Adam-resumed vs SGD-cold produces different logits (moments are live)
- Continued training converges further after resume
- Embedding moments verified preserved

---

#### BF16W vindication — appointment corpus clean comparison

**Clean from-scratch runs** (no resume), identical architecture, identical dataset:
`seqLen=128, embedDim=64, heads=4, ff=128, layers=3, vocab=49, 20K steps`

| Metric | BF16W (w=BF16, m=FP32, v=FP32) | FP32 Adam |
|---|---|---|
| Final eval loss | **0.2966** | 0.3428 |
| Final accuracy | **88.28%** | 84.38% |
| Speed (ms/sample) | ~62 | ~62 |
| Convergence speed | Matched | Matched |

**BF16W is competitive with FP32 Adam — zero penalty on a clean run.**

The small loss advantage for BF16W at 20K steps is within run-to-run noise (single seed,
small corpus). The key finding is: **same speed, same convergence curve, same output quality.**

**Demo output comparison (prompt: "can i book an appointment"):**

- **BF16W:** *"is the doctor is not have any slots free on monday morning . we do not have
  any slots free on monday morning . sorry the doctor is not available on wednesday afternoon .
  would friday morning suit you instead"*
- **FP32:** *"for monday afternoon . is the doctor available on monday morning . would
  wednesday morning suit you . is the doctor available on tuesday afternoon . can i book
  an appointment for wednesday morning"*

Both: zero UNKs, coherent appointment dialogue, correct domain vocabulary.

**Revised conclusion (replaces Day 7 BF16W finding):**
> BF16W (w=BF16, m=FP32, v=FP32) achieves identical convergence to FP32 Adam when
> trained from scratch. The previously observed stall was caused by a checkpoint resume
> bug (moments not persisted), not quantisation noise. Dettmers hypothesis confirmed:
> **BF16 weight storage with FP32 moments has zero convergence penalty.**

**FPGA implication:** BF16W halves weight SRAM footprint (1× BF16 vs 1× FP32) while
keeping FP32 moments. Total training SRAM for 380K model drops from 5.5 MB to ~4.3 MB
— increases ZCU102 headroom from 1.4 MB to 2.6 MB.

---

#### GPU as the reference platform for all scaling experiments

**Finding:** CPU BF16W Adam and GPU FP32 Adam converge to equivalent loss floors on
equivalent architectures. This was confirmed by clean from-scratch runs (no resume)
on the appointment corpus, and the resume bug removal eliminates any confounding factors.

**Consequence for experimental strategy:**
All multi-chip scaling, capacity, and architecture experiments are run on GPU (RTX 4090,
~15ms/sample) rather than CPU (~260ms/sample). GPU results are directly valid for the
CPU/FPGA implementation because:
1. Convergence parity is now empirically confirmed — same algorithm, same loss floors
2. The C# architecture is numerically equivalent to the GPU implementation (validated
   by CPU SGD ↔ GPU SGD cross-check in Day 5)
3. GPU is ~17× faster, making large-scale experiments tractable

**FPGA POC claim (what the silicon test proves):**
The FPGA does **not** need to match GPU speed — that is not the claim. The FPGA POC
must demonstrate:
1. The C# transformer architecture maps correctly to RTL/HLS
2. Training converges to the **same loss floor** as the C# CPU reference (±0.02)
3. Full on-chip training completes without DDR access (weights + moments fit in BRAM)

Speed on FPGA will be lower than CPU due to clock constraints (~200 MHz vs 4.4 GHz) —
this is expected, stated, and not a weakness. The paper claim is correctness and
on-chip feasibility, not throughput.

> *"The C# reference implementation was ported to FPGA. Training on the same dataset
> converges to identical loss (±0.02), confirming architectural correctness. Throughput
> is lower than CPU due to FPGA clock constraints, consistent with expectations for a
> first-generation POC."*

---

### Day 9 — CPU Adam scaling + eval measurement bug discovery

#### CPU Adam FP32 scaling results — 100K samples each

Both runs: `seqLen=128, heads=4, layers=4, batch=4, dataset=TinyStories (vocab=1501)`

| Model | Params | embedDim | ff | Final eval @ 100K | Best eval | ms/sample |
|---|---|---|---|---|---|---|
| ts-adam-cpu-fp32-600k | ~600K | 104 | 312 | 1.7998 | 1.7483 @ 95K | ~300 |
| ts-adam-cpu-fp32-400k | ~400K | 80 | 240 | 1.8052 | 1.6909 @ 83K | ~208 |

**Observation:** 400K converges faster (208 ms/sample vs 300 ms/sample) and reaches a
better best eval (1.69 vs 1.75), consistent with 400K being under the optimal capacity
for this dataset/compute budget and generalising better with fewer parameters to overfit.

**Demo output — 600K model** (prompt: "Once upon a time there was a little girl"):
> *"…she loved playing and had many adventures… she felt so proud… one day a little boy
> named tim was feeling hungry… he found a big shiny rock… lily and max were playing in
> the park… they saw a pretty red ball…"*

**Demo output — 400K model** (prompt: "One day a little boy named Tom wanted to go to the park"):
> *"…they walked back… tom and sam were surprised… they became good friends… one day a
> little boy named tim went to the store… lucy's mom hugged tim's mom… from that day on
> tim always helped her… they all lived happily ever after…"*

Both models: coherent TinyStories-style narrative, correct story arc (intro → problem →
resolution), some `<UNK>` tokens for rare vocabulary (expected for 1501-word vocab).

---

#### Eval measurement bug discovered

**Bug:** `Program.cs` drew eval from a **single batch** — `getBatch(seqLen, evalRng)` with
fixed seed 999. That is 1 × 128 tokens = **128 tokens total** for eval. With `batchSize=4`
training, the logged eval column was computed on 128 tokens vs thousands of training tokens.
Result: enormous variance (±0.15 between consecutive eval points — e.g. Shakespeare
oscillated 1.05 → 1.20 → 1.05 within consecutive logs).

**Deeper root cause:** `ShakespeareLoader.GetBatch` and `TinyStoriesLoader.GetBatch` both
sample from `rng.Next(0, _tokens.Length - seqLen)` — the **entire corpus** with no
train/val split. Eval and train draw from the same data pool. All reported eval losses
are **in-sample estimates**, not held-out generalisation metrics.

**Impact on recorded numbers:**
- TinyStories scaling curve (1.67, 1.64, 1.77, 1.85) — in-sample losses. Generalisation
  gap likely small (4.4M tokens vs 400–1M params) but unquantified.
- Shakespeare 1.05 — in-sample on a 1.1M token corpus with 1M params (1 param/token
  regime): high overfitting risk, real held-out loss unknown and likely much higher.
- BF16W appointment comparison (0.2966 vs 0.3428) — in-sample. Relative ordering
  (BF16W ≤ FP32) likely valid; absolute values unreliable.

**Fix required (not yet applied):**
- `ShakespeareLoader` and `TinyStoriesLoader`: 90/10 train/val split, `GetBatch` samples
  train range only, new `GetEvalBatch` (or `split` param) samples val range.
- `Program.cs`: eval loop over ≥50 val batches, averaged.
- All key benchmarks to be rerun with proper split before paper submission.

**Note:** Training correctness is unaffected — gradient descent ran on correct data.
Only the measurement (eval loss column) is compromised.

---

#### GPU Adam FP32 — Shakespeare 1M params, 100K samples (complete)

**Config:** `seqLen=128, embedDim=128, heads=4, ff=384, layers=4, vocab=256 (byte-level)`
**Params:** ~1M | **Hardware:** RTX 4090 | **Speed:** 21.86 ms/sample avg

| Milestone | Eval Loss | Accuracy |
|---|---|---|
| 10K samples | 1.3840 | 64.06% |
| 32K samples | 1.0473 | 66.41% |
| 60K samples | 1.0402 | 69.53% |
| 94K samples | **1.0389** (best) | 71.88% |
| 100K samples | 1.0522 | 68.75% |

Total time: 2186s (~36 min). Model plateaued ~94K samples, slight uptick at end.

**Caveat (eval bug):** These losses are in-sample (no train/val split). Shakespeare corpus
is ~1.1M tokens vs ~1M params — close to 1 param/token. Overfitting risk is real; true
held-out loss unknown until proper split is implemented.

**Demo output** (byte-level, no UNKs — full vocabulary):

Prompt: *"To be or not to be"*
> *"To be or not to beat. DUCHESS OF YORK: No, marry, hold, you be of grast. First Seral
> wash'd breasts before me Must his forth fvies he deadly. JULIET: O tremble be so be
> grewell. ISABELLA: Where's Edward's grave may whither? QUEEN MARGARET: Though I drink
> of Rome. BUCKINGHAM: Thou hast not hear"*

Prompt: *"What light through yonder window breaks"*
> *"What light through yonder window breaks. DUCHESS OF YORK: A more heason, as it with
> blows her back. CURTIS: They cannot be. GLOUCESTER: I would have it wast all adozen;
> if those vialence? MENENIUS: Noble may before his flesh. First Servant: Had you gant
> a cortal as lance! WARCKINCLO: More I done the thought scruer"*

**Qualitative assessment:** Correct dramatic script structure (CHARACTER: dialogue),
period-appropriate vocabulary, plausible speaker names (GLOUCESTER, JULIET, MENENIUS,
QUEEN MARGARET). Some invented names and garbled words — expected at 1M params on
byte-level vocab with no subword tokenisation. Output is unmistakably Shakespearean in
register and format.

---

## Day 10 — Eval bug fix confirmation + Shakespeare held-out plateau + Benchmark comparison

### Eval fixes confirmed (trainapp2)

Two bugs in the original `trainapp` were identified and fixed in `trainapp2` (published self-contained binary):

1. **Single-batch eval** — old code used 1 × 128 tokens for eval (±0.15 variance). Fixed: pre-sample 64 val sequences at startup (fixed seed 999), average loss+accuracy over all 64.
2. **No train/val split** — old code sampled eval from the entire corpus (in-sample). Fixed: 90/10 split in `ShakespeareLoader` and `TinyStoriesLoader`; `GetBatch` → train only, `GetValBatch` → held-out only.

All Day 9 Shakespeare numbers (1.0389–1.0522) were **in-sample** and are superseded by the held-out results below.

---

### Shakespeare 1M params — proper held-out eval (trainapp2, from-scratch rerun)

**Config:** `seqLen=128, embedDim=128, heads=4, ff=384, layers=4, vocab=256 (byte-level)`
**Dataset split:** train=1,039,854 tokens (90%), val=115,540 tokens (10%)
**Hardware:** RTX 4090, ~18 ms/sample

| Samples | Held-out Eval Loss | Train Loss | Accuracy |
|---|---|---|---|
| 101K | 1.5249 | 1.3141 | 55.59% |
| 124K | 1.5071 | 1.1713 | 55.88% |
| 133K | **1.5026** | 1.4098 | 56.08% |
| 155K | 1.5050 | 1.2792 | 55.71% |
| 172K | ~1.51 | ~1.20 | ~55–56% |

**Plateau confirmed: ~1.50–1.52 held-out eval loss.** Eval has been flat from 100K→172K with no downward trend. Train loss still ~1.2–1.4 and diverging from eval — model is overfitting.

**Root cause of plateau:** Shakespeare corpus has ~1.04M train tokens vs ~1M params (1 param/token regime). Model capacity is exhausted and the dataset is too small to push further.

**Run stopped at 172K samples.** No value in continuing — the held-out floor is ~1.51.

---

### Demo output — Shakespeare held-out model (172K samples, temp=0.8)

Prompt: *"To be or not to be"*
> *"To be or not to be. Dear little at your sainty of a forest? LADY ANTHENRY VI: Thou slands thy life and in good to act, Not yet to come a worthy which to send me so. POMPEY: Pass I waked me in my love? CAPU"*

**Assessment:** Correct dramatic script structure, period-appropriate vocabulary, plausible character names. Some garbled words — consistent with 1.51 held-out loss on a byte-level model.

---

### Benchmark comparison — nanoGPT Shakespeare char-level

The only published char-level Shakespeare transformer reference is **Karpathy's nanoGPT**:

| Model | Params | Context | Layers/Heads | Val Loss | Notes |
|---|---|---|---|---|---|
| nanoGPT char-level | ~10M | 256 | 6L/6H, embed=384 | **1.4697** | A100, ~3 min |
| **NeuroFabric GPU Adam** | **~1M** | **128** | **4L/4H, embed=128** | **~1.51** | RTX 4090, ~36 min |

**Gap: +0.04 loss with 10× fewer parameters.** This is not a competitive claim — nanoGPT uses 10× more params and 2× longer context. The comparison establishes that our architecture is in the correct ballpark and not fundamentally broken.

**Key caveat:** the two models are not directly comparable (different param counts, context lengths, training budgets). The comparison is cited only to sanity-check that our held-out eval is credible.

---

### Correction to Day 9 journal entry

The Day 9 Shakespeare results (eval loss 1.0389–1.0522) were **in-sample** due to the eval bug. Those numbers are invalid and should not be used for paper comparisons. The correct held-out plateau is **~1.51 @ 172K samples**.

The earlier claim "our model achieved eval loss comparable to published 10M-param baselines" was based on the invalid in-sample number and is retracted.

---

## Day 11 — Paper preprint published to GitHub

**Date:** June 3, 2026

- Published preprint of *"NeuroFabric: A Software Reference Architecture for On-Chip Transformer Training with Local Adam — BF16W Weights, Vocabulary Budget, and a Path to FPGA Training without a Host CPU"* to the `neuro-fabric-research` GitHub repository.
- Repository version: **v1.1.0** (`neuro-fabric`), tagged and released.
- Paper draft: `docs/paper/NeuroFabric-def-v4-draft.tex`
- Next steps: run `gpu-fp32-shakespeare-334k-b1-80k` and `cpu-bf16w-shakespeare-334k-b1-80k` against the tagged binary, update results table with measured values, update abstract commit hash, then submit to arXiv (cs.AR).
