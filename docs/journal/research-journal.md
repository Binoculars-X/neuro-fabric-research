# NeuronFabric — Research Journal
<!-- CRITICAL: max 30 lines per daily block. Date format: dd/MM/YY -->

---

## 20/05/26 — Day 1: Neuro.Core MLP Baseline

- Built full forward + backprop pipeline: `NeuronCore`, `NeuronLayer`, `NeuralBus`
- MNIST 97%+, IRIS 99%+ accuracy confirmed
- xUnit + FluentAssertions test suite (unit / integration / slow)
- CI pipeline live (GitHub Actions)

---

## 21/05/26 — Day 2: Neuro.Attention Phase 2a (Transformer)

- `EmbeddingLayer`, `AttentionCore`, `AttentionLayer`, `TransformerBus` implemented
- GeLU, weight tying, Pre-LN architecture; `.neuro` checkpoint format (Save/Load)
- `ShakespeareLoader` (byte-level, vocab=256); `Evaluate()` + `Sample()` added
- Parallel MatMul via `Parallel.For`; `TrainApp` (Ctrl+C save, `--resume`) + `DemoApp`
- 61 tests passing; 100K-step super-slow test: Shakespeare plateaus ~30–33% accuracy
- **Result:** architecture correct; 115K params hit capacity ceiling on byte-level vocab

---

## 22/05/26 — Day 3: Batch Training + CPU Optimisation + Domain LLMs

- `TransformerBus.TrainBatch()` with gradient accumulation; `--batch-size` arg
- `TinyStoriesLoader` (top-1500 vocab, download-on-demand); 20K-step slow test
- CPU optimisation: `Parallel.For` threshold-gated (>64 rows); factory override pattern
- New project `Neuro.Cpu.Optimized`; `TrainApp` uses it by default (`--no-optimize` flag)
- **Appointment corpus** (vocab=49): 20K steps → loss=0.42, accuracy=78.9%; fluent output confirmed
- **MultiWOZ** (vocab=302, 56K system turns): trained to 100K steps → loss=2.05, accuracy=50.8%
- **Key insight:** tiny vocab frees ~96K params for reasoning → fluent domain output at 100K params
- Parallel batch clone experiment tried and reverted: overhead > compute at 115K params

---

## 23/05/26 — Day 4: AVX2 + Workspace Pooling

- `AttentionCore.Forward` made `virtual`; `OptimizedAttentionCoreHeavy` added (`#if HEAVY_CORE`)
- AVX2/FMA vectorised MatMul (MulInto, MulTransposeBInto, MulTransposeAInto)
- Pre-allocated workspace arrays — zero GC allocation per step (except 2 return arrays)
- `*.neuro` added to `.gitignore`; logging changed from ms/step → ms/sample

| Implementation | ms/sample |
|---|---|
| Reference serial | 47.0 |
| Parallel.For scalar | 21.6 |
| + AVX2/FMA | 16.7 |
| + workspace pooling | **13.9** |

---

## 24/05/26 — Day 5: GPU Acceleration + 1M-Param TinyStories

- `Neuro.Gpu`: `TorchTransformerModel` (CUDA), `GpuTransformerBus` (`--gpu`), `AdamTransformerBus` (`--adam`)
- 10 tests: identical forward logits + SGD loss curves vs CPU reference
- Removed per-layer `ClipGradNorm` in backward loop — was attenuating deep-stack gradients
- **1M param TinyStories** (embedDim=128, heads=4, ff=384, layers=4, vocab=1501):
  - CPU SGD: 51 ms/sample, loss 5.47 @8k
  - GPU SGD: 11 ms/sample, loss 5.43 @8k (4.4× faster, identical convergence)
  - GPU Adam: 18 ms/sample, loss **2.71 @8k** — Adam critical for large-vocab models
- GPU Adam plateau: **1.67 @ 80K samples** ⚠️ *[INVALID — in-sample, no train/val split; see Day 9]* → TinyStories single-chip upper bound
- CPU Adam float32: 296 ms/sample, loss 2.95 @8k; BF16 moments variant: 337 ms/sample, loss 2.94
- **BF16 moments finding:** naive truncation degrades convergence — FP32 moments required
- CPU SGD to 2.18M samples: loss 2.60 → **optimizer gap: ~0.34 loss, 27× more samples**

---

## 25/05/26 — Day 6: Paper v2 Experiments + CPU SGD Plateau

- CPU SGD plateau confirmed: **1.97 @ 2.24M samples** ⚠️ *[INVALID — in-sample, no train/val split; see Day 9]* (slowly descending ~0.03/50K)
- BF16 moments (m=BF16, v=BF16) run to 43K: loss 2.75, gap vs GPU Adam widening → **design rejected**
- Revised design: **w=BF16, m=FP32, v=FP32** — 50% weight SRAM saving, zero convergence penalty
- Concurrent CPU SGD + CPU BF16 Adam on R9 9900x: no interference (both memory-bound)
- Paper experiments scoped: (1) BF16W full convergence, (2) GPU BF16 curve, (3) AVX2 benchmark, (4) demo quality

---

## 26/05/26 — Day 7: Scaling Experiments + BF16W Validation

- `CpuAdamBF16WeightsTransformerBus` (w=BF16, m=FP32, v=FP32): loss 2.87 @10K → tracking GPU Adam
- **Scaling table (GPU Adam FP32, 80K samples, TinyStories vocab=1501):**

| Params | embedDim | Floor @80K | BF16W SRAM | FP32 training SRAM |
|---|---|---|---|---|
| ~380K | 80 | ~1.85 ⚠️ | 0.76 MB | **5.5 MB ← ZCU102 fits** |
| ~590K | 104 | ~1.77 ⚠️ | 1.18 MB | 8.5 MB |
| ~800K | 96 | **~1.64** ⚠️ | 1.60 MB | 11.5 MB |
| ~1M | 128 | ~1.67 ⚠️ | 2.00 MB | 14.4 MB |

*⚠️ Floor estimates are in-sample (no train/val split); see Day 9 for retraction. Relative ordering valid, absolute values unreliable.*

- Capacity knee: **600K–800K**; above 800K no gain; 380K fits ZCU102 BRAM for full on-chip training
- CPU FP32 Adam 600K and 400K runs started (100K samples each, ~300 ms/sample)

---

## 27/05/26 — Day 8: Checkpoint Resume Bug Fix + BF16W Vindication

- **Bug:** `TransformerBus.Save` serialised weights only; on `--resume` Adam moments reset to zero
- **Impact:** Day 7 BF16W stall was entirely due to corrupted moment state, not quantisation noise
- **Fix:** `CollectWeights`/`LoadWeights` made `virtual`; `AdamAttentionCore/Layer/EmbeddingLayer` override to persist m/v/step; backward-compatible with old `.neuro` files
- 8 new tests in `CpuAdamResumeTests` — all pass
- **BF16W vindication** (clean from-scratch, appointment corpus):
  - BF16W final loss: **0.2966**, accuracy 88.3% vs FP32 Adam: 0.3428, 84.4% — competitive
  - Same speed (~62 ms/sample), same convergence curve — **zero penalty confirmed**
- GPU established as reference platform (17× faster, convergence parity confirmed)

---

## 28/05/26 — Day 9: CPU Adam Scaling + Eval Bug Discovery

- CPU Adam FP32 scaling: 600K → loss 1.75 @95K, 400K → loss 1.69 @83K (both coherent demo output)
- **Eval bug found:** `Program.cs` used 1×128 tokens for eval (±0.15 variance); no train/val split
- **Fix required:** 90/10 split in all loaders; `GetValBatch`; average over ≥50 val batches
- All prior eval numbers are in-sample estimates — relative ordering valid, absolute values unreliable
- GPU Adam Shakespeare 1M (byte-level, 100K samples): eval loss **1.0389** (in-sample, invalid)
- Demo: correct dramatic structure, period-appropriate vocabulary, plausible character names

---

## 29/05/26 — Day 10: Eval Bug Fix + Shakespeare Held-Out Plateau + nanoGPT Comparison

- Eval fixes in `trainapp2`: 64 pre-sampled val sequences (fixed seed); 90/10 train/val split in all loaders
- **Shakespeare 1M held-out plateau: ~1.51 @ 172K samples** (train ~1.2–1.4, diverging → overfitting)
  - Root cause: ~1.04M train tokens vs 1M params (1 param/token) — dataset too small
  - Day 9 in-sample number (1.039) retracted
- **nanoGPT benchmark:**

| Model | Params | Val Loss |
|---|---|---|
| nanoGPT char-level | ~10M | 1.4697 |
| **NeuronFabric GPU Adam** | **~1M** | **~1.51** |

- +0.04 loss gap with 10× fewer params — architecture in correct ballpark, not broken
- All future scaling experiments to use `trainapp2` with proper held-out eval

---

## 30/05/26 — Day 11: TinyStories GPU Adam vs Adam BF16W Convergence

**Goal:** produce clean held-out eval curves for GPU FP32 Adam and CPU BF16W Adam on the 1M param TinyStories config; confirm convergence parity with proper 90/10 split.

**Planned experiments → [experiments/EXP-001-tinystories-gpu-bf16w.md](experiments/EXP-001-tinystories-gpu-bf16w.md)**

- Config: `embedDim=80, heads=4, ff=240, layers=4` ≈ 380K params (ZCU102 FPGA target size)
- Run A: GPU FP32 Adam, 100K samples — establishes held-out baseline (expected ~1.85–1.90 floor)
- Run B: CPU BF16W Adam, 100K samples — must land within 0.05 of Run A to confirm parity
- Both use `trainapp2` (64-batch val, 90/10 split, fixed seed 999)
- Success unlocks paper Section 4 final table and BF16W silicon claim

**Codebase cleanup (paper 1 prep):**
- Splitting `neuro-fabric` (implementation) from `neuro-fabric-research` (paper + experiments)
- Research repo to hold: journal, experiment reports, paper `.tex`, result CSVs/plots
- Implementation repo to hold: C# src only, no research artefacts
- Goal: clean separation so paper codebase is citable and reproducible without training binaries

---

## 30/05/26 — Day 12: Param Scaling + BUG-001 Discovery

- EXP-002: 334K param run (embedDim=88, ff=264, layers=4) Shakespeare char-level b=1
- **BUG-001 found:** `TransformerBus.TrainBatch` calls `TrainStep` B times at `lr/B` — wrong for Adam
  - For Adam, this corrupts moment estimates (B sequential noisy updates vs 1 true batch gradient)
  - Evidence: b=1 reaches eval 1.83 at 15K samples; b=16 still at 2.13 at 15K samples (4× slower/sample)
  - All EXP-001 b=16 results valid (still converge), just suboptimal — b=1 is strictly better per sample
  - Fix: accumulate gradients over B samples, call `optimizer.step()` once at full lr
  - See → [bugs/bug-001-trainbatch-sequential-adam-steps.md](bugs/bug-001-trainbatch-sequential-adam-steps.md)
- **BUG-002 found & fixed:** linear decay resume stretches schedule denominator each resume
  - `absoluteTotalSteps = globalStepOffset + newIterations` recalculated fresh — ignores saved `TotalSteps`
  - LR jumped 0.000030 → 0.001141 after resuming 150K checkpoint with +100K samples
  - Fix: use `max(bus.TotalSteps, globalStepOffset + totalSteps)` on resume
  - See → [bugs/bug-002-linear-decay-resume-stretches-schedule.md](bugs/bug-002-linear-decay-resume-stretches-schedule.md)
- Running 334K GPU FP32 (b=1, 80K samples) vs 334K CPU BF16W (b=1, 80K samples) in parallel
- **EXP-002 preliminary results (80K samples, b=1):**
  - GPU Adam FP32: eval **1.5394** — clean Shakespeare structure, correct character names ✅
  - CPU Adam BF16W: eval **1.5375** — clean demo output, matches GPU within noise ✅
  - CPU Adam FP32: eval **1.5407** — metrics look fine, but demo output had `:s:s:s` prefix garbage ❌
  - **Conclusion:** 80K samples sufficient for GPU and BF16W; CPU FP32 checkpoint corrupt or undertrained
  - CPU FP32 re-run launched as **exp002-2** with 100K samples (terminal `a2e18fbc`)
  - GPU TinyStories 1M param run also launched (terminal `fae3bd50`)

---

## 31/05/26 — Day 13: R1.0 Release + Researcher Quick-Start Bats

- **R1.0 released** — known bugs BUG-001 and BUG-002 acknowledged but excluded from scope
  - Constraints applied: `b=1` only (sidesteps BUG-001 sequential Adam steps); no `--resume` (sidesteps BUG-002 LR schedule stretch)
  - Both bugs documented; fixes pending for R1.1
- **Researcher quick-start bat files** created in `neuro-fabric/run/`:
  - `train-gpu-adam-shakespeare.bat` — GPU Adam FP32, 334K params, 150K samples
  - `train-cpu-adam-bf16w-shakespeare.bat` — CPU BF16W, 334K params, 150K samples, `NoGpu=true` (no CUDA required)
  - `demo-gpu-adam-shakespeare.bat` — interactive demo from GPU checkpoint
  - `demo-cpu-adam-bf16w-shakespeare.bat` — interactive demo from BF16W checkpoint (no CUDA)
  - Checkpoints saved to `run/results/`; guard exits with error if checkpoint already exists
- **`NoGpu=true` build flag** added to `TrainApp.csproj` + `#if !NO_GPU` guards in `Program.cs`
  - CPU-only build excludes `Neuro.Gpu` / TorchSharp entirely — compiles on any .NET 10 machine
  - Demo app already had no GPU dependency
- **Planned R1.0 validation runs** (334K params, 150K samples, b=1, linear decay):
  - GPU Adam FP32 Shakespeare (`train-gpu-adam-shakespeare.bat`)
  - CPU Adam BF16W Shakespeare (`train-cpu-adam-bf16w-shakespeare.bat`)

---

## 01/06/26 — Day 14: TinyStories Convergence Analysis + Paper Figures

- **Shakespeare 334K canonical results** (85K samples, run/results/):
  - GPU FP32: eval **1.5281**, 56.2%, 16.4 ms/sample ← oracle
  - CPU FP32: eval **1.5425**, 55.3%, 201.5 ms/sample (+0.014 gap, 12.3× slower)
  - CPU BF16W (post bug-004/005): eval **1.5547**, best **1.5545 @ 82K**, 54.72%, 137.9 ms/sample ✅
- **BUG-004:** `_step` private in base shadowed by derived BF16W classes — bias correction resets. Fix: `private→protected` in 3 bases, remove shadow in 6 derived classes. 107/107 tests pass.
- **BUG-005:** BF16W FP32 master overwritten from BF16 decode — sub-BF16 updates lost. Fix: `w[i,j] = wf`. Confirmed by re-run.
- **BUG-003 (CPU FP32 `aW` artifact):** Investigated — model memorized bytes 97+87 as zero-context prior. GPU FP32 and BF16W unaffected. Quantitative comparison: statistically identical char distribution. CPU FP32 removed from paper.
- **Paper:** CPU FP32 row removed; TinyStories 442K section removed; `\repourl` macro added; run/ scripts listed in Reproducibility; double References fixed; DRAFT watermark added; cs.AR submission in progress (endorsement requested from Prof. Cheung).
- **BF16W demo (85K):** coherent Shakespeare dialogue, no garbage artifact. Gap vs GPU: +0.027 eval loss (1.7%).
- **exp002b created:** `exp002-cpu-adam-bf16w-shakespeare-334k-85k.md` + .neuro + .log archived.

---

## 02/06/26 — Day 15: BUG-006 Fix + Release v1.0.2 + Paper v3 + exp003 Reports

- **BUG-006 (Adam bias correction inflated):** `ApplyUpdate` called once per weight matrix per step; `_step` incremented each call → at real step 1, AttentionCore (4 matrices) had `_step=4`, giving `bc1=1−0.9^4=0.344` instead of correct 0.1. **Fix:** single `GlobalStep` counter in `TransformerBus`, incremented once per `TrainStep`, passed through `Backward(dX, lr, step)` → `ApplyUpdate(w, grad, lr, step)`. Affects 9 files: `AttentionCore`, `AttentionLayer`, `TransformerBus`, and all 6 Adam variants.
- **Build:** 0 errors post-fix; test call sites updated to pass `step: 1`.
- **GitHub release v1.0.2** created on commit `86642e2` (version.txt updated to `v1.0.2`).
- **exp003 runs (334K params, post-BUG-006 fix, b=1, 80K samples):**
  - GPU FP32: best eval **1.5226 @ 80K**, 55.79%, 19.37 ms/sample, 1550s total ✅
  - CPU BF16W: best eval **1.5477 @ 78K**, 54.57%, 149.51 ms/sample, 11961s total ✅
  - Gap: +0.025 val loss, 7.7× slower — slightly tighter than exp002 (+0.027, 8.4×)
- **exp003 reports** created: `docs/journal/experiments/exp003/` (GPU + BF16W markdown reports with raw logs and demo output).
- **Loss chart (Figure 2)** regenerated from 80K logs; "exp003" label removed from title.
- **Paper v3** (`neuronFabric-v3-draft.tex`) created and fully updated:
  - Release v1.0.1 → v1.0.2 (commit `86642e2`) throughout
  - 85K → 80K samples throughout
  - Results table: GPU 1.5226/55.79%/19.37, BF16W 1.5477/54.57%/149.51
  - Convergence gap: +0.027 → +0.025, 8.4× → 7.7×
  - Figure 2 moved to after Training curve paragraph (was before it)
  - Sample output: single MENENIUS prompt (CPU BF16W, temp 0.8) — second prompt removed
  - Demo closing: "11,961s on CPU (no GPU required)"

---

## 03/06/26 — Day 16: Pre-Publication Verification Plan

**Goal:** Before arXiv submission, independently verify every claim in the paper that could be wrong.
Four verification tracks identified:

**Track 1 — Convergence benchmarking (are our loss numbers reasonable?)**
- Compare Shakespeare eval loss vs published char-level baselines (nanoGPT, Karpathy minGPT)
- Compare TinyStories eval loss vs published word-level baselines at similar param budgets
- If our numbers are significantly better than expected → likely a bug (eval leak, wrong split)
- If significantly worse → possible Adam implementation issue
- Tools: literature search + re-run with explicit logging of train vs val split sizes

**Track 2 — Demo output coherence metrics (is "coherent" a real claim?)**
- Current evidence: subjective human review only — not acceptable for a paper
- Plan: compute perplexity on held-out Shakespeare val set from the BF16W checkpoint
- Compare character n-gram entropy vs Shakespeare ground truth distribution
- Optionally: BLEU/ROUGE vs reference passages (limited validity for char-level, but indicative)

**Track 3 — Paper formula verification**
- Re-derive Adam bias correction independently: confirm bc1/bc2 formula in code matches paper
- Verify BF16W SRAM calculation: 334K × 10 bytes = 3.34 MB — check param count from code
- Verify vocabulary-budget table: 100K params, d=64, three domains — re-run or verify from logs
- Verify inter-chip bandwidth formula: T×d×4 = 128×88×4 = 45,056 bytes

**Track 4 — FPGA arithmetic verification**
- Count FMA operations per forward pass: attention (QKV matmuls, scores, weighted sum) + FF
- Count FMA operations per backward pass
- Estimate cycles at 150–200 MHz with realistic FMA parallelism (not 600 units — verify this number)
- Verify BRAM block count: how many 36Kb BRAMs needed for 3.34 MB, check against ZCU102 spec
- Verify DSP48 utilisation estimate for BF16 multiply-accumulate

---

## 03/06/26 — Day 16 (cont.): CPU BF16W Demo Chat Output

Ran `2.demochat.bat` on the CPU BF16W Shakespeare checkpoint (v1.1.0+e9ab47a), prompt `HAMLET:`:

```
Running demo: cpu-bf16w-shakespeare-334k-b1-80k

NeuronFabric Demo v1.1.0+e9ab47ad79c442e1dc1eb0d4cc4aa61d8a127dfa  |  Apache 2.0 License  |  github.com/neuro-fabric

Loading checkpoint.neuro ...
Model ready  seqLen=128  vocabSize=256  savedWith=v1.1.0+e9ab47ad79c442e1dc1eb0d4cc4aa61d8a127dfa
Temperature=0.8  GenerateLength=300  Dataset=shakespeare

Type a prompt and press Enter to generate. Empty line = quit.
------------------------------------------------------------

> HAMLET:

HAMLET:
Break him of him; for that will for straight,
The both speak ancible o'er an to him all in.

Third Citizens:
Here it field at be that were more cames,
And then is only back'd lawly fortune our peint.
Indlow. For my name more.

DUKE VINCENTIO:
It my straight souls, but mine so rebost 's be
------------------------------------------------------------
```

- Output added to paper v5 `\textbf{Sample output}` paragraph (replacing older MENENIUS sample)

## 03/06/26 — Day 16 (cont.): neuronFabric-preprint.pdf Released on GitHub

- Released `neuronFabric-preprint.pdf` publicly on GitHub today
- PDF compiled from `neuronFabric-preprint-v5.tex` — full paper including all figures and references
- This is the pre-arXiv public release; arXiv submission pending endorsement

---

## 09/06/26 — Day 17: Byte-Level Vocab + TinyStories Scaling Experiments

**Key change: TinyStoriesLoader migrated to byte-level vocab=256 (was word-level vocab=1501)**
- `BuildVocab` / `SplitWords` / `_wordToId` removed entirely
- Tokenisation: `File.ReadAllBytes` → `int[]`; `VocabSize = 256`
- New cache file `tokens_bytes.bin` (old `tokens.bin` / `vocab.txt` ignored)
- `TinyStoriesSubsetLoader` updated identically; `Encode`/`Decode` use UTF-8 bytes
- Demo and TrainApp quality check fixed: byte tokens decoded directly (no space prefix, no UNK stats)
- UNK tracking removed from all output paths — no longer meaningful

**Motivation:** word-level vocab=1501 caused ~17.5% UNK in generation; vocab built from val-only corpus
missed rare words. Byte-level eliminates UNK entirely and reduces embedding tax:
`2×1501×120 ≈ 360K params → 2×256×120 ≈ 62K params` (saves ~300K params for 1M model)

**Scaling experiments (TinyStories, byte-level, GPU Adam FP32, b=32):**

| Exp | Params | Samples | Eval Loss | BPC | Accuracy | ms/sample |
|---|---|---|---|---|---|---|
| EXP-005 | ~1M | 500K | **0.808** | **1.166** | 75.3% | 14.7 |
| EXP-006 | ~200K | 250K | 1.060 | 1.530 | 67.6% | 6.8 |
| EXP-007 | ~110K | 250K | 1.157 | 1.669 | 65.3% | 6.2 |

- **1M model (EXP-005):** eval loss 0.808, BPC 1.166 — approaching theoretical English entropy (~1.0–1.3 bits/char)
  - Train/eval gap near-zero throughout 500K samples — no overfitting (22M tokens vs 1M params)
  - Demo output: coherent stories, minor grammar errors, no nonsense words
- **200K model (EXP-006):** BPC 1.530 — real English words, broken grammar, capacity limit visible
- **110K model (EXP-007):** BPC 1.669 — near capacity cliff; frequent nonsense words, story rhythm survives
- **Capacity cliff finding:** ~200K params is practical minimum for recognisable byte-level English output

**Build improvements:**
- `build.bat` refactored: single `dotnet restore` up front, all builds use `--no-restore`
- Prevents repeated TorchSharp-cuda-windows (~2.5 GB) restore on every build call

**Experiment reports:** [exp005](experiments/exp005-gpu-fp32-tinystories-1000k-b32-500k.md) · [exp006](experiments/exp006-gpu-fp32-tinystories-200k-b32-250k.md) · [exp007](experiments/exp007-gpu-fp32-tinystories-110k-b32-250k.md)

---

## 11/06/26 — Day 18: Exp LUT Implementation + EXP-008

- **Linear exp LUT-256 [-20,0] run completed** (EXP-008): 80K samples, eval loss **1.5383**, BPC 2.2194, 54.48% accuracy, 137 ms/sample
  - LUT-256 matched exact-exp baseline (1.5477) — **no convergence degradation** confirmed
  - Demo output quality identical to EXP-003; Shakespeare dialogue coherent ✅
  - Logged as [EXP-008](experiments/exp008-cpu-bf16w-lut256-shakespeare-334k-b1-80k.md) — marked as linear-range LUT (not hardware-standard)
- **2^n·2^f exp LUT refactor:** replaced linear range with `exp(x) = 2^floor(x·log₂e) · 2^frac(x·log₂e)`
  - Exact integer part via IEEE 754 bit manipulation; LUT covers fractional part [0,1) only
  - Correct range: all finite floats handled (no clamping to [-20,0])
- **Static Configure/Exp API:** `ExpLutHelper.Configure(size)` sets global `_activeLut`; `ExpLutHelper.Exp(x)` uses it — no per-object fields, no constructor threading
  - `_expLut` field and `expLutSize` param removed from all 8 classes (cores → layers → buses)
  - `Program.cs` calls `Configure(expLutSize)` once after arg parsing
- **21 LUT tests passing** (ExpLutHelperTests + CpuAdamBF16WeightsLutBusTests), 0 errors, 0 warnings
- **Next:** rerun with correct 2^n·2^f LUT → EXP-009

---

## 12/06/26 — Day 19: MoE Proof-of-Concept Results (EXP-010, EXP-011)

**Goal:** Prove N simple chips (MoE experts) cooperating with top-2 routing can match 1M dense BPC at ≤400K active params/token. Architecture constraint: each expert is a fully independent `AdamTransformerBus` — no merged TorchSharp model. Only token embeddings and output activations cross chip boundaries, mirroring real FPGA inter-chip communication.

**BUG-007 fixed:** `MoETransformerBus.TrainBatch` now uses gradient accumulation — each expert calls `BackwardAndStep` once per batch (not B sequential Adam steps).

**EXP-010 — MoE 4×200K @ 250K samples**
- 4 × ~200K experts (embed=64, heads=2, ff=192, layers=4) + coordinator; topK=2; b=32
- **Eval loss 0.922, BPC 1.330, accuracy 71.80%** — decisively beats single 200K (EXP-006: 1.530); just behind dense 1M at same samples (EXP-005@250K: 0.865)
- Routing did not collapse; ~22 ms/sample (serial expert loop in software)

**EXP-011 — MoE 10×100K @ 500K samples**
- 10 × ~100K experts (embed=48, heads=2, ff=144, layers=4) + coordinator; topK=2; b=32
- **BPC 1.223, eval loss 0.848, accuracy 73.99%** — beats dense 300K @ 500K (EXP-012: 0.962) by 0.114 loss
- Same ~300K active params/token as EXP-012 dense; 10× routing ratio beats 4× (EXP-010: 1.330 BPC @ 250K)
- GPU software: ~66 ms/sample (~10× slower than dense 300K) — software serialization artifact, not inherent MoE cost; on ASIC all expert tiles compute in parallel

**Key findings:**
- MoE cooperation confirmed: 10× experts at same active compute → +13.5% loss reduction vs dense 300K
- Higher routing ratio (10×) extracts more specialization than lower ratio (4×) at same active param budget
- Dense scaling plateau confirmed: 200K→300K dense gains almost nothing (EXP-006: 1.060 → EXP-012: 1.055)
- **Core thesis validated:** MoE pays routing cost once in silicon area, not in per-token energy

**FPGA XSim development started** ([FEAT-001](plan/feat-001-fpga-xsim-development-pipeline.md))
- Bottom-up RTL pipeline: BF16 MAC → exp LUT → matmul → attention → MLP → LayerNorm → Adam
- Steps 1, 1b, 2, 2b complete and passing Vivado XSim simulation:
  - `bf16_mac.sv` — 3-stage BF16×BF16 MAC
  - `exp_lut.sv` — 4-stage pipelined 256-entry BRAM LUT (`2^n·2^f`)
  - `bf16_matmul.sv` — 4×4×4 BF16×BF16 matmul with K=4 adder tree
  - `bf16w_matmul.sv` — 4×4×4 FP32×BF16 matmul (bf16w training path, maps to `AdamBF16WeightsAttentionCore`)
- C# `Neuro.Attention.XSim.LocalTests` project: generates hex test vectors from C# reference, invokes xvlog/xelab/xsim automatically, checks pass/fail as xUnit tests
- Verification: **1 ULP tolerance** using `ReferenceExactHardwareMode` (`(float)((double)x op (double)y)`) to match XSim's shortreal→double promotion artifact

