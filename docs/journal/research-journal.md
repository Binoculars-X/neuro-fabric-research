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
