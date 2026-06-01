# BUG-006 — Adam bias correction uses per-matrix step count — `_step` inflated 4-6× real training step

## Status
✅ Fixed — release v1.0.2 (commit `86642e2`, 02/06/26)

## Discovered
01/06/26 — Day 14. Code review during investigation of bug-003 (CPU FP32 demo noise).

## Fixed
02/06/26 — Day 15. Option A implemented: `GlobalStep` counter in `TransformerBus` incremented once per `TrainStep`, passed through `Backward(dX, lr, step)` → `ApplyUpdate(w, grad, lr, step)`. `_step++` removed from all 6 Adam `ApplyUpdate` overrides.

## Root Cause

`ApplyUpdate(w, grad, lr)` is called **once per weight matrix** per training step. Each call increments `_step`. But `_step` is used for Adam bias correction:

```csharp
float bc1 = 1f - MathF.Pow(Beta1, _step);
float bc2 = 1f - MathF.Pow(Beta2, _step);
```

Call counts per real training step:
- `AdamAttentionCore._step`: called for Wq, Wk, Wv, Wo → `_step` grows **4×** per real step
- `AdamAttentionLayer._step`: called for Wff1, Wff2 → `_step` grows **2×** per real step
- `AdamEmbeddingLayer._step`: called once via `UpdateWeights` → **correct** ✓

After N real training steps:
- `bc1` for attention core = `1 - 0.9^(4N)` instead of `1 - 0.9^N`
- At step 1: `bc1 = 0.344` instead of `0.1` — bias correction under-applied by 3.4×
- At large N: both converge to 1.0 so the error diminishes over time

## Impact

- Bias correction is wrong throughout training, worst in early steps
- Affects **all variants equally** (FP32, BF16W, BF16) — cannot explain CPU FP32 vs BF16W demo quality difference
- Adam effective update magnitude is different from correct Adam — convergence is suboptimal but not broken
- All current results were produced with this bug — fixing it will change convergence behaviour

## Fix

Pass the global training step from `TransformerBus.TrainStep` down to `ApplyUpdate` instead of tracking it locally per weight matrix.

**Option A — pass globalStep as parameter (cleanest):**

Change signature:
```csharp
protected virtual void ApplyUpdate(float[,] w, float[,] grad, float lr, int step)
```

Remove `_step++` from all `ApplyUpdate` overrides. In `TransformerBus`, pass `bus.GlobalStep` (already tracked) when calling backward/update.

**Option B — increment once per layer update cycle:**

Add a `BeginStep()` method called once per training step that increments `_step`, and remove `_step++` from `ApplyUpdate`. `TransformerBus.TrainStep` calls `BeginStep()` on each layer before the update loop.

Option A is cleaner — `GlobalStep` already exists on `TransformerBus` and is correct.

## Files Affected

- `src/Neuro.Attention/Adam/AdamAttentionCore.cs` — remove `_step++`, use passed step
- `src/Neuro.Attention/Adam/AdamAttentionLayer.cs` — same
- `src/Neuro.Attention/Adam/BF16/AdamBF16AttentionCore.cs` — same
- `src/Neuro.Attention/Adam/BF16/AdamBF16AttentionLayer.cs` — same
- `src/Neuro.Attention/Adam/BF16Weights/AdamBF16WeightsAttentionCore.cs` — same
- `src/Neuro.Attention/Adam/BF16Weights/AdamBF16WeightsAttentionLayer.cs` — same
- `src/Neuro.Attention/AttentionLayer.cs` / `AttentionCore.cs` — update `ApplyUpdate` signature
- `src/Neuro.Attention/TransformerBus.cs` — pass `GlobalStep` at call site

## Note

`AdamEmbeddingLayer.UpdateWeights` is called once per step and is **correct** — no change needed there.
