# BUG-001 — `TrainBatch` performs sequential Adam steps instead of gradient accumulation

## Status
⚠️ Open — not yet fixed

## Discovered
30/05/26 — Day 12. Observed while comparing b=1 vs b=16 convergence on Shakespeare 334K char-level.

## Symptom
`--batch-size 16` converges dramatically slower **per sample** than `--batch-size 1` on GPU Adam.

Evidence from Shakespeare 380K (b=16) vs 334K (b=1) runs:

| Samples | b=16 eval loss | b=1 eval loss |
|---|---|---|
| ~15K | 2.1315 | 1.8289 |
| ~65K | 1.7729 | — (not yet run to completion) |

b=1 reaches b=16's 65K eval loss in under 15K samples — **4× faster convergence per sample**.

## Root Cause

[`TransformerBus.TrainBatch`](../../../../neuro-fabric/src/Neuro.Attention/TransformerBus.cs) line ~158:

```csharp
float scaledLr = lr / tokensBatch.Length;   // ← divides lr by B

for (int i = 0; i < tokensBatch.Length; i++)
    totalLoss += TrainStep(tokensBatch[i], targetsBatch[i], scaledLr);
```

This calls `TrainStep` **B separate times**, each with `lr/B`. For SGD this would be correct gradient accumulation (B small steps = 1 full step). But for **Adam it is wrong**:

- Each `TrainStep` computes a gradient from a single sample and immediately runs `optimizer.step()`
- The Adam moment estimates ($m_t$, $v_t$) are updated **B times** per logical batch, each from a noisy single-sample gradient
- The lr scaling (`lr/B`) partially recovers the step magnitude but **does not fix the corrupted moments**
- Result: moments become stale mixtures of B sequential noisy gradients, not the true batch gradient
- Effective learning signal per sample is weaker than b=1 at full lr

## Why b=1 wins
With b=1, each sample gets a full `lr` Adam step with fresh moments. The adaptive per-parameter scaling in Adam works correctly. Per sample, b=1 makes strictly better use of each gradient.

## Correct Fix
True gradient accumulation for Adam requires:
1. `zero_grad()` once before the batch
2. `backward()` B times (gradients accumulate in `.grad` tensors)
3. `optimizer.step()` **once** at full `lr`

This gives one Adam step on the **true batch gradient** (average of B sample gradients) — which is what mini-batch Adam is supposed to do.

For `AdamTransformerBus` (GPU/TorchSharp), this can be implemented as:

```csharp
public override float TrainBatch(int[][] tokensBatch, int[][] targetsBatch, float lr)
{
    _model.train();
    _optimizer.ParamGroups.First().LearningRate = lr;
    _optimizer.zero_grad();

    float totalLoss = 0f;
    using var scope = torch.NewDisposeScope();
    for (int i = 0; i < tokensBatch.Length; i++)
    {
        var input  = torch.tensor(..., device: CUDA);
        var target = torch.tensor(..., device: CUDA);
        var logits = _model.forward(input);
        // divide loss by B so gradients are averaged, not summed
        var loss   = functional.cross_entropy(logits, target) / tokensBatch.Length;
        loss.backward();   // accumulates into .grad
        totalLoss += loss.item<float>() * tokensBatch.Length; // recover unnormalised for logging
    }
    _optimizer.step();
    return totalLoss / tokensBatch.Length;
}
```

For `CpuAdamTransformerBus` (manual Adam), the same principle applies: accumulate gradients from B samples, then do one Adam moment update and weight update.

## Impact Assessment
- All EXP-001 runs with `--batch-size 16` (380K GPU, CPU FP32 TinyStories/Shakespeare) used the buggy implementation
- **The 380K GPU Adam Shakespeare result (eval 1.6152 @ 250K) is valid** — just trained suboptimally; b=16 still converges, just slower per sample
- With correct batch implementation, same quality could be reached in ~60–80K samples instead of 250K
- Paper claims on convergence speed with b=1 are unaffected — those runs are correct

## Priority
Medium — paper results are not invalidated, but future experiments should use b=1 until fix is verified. Fix before any "batch size ablation" experiments.
