# BUG-005 — BF16W FP32 master weight overwritten from BF16 decode — sub-BF16 updates lost

## Status
✅ Fixed — 01/06/26

## Discovered
01/06/26 — Day 14. Code review of `AdamBF16WeightsAttentionCore` and `AdamBF16WeightsAttentionLayer`.

## Root Cause

In both BF16W `ApplyUpdate` implementations, after computing the weight update in FP32 (`wf`), the FP32 working copy was written back by **re-decoding from BF16**:

```csharp
// BUG — before fix
float wf    = Bf16.Decode(wBf16[i, j]);
wf         -= lr * mHat / (MathF.Sqrt(vHat) + Epsilon);
wBf16[i, j] = Bf16.Encode(wf);
w[i, j]     = Bf16.Decode(wBf16[i, j]);  // ← discards sub-BF16 precision
```

`Bf16.Encode → Bf16.Decode` is a lossy round-trip (BF16 has ~7 mantissa bits ≈ 0.8% relative precision). Any update smaller than BF16's precision floor for that weight magnitude is silently rounded away, and the re-decode propagates that loss into `w[i,j]` — the FP32 copy used by the forward pass.

During **LR warmup**, `effectiveLr` starts near zero (`lr × step/warmupSteps`). Many gradient steps produce updates smaller than BF16 precision, so `w[i,j]` does not accumulate them. FP32 Adam (no encode/decode) accumulates all updates correctly → diverging convergence between variants.

## Fix

Use `wf` (the FP32 result) directly as the master:

```csharp
// FIXED
float wf    = Bf16.Decode(wBf16[i, j]);
wf         -= lr * mHat / (MathF.Sqrt(vHat) + Epsilon);
wBf16[i, j] = Bf16.Encode(wf);  // SRAM / checkpoint copy stays BF16
w[i, j]     = wf;               // FP32 master accumulates at full precision
```

`wBf16` is the SRAM representation (correct for FPGA target); `w` is the runtime forward-pass master.

## Files Changed

- `src/Neuro.Attention/Adam/BF16Weights/AdamBF16WeightsAttentionCore.cs` — line ~76
- `src/Neuro.Attention/Adam/BF16Weights/AdamBF16WeightsAttentionLayer.cs` — line ~60

## Tests

107/107 passed after fix (`dotnet test Neuro.Attention.Tests`).

## Impact on Results

- All prior BF16W runs used the buggy implementation
- The solo BF16W re-run (in progress) was started before this fix → result will still reflect buggy behaviour
- After the fix, BF16W convergence should improve, particularly during warmup
- A new clean BF16W run after this fix is needed for valid paper numbers
- **01/06/26 — confirmed:** re-run after fix shows visibly better convergence ✅ bug was real
