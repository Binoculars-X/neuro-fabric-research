# BUG-004 — BF16W `_step` shadows base class `_step` — bias correction resets on resume

## Status
🐛 Confirmed — fix identified, not yet applied

## Discovered
01/06/26 — Day 14. Found during code review of `AdamBF16WeightsAttentionCore` while investigating cold-start warmup behaviour.

## Root Cause

`AdamAttentionCore` (base) declares `_step` as `private`:

```csharp
// AdamAttentionCore.cs
private int _step;  // incremented in base ApplyUpdate
```

`AdamBF16WeightsAttentionCore` (derived) overrides `ApplyUpdate` and declares its own `_step`:

```csharp
// AdamBF16WeightsAttentionCore.cs
private int _step;  // NEW field — shadows base, not the same variable
```

The base class `CollectWeights` / `LoadWeights` serialize and restore only the **base** `_step` field (which is always 0, because the derived `ApplyUpdate` never increments it). The derived `_step` is never saved.

On every checkpoint load, the derived `_step` resets to 0, so Adam bias correction terms `bc1 = 1 - β1^1` and `bc2 = 1 - β2^1` are used for the first update instead of the correct `bc1 = 1 - β1^(N+1)`. This produces an inflated first update (`bc1 ≈ 0.1` instead of `≈ 1.0` at large step N), corrupting weights at the start of each resumed run.

## Fix

Change `private int _step` in `AdamAttentionCore` to `protected`:

```csharp
// AdamAttentionCore.cs
protected int _step;   // was: private
```

Then remove the shadowing `private int _step` from `AdamBF16WeightsAttentionCore`.

The base `CollectWeights` already saves/loads `_step` correctly — no serialization changes needed. The derived class will share the same field and it will be persisted through the existing mechanism.

Same fix likely needed in `AdamBF16TransformerBus` path if it also overrides `ApplyUpdate` with its own `_step`.

## Impact

- **Resume runs only** — fresh runs (no `--resume`) are unaffected
- BF16W bias correction is wrong for the first `~1/( 1-Beta1 ) = 10` steps after each resume
- May produce a visible loss spike at resume boundaries in the training log
- Current clean non-resumed runs (exp003 GPU FP32, pending BF16W solo run) are **not affected**
- Does NOT explain the CPU FP32 demo garbage output (that run was not resumed)

## Files

- `src/Neuro.Attention/Adam/AdamAttentionCore.cs` — change `private int _step` → `protected int _step`
- `src/Neuro.Attention/Adam/BF16Weights/AdamBF16WeightsAttentionCore.cs` — remove shadowing `private int _step`
- Check `src/Neuro.Attention/Adam/BF16/` for same pattern
