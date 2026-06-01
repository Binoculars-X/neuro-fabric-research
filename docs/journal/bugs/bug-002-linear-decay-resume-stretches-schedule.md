# BUG-002 — Linear decay resume stretches schedule denominator on each resume

## Status
✅ Fixed in `Program.cs` — 30/05/26

## Discovered
30/05/26 — Day 12. Observed when resuming GPU Adam TinyStories 377K from 150K checkpoint with `--resume` + 100K more samples.

## Symptom
After resume, LR jumped from the final decayed value back to a mid-schedule value:

```
150,000    2.4799    2.5314    41.15%    19.88    0.000030   ← end of first run
155,000    2.6135    2.6778    39.22%    16.79    0.001141   ← after resume (+100K)
```

LR went from `0.000030` → `0.001141` — effectively restarting decay from 62% progress. Train loss also jumped from 2.48 → 2.61.

## Root Cause

`Program.cs` line ~192 (before fix):

```csharp
int absoluteTotalSteps = globalStepOffset + totalSteps;
```

On first run (150K samples, step offset=0): `absoluteTotalSteps = 0 + 150000 = 150000`  
On resume (+100K): `absoluteTotalSteps = 150000 + 100000 = 250000`

The decay denominator expanded from 150K to 250K. At step 155K:
```
progress = (155000 - 200) / (250000 - 200) ≈ 0.62
lr = 0.003 × (1 - 0.62) = 0.00114
```

`TotalSteps` **is** saved to the checkpoint and loaded on resume — but the code never used it. It recalculated `absoluteTotalSteps` from scratch every run.

## The Spec Claim That Was Wrong

[`LR-DECAY-SPEC.md`](../../../../neuro-fabric/docs/LR-DECAY-SPEC.md) stated:

> *"On resume, `GlobalStep` is restored from the checkpoint. The schedule recomputes `effective_lr` from the resumed step position — no manual `--lr` adjustment needed."*

This is only true if you pass the **original total iterations** on resume. Passing `remaining_budget` (the natural/intuitive thing) silently stretches the schedule.

## Fix

```csharp
// Before:
int absoluteTotalSteps = globalStepOffset + totalSteps;

// After:
int absoluteTotalSteps = (resume && bus.TotalSteps > 0)
    ? Math.Max(bus.TotalSteps, globalStepOffset + totalSteps)
    : globalStepOffset + totalSteps;
```

On resume, use the saved `TotalSteps` as the denominator. If extending beyond the original plan (`globalStepOffset + totalSteps > bus.TotalSteps`), take the max so LR never jumps upward — the schedule continues decaying from wherever it was.

## Behaviour After Fix

| Scenario | absoluteTotalSteps |
|---|---|
| Fresh run, 150K samples | 150,000 |
| Resume +100K (within original plan) | 150,000 (saved) — LR continues from 0.000030 ✓ |
| Resume +100K (extending beyond 150K) | 250,000 (max) — LR restarts decay gently ✓ |

## Impact
- The bad resume checkpoint (`exp001-gpu-adam-tinystories-377k.neuro` at 250K steps with broken LR) should be discarded and rerun fresh from 250K
- All fresh (non-resumed) runs are unaffected
- CPU runs (BF16W, FP32 Shakespeare) currently running are unaffected (no resume used)
