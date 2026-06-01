# BUG-003 — CPU FP32 demo produces garbage output; GPU and BF16W unaffected

## Status
⚠️ Open — root cause not yet confirmed

## Discovered
31/05/26 — Day 13. Observed when running demo on CPU FP32 Shakespeare 334K checkpoint (exp002-2, 100K samples).

## Symptom
Same prompt tested on all three variants. GPU and BF16W produce coherent Shakespeare-style text. CPU FP32 produces garbage for the first ~50–100 characters before (sometimes) recovering.

| Variant | Samples | Eval Loss | Demo output (same prompt) |
|---|---|---|---|
| GPU Adam FP32 | 80K | 1.5394 | ✅ coherent |
| CPU BF16W | 80K | 1.5375 | ✅ coherent |
| CPU FP32 | 100K | 1.5269 | ❌ garbage |

The FP32 CPU run has **lower eval loss** (better trained by metrics), yet produces worse demo output. This rules out under-training as the cause.

## Root Cause Hypotheses

1. **Overfitting (most likely)** — CPU FP32 has the lowest eval loss (1.5269) yet the worst demo output. This is a classic overfitting signature: the model has memorised mid-sequence patterns at the expense of generalisation. FP32 has no quantisation noise, which means no implicit regularisation. BF16W's weight rounding acts as accidental regularisation that prevents this.
2. **FP32-specific bug in CPU Adam path** — a bug in gradient accumulation, moment update, or LR scaling that only manifests in full-precision. BF16W masks it via quantisation. Could interact with BUG-001 (sequential Adam steps).
3. **Checkpoint corruption** — unlikely; eval loss is clean and monotonically decreasing.
4. **BOS / cold-start** — not the primary cause; GPU and BF16W both lack BOS yet demo fine.

Most likely: **FP32 overfits without the regularising effect of BF16W quantisation noise**, possibly compounded by a CPU Adam bug.

## Impact

- CPU FP32 demo output is not usable for paper 1
- Eval loss metrics for CPU FP32 are **valid** — loss is computed mid-sequence
- Paper 1 table (GPU vs BF16W) is **unaffected**

## Investigation Needed

- Inspect logit distribution of first generated token across all three checkpoints
- Check if CPU Adam moment buffers (m, v) have abnormal magnitudes in FP32 vs BF16W
- Re-run CPU FP32 with a seeded multi-word prompt to see if recovery occurs

## Workaround (paper 1)

Exclude CPU FP32 from paper 1 demo results. Use GPU FP32 as baseline and CPU BF16W as the efficiency result. The key claim (BF16W ≈ FP32 quality) holds.

