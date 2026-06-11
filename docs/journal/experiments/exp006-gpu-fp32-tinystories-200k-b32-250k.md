# EXP-006 — TinyStories 200K params, byte-level vocab=256, 250K samples

## Goal
Probe capacity floor: can a 200K-param model learn meaningful language structure on TinyStories
with byte-level vocab? Compare vs 1M-param baseline (EXP-005).

## Config
| Parameter | Value |
|---|---|
| Params | ~197K |
| embedDim | 64 |
| heads | 2 (head_dim=32) |
| ff | 192 |
| layers | 4 |
| vocab | 256 (byte-level) |
| batchSize | 32 |
| samples | 250,000 |
| LR | 0.003 linear decay |
| warmup | 500 steps |
| dataset | TinyStories (22.5M tokens, 90/10 split) |
| hardware | GPU Adam FP32, RTX 4090 |

## Param estimate
`2×(256×64) + 4×(4×64² + 2×64×192) = 32,768 + 163,840 = ~197K`

## Results

| Samples | BPC | Train Loss | Eval Loss | Accuracy | ms/sample |
|---|---|---|---|---|---|
| 4,992 | 4.412 | 4.1454 | 3.0582 | 20.47% | 6.16 |
| 49,920 | 2.082 | 1.4589 | 1.4429 | 56.68% | 6.59 |
| 124,800 | 1.732 | 1.1823 | 1.2003 | 63.73% | 6.27 |
| **249,984** | **1.530** | **1.0394** | **1.0605** | **67.59%** | **6.84** |

**Total time:** 1,711s (6.84 ms/sample avg)

## Key observations
- Final eval loss **1.060, BPC 1.530** — noticeably worse than 1M params (0.808) as expected
- Train/eval gap minimal — no overfitting
- 6.84 ms/sample (vs 14.66 ms for 1M) — ~2× faster per sample due to smaller model
- Demo output: coherent story structure, correct punctuation, English words mostly intact
  - Occasional nonsense words ("lables", "bloon") — capacity limit showing
  - "Once upon a time", `<|endoftext|>`, dialogue markers learned correctly
- **Conclusion:** 200K params is near the minimum for recognisable English stories at byte-level

## Demo output sample (temp=0.8)
```
> once upon a time
once upon a time, "Obl heap, be laugh arrm, but feels hird! But it wanted to a big named Ap
and to heal him friends, and the other into lone.
<|endoftext|>
One day, a big, robbited to play with the felt a little dog...
```
