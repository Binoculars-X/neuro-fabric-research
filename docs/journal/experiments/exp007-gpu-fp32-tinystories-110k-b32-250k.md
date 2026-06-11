# EXP-007 — TinyStories 110K params, byte-level vocab=256, 250K samples

## Goal
Find the capacity cliff: a model small enough to visibly fail at byte-level TinyStories.

## Config
| Parameter | Value |
|---|---|
| Params | ~110K |
| embedDim | 48 |
| heads | 2 (head_dim=24) |
| ff | 128 |
| layers | 4 |
| vocab | 256 (byte-level) |
| batchSize | 32 |
| samples | 250,000 |
| LR | 0.003 linear decay |
| warmup | 500 steps |
| dataset | TinyStories (22.5M tokens, 90/10 split) |
| hardware | GPU Adam FP32, RTX 4090 |

## Param estimate
`2×(256×48) + 4×(4×48² + 2×48×128) = 24,576 + 86,016 = ~110K`

## Results

| Samples | BPC | Train Loss | Eval Loss | Accuracy | ms/sample |
|---|---|---|---|---|---|
| 4,992 | 4.485 | 3.9170 | 3.1089 | 19.34% | 5.65 |
| 49,920 | 2.207 | 1.5778 | 1.5297 | 54.72% | 6.78 |
| 124,800 | 1.830 | 1.2635 | 1.2685 | 61.54% | 5.97 |
| **249,984** | **1.669** | **1.1269** | **1.1569** | **65.33%** | **6.18** |

**Total time:** 1,545s (6.18 ms/sample avg)

## Key observations
- Final eval loss **1.157, BPC 1.669** — worse than 200K (1.060) and significantly worse than 1M (0.808)
- Train/eval gap minimal — no overfitting
- 6.18 ms/sample — fastest of the three runs
- Demo output: visibly degraded — frequent nonsense words, grammar collapses mid-sentence
  - Story structure partially survives (openings, `<|endoftext|>`, some dialogue)
  - Words break mid-token: "Jeieny", "deambblan", "furgering", "danceved"
  - Some coherent phrases survive: "one little cat", "wanted to play with the felt a little dog"

## Capacity cliff verdict
| Model | Params | Eval Loss | BPC | Quality |
|---|---|---|---|---|
| EXP-005 | ~1M | 0.808 | 1.166 | Coherent stories, minor grammar errors |
| EXP-006 | ~200K | 1.060 | 1.530 | Real words, broken grammar |
| **EXP-007** | **~110K** | **1.157** | **1.669** | Frequent nonsense words, partial coherence |

**Conclusion:** 110K params is near the capacity cliff for byte-level TinyStories.
The model learns English sentence *rhythm* and *structure* but lacks capacity to reliably form real words.
~200K appears to be the practical minimum for recognisable output; ~50K would likely produce near-random bytes.

## Demo output sample (temp=0.8)
```
> once upon a time
once upon a time, "Obl heap, be laugh arrm, but feels hird! But it wanted to a big named Ap...

> one girl
one girl of the deambblan wald his friend and the saw and furgering not can push.
One day, Jeieny would felt out the happy and danceved the boat...
```
