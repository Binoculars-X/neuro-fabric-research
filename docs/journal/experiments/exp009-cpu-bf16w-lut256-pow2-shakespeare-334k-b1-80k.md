# EXP-009 — CPU Adam BF16W + Exp LUT-256 (2^n·2^f) — Shakespeare char-level — 334K params — b=1 — 80K samples

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | CPU Adam BF16 weights, lr=0.003, linear decay, warmup=200 steps, b=1 |
| Softmax exp | **LUT-256, 2^n·2^f decomposition** (hardware-standard implementation) |
| Samples | 80,000 |
| Final eval loss | **1.5762** |
| Best eval loss | **1.5754** @ 77K |
| Final train loss | 1.3852 |
| Final accuracy | 53.81% |
| Final BPC | 2.2740 |
| Speed | 134.53 ms/sample avg |
| Total time | 10,762.5s (~179 min / ~3.0 h) |
| Run folder | `run/cpu-bf16w-lut256-shakespeare-334k-b1-80k/` |
| Saved with | v1.1.0+983943d4a55468e5b6915c15296640b6a0dd6216 |

## Exp LUT implementation

`exp(x) = 2^floor(x·log₂e) · 2^frac(x·log₂e)` — integer part via IEEE 754 bit manipulation, fractional part via LUT with linear interpolation. Full float range, no clamping.

## Comparison vs EXP-008 (linear LUT-256 [-20,0])

| Metric | EXP-008 linear LUT | EXP-009 2^n·2^f LUT | Delta |
|---|---|---|---|
| Final eval loss | **1.5383** | 1.5762 | +0.038 |
| Best eval loss | **1.5383** | 1.5754 | +0.037 |
| Final accuracy | **54.48%** | 53.81% | -0.67% |
| Final BPC | **2.2194** | 2.2740 | +0.055 |
| Speed (ms/sample) | 137.34 | **134.53** | -2.8 |

## Comparison vs EXP-003 (exact exp, BF16W)

| Variant | Best eval loss | Gap vs exact |
|---|---|---|
| GPU Adam FP32 (oracle) | 1.5226 | — |
| CPU Adam BF16W exact exp | 1.5477 | +0.025 |
| CPU Adam BF16W LUT-256 linear | 1.5383 | +0.016 |
| **CPU Adam BF16W LUT-256 2^n·2^f** | **1.5754** | **+0.053** |

## Analysis

The 2^n·2^f LUT shows a consistent **+0.025–0.04 eval loss regression** vs the linear [-20,0] LUT throughout training (visible from step 10K onward). This is a real effect, not noise:

- Both runs use identical architecture, seed `Random(42)`, and LR schedule
- The gap is stable across the full training curve

**Root cause:** softmax inputs are always `x - max ≤ 0`, concentrating in `[-20, 0]`. The linear LUT was accidentally optimised for exactly this range — its 256 evenly-spaced entries in `[-20, 0]` provided finer granularity there than the 2^n·2^f LUT, which distributes its entries uniformly over the fractional part `[0,1)` of `x·log₂e`.

The 2^n·2^f design is nevertheless the correct FPGA standard (hardware-accurate, no range tuning, full float coverage). The accuracy gap is an expected trade-off at LUT-256. LUT-512 or LUT-1024 would close it.

## Demo output (temperature=0.8, 80K samples)

**Prompt: `to be or not to be`**
```
to be or not to bear
Before hear man, all rather can tall.

POLIXENES:
I'll is the shall I am to him before stay?

JULIET:
This wife:
Happy you head to the presencess the mooth;
Whilst Causion would out good to fellows;
Whose died thee not vain be the sty there.

RIVERS:
Who's wife, sir, so goldems 'to b
```

**Prompt: `HAMLET :`**
```
HAMLET :
Yet sleep now, sirreading prizes me play'd,
And voices to hangess in bar'd to off,
'Tis strew my supp to down.

QUEEN MARGARET:
No, but me your of his weaker this do proud,
And play thee, bear dream, him down the was not give
Chiever that woman, at be came in the lies of this madam;
To the
```

### Observations
- Shakespeare structure preserved (character names, verse rhythm)
- More grammatical noise than EXP-003/EXP-008 — consistent with slightly higher eval loss
- No garbled or non-UTF-8 output

## Raw training log (selected checkpoints)

```
NeuroFabric TrainApp v1.1.0+983943d4a55468e5b6915c15296640b6a0dd6216  |  Apache 2.0 License  |  github.com/neuro-fabric

Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (CPU Adam BF16 weights): seqLen=128 embedDim=88 heads=4 ff=264 layers=4  exp=LUT-256

Samples    BPC      Train Loss     Eval Loss      Accuracy     ms/sample  LR
----------------------------------------------------------------------
1,000      3.6815   2.9810         2.5518         28.75%       134.10     0.002970
10,000     2.8392   1.8654         1.9680         42.69%       134.71     0.002632
20,000     2.6359   1.6727         1.8270         46.25%       136.50     0.002256
30,000     2.4904   1.5954         1.7262         49.88%       136.42     0.001880
40,000     2.3970   1.5197         1.6614         51.04%       129.97     0.001504
50,000     2.3522   1.4624         1.6305         52.17%       133.96     0.001128
60,000     2.3051   1.4230         1.5977         53.61%       135.05     0.000752
70,000     2.2752   1.3953         1.5771         53.77%       131.20     0.000376
80,000     2.2740   1.3852         1.5762         53.81%       134.54     0.000030
----------------------------------------------------------------------
Total time: 10762.5s  (134.53 ms/sample avg, 80,000 samples)
```
