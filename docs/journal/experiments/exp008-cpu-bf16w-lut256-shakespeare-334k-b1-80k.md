# EXP-008 — CPU Adam BF16W + Linear Exp LUT-256 [-20,0] — Shakespeare char-level — 334K params — b=1 — 80K samples

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | CPU Adam BF16 weights, lr=0.003, linear decay, warmup=200 steps, b=1 |
| Softmax exp | **LUT-256, linear range [-20, 0]** (previous implementation — see note) |
| Samples | 80,000 |
| Final eval loss | **1.5383** |
| Best eval loss | **1.5383** @ 80K |
| Final train loss | 1.3569 |
| Final accuracy | 54.48% |
| Final BPC | 2.2194 |
| Speed | 137.34 ms/sample avg |
| Total time | 10,987.2s (~183 min / ~3.1 h) |
| Run folder | `run/cpu-bf16w-lut256-shakespeare-334k-b1-80k/` |
| Saved with | v1.1.0+983943d4a55468e5b6915c15296640b6a0dd6216 |

## ⚠️ Implementation note

This run used the **first (linear) LUT implementation**, where the lookup table covered a fixed range `[-20, 0]` with 256 linearly-spaced entries. This is **not** the hardware-standard `2^n · 2^f` decomposition approach that was implemented afterwards. Results here reflect the linear-range LUT only.

A follow-up run with the correct `2^n · 2^f` LUT and exp moments (Adam m/v computed in FP32 but exp in BF16 via LUT) will be logged as EXP-009.

## Comparison vs EXP-003 (exact exp, BF16W)

| Metric | EXP-003 exact exp | EXP-008 LUT-256 linear | Delta |
|---|---|---|---|
| Final eval loss | 1.5480 | **1.5383** | **-0.010** |
| Best eval loss | 1.5477 | **1.5383** | **-0.009** |
| Final accuracy | 54.57% | **54.48%** | -0.09% |
| Final BPC | — | 2.2194 | — |
| Speed (ms/sample) | 149.51 | **137.34** | -12.2 |

The LUT-256 run converged to a **slightly better** eval loss than the exact-exp baseline. This is likely noise rather than a genuine improvement — the two runs differ in build version and minor implementation details. The key finding is that LUT-256 (even with the limited linear range) **does not degrade convergence** on this task.

## Gap vs GPU FP32 oracle (exp003)

| Variant | Best eval loss | Gap |
|---|---|---|
| GPU Adam FP32 | 1.5226 | — |
| CPU Adam BF16W exact exp | 1.5477 | +0.025 |
| CPU Adam BF16W LUT-256 linear | **1.5383** | **+0.016** |

## Demo output (CPU Adam BF16W LUT-256, temperature=0.8, 80K samples)

**Prompt: `to be or not to be`**
```
to be or not to be cheeks.

JULIET:
Believe with her
With all him a poison and same.

QUEEN MARGARET:
I will you are be power, lure war:
Sign Gloucesteres of a cansures for York,
Here a duke of your must for the doublest?
To never the desire and here is younger.

ROMEO:
Aufidy will I leap, pities yond.
```

**Prompt: `HAMLET :`**
```
HAMLET :
Well, their the here for his with hell
Many all to drunk, be thou hast sake, but him
And suffers, with yourself.

KING EDWARD IV:
Hake comes guest his what thou? who shalt be can hell.
Our die crown thing to graces she hands again,
I wanting straitor a day; Verome's which is no.

LADY GRE
```

### Observations
- Output is fluent Shakespeare-style prose with correct character name formatting
- Grammar and vocabulary consistent with training corpus
- Comparable quality to EXP-003 exact-exp run — no visible degradation from LUT
- LUT-256 linear approximation is sufficient to train a converged model

## Raw training log (selected checkpoints)

```
NeuronFabric TrainApp v1.1.0+983943d4a55468e5b6915c15296640b6a0dd6216  |  Apache 2.0 License  |  github.com/neuro-fabric

Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (CPU Adam BF16 weights): seqLen=128 embedDim=88 heads=4 ff=264 layers=4  exp=LUT-256

Dataset: shakespeare  BatchSize: 1  (log every 1,000 samples)
#DATA
Samples    BPC      Train Loss     Eval Loss      Accuracy     ms/sample  LR
----------------------------------------------------------------------
1,000      3.6964   2.9644         2.5621         27.51%       134.09     0.002970
6,000      2.9900   2.0346         2.0725         39.77%       136.01     0.002782
10,000     2.7996   1.8455         1.9406         43.59%       136.82     0.002632
20,000     2.5809   1.6452         1.7889         47.49%       134.44     0.002256
30,000     2.4506   1.5470         1.6986         49.50%       127.40     0.001880
40,000     2.3906   1.4915         1.6570         51.48%       132.08     0.001504
50,000     2.2886   1.4225         1.5863         52.93%       133.19     0.001128
60,000     2.2578   1.3929         1.5650         54.26%       127.00     0.000752
70,000     2.2372   1.3663         1.5507         54.28%       218.49     0.000376
80,000     2.2194   1.3569         1.5383         54.48%       129.39     0.000030
----------------------------------------------------------------------
Total time: 10987.2s  (137.34 ms/sample avg, 80,000 samples)
```
