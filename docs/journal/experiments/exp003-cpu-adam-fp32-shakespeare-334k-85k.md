# EXP-003 — CPU Adam FP32 — Shakespeare char-level — 334K params — b=1 — 85K samples

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | CPU Adam FP32, lr=0.003, linear decay, warmup=200 steps, b=1 |
| Samples | 85,000 |
| Final eval loss | **1.5425** |
| Best eval loss | **1.5411** @ 84K |
| Final train loss | 1.3075 |
| Final accuracy | 55.32% |
| Speed | 201.50 ms/sample avg (single-threaded CPU) |
| Total time | 17127.7s (~4h 45m) |
| Checkpoint | run/results/exp-cpu-adam-fp32-shakespeare-334k-85k.neuro |
| savedWith | v1.0.1+b4478463854f94e6f47694ddfc13afe1546a6488 |

## Purpose

Solo clean run to establish CPU FP32 as the platform-isolation baseline:
- GPU vs CPU FP32 gap isolates **platform cost** (no precision change)
- CPU FP32 vs CPU BF16W gap isolates **BF16W precision cost** (same platform)

## Log (selected rows)

```
Samples    Train Loss     Eval Loss      Accuracy     ms/sample  LR
--------------------------------------------------------------
1,000      2.9776         2.5413         26.87%       124.74     0.002972
2,000      2.4582         2.3829         31.51%       127.30     0.002936
59,000     1.4084         1.5990         53.41%       205.50     0.000920
60,000     1.3910         1.5954         52.93%       205.35     0.000884
61,000     1.4007         1.5874         53.66%       209.08     0.000849
62,000     1.3975         1.6139         53.16%       210.26     0.000814
63,000     1.3839         1.5810         53.72%       212.91     0.000778
64,000     1.3844         1.5781         53.78%       224.32     0.000743
65,000     1.3739         1.5893         53.89%       213.10     0.000708
66,000     1.3747         1.5870         54.03%       206.70     0.000672
67,000     1.3685         1.5972         53.31%       206.27     0.000637
68,000     1.3732         1.5809         54.53%       196.35     0.000601
69,000     1.3646         1.5995         53.96%       211.76     0.000566
70,000     1.3641         1.5712         54.22%       207.06     0.000531
71,000     1.3521         1.5730         54.15%       208.09     0.000495
72,000     1.3448         1.5630         54.77%       207.92     0.000460
73,000     1.3533         1.5568         54.60%       210.32     0.000425
74,000     1.3516         1.5559         54.50%       209.58     0.000389
75,000     1.3346         1.5543         54.83%       210.01     0.000354
76,000     1.3383         1.5520         54.65%       211.06     0.000318
77,000     1.3415         1.5516         54.93%       230.58     0.000283
78,000     1.3251         1.5475         55.30%       208.53     0.000248
79,000     1.3227         1.5420         55.36%       208.67     0.000212
80,000     1.3227         1.5432         55.08%       208.23     0.000177
81,000     1.3274         1.5470         55.30%       209.09     0.000142
82,000     1.3120         1.5424         55.40%       206.87     0.000106
83,000     1.3157         1.5439         55.36%       207.42     0.000071
84,000     1.3183         1.5411         55.29%       209.62     0.000035  ← best eval
85,000     1.3075         1.5425         55.32%       211.33     0.000030
--------------------------------------------------------------
Total time: 17127.7s  (201.50 ms/sample avg, 85,000 samples)
Checkpoint saved: run/results/exp-cpu-adam-fp32-shakespeare-334k-85k.neuro
```

## Validation — Demo Output

Run: `demo-cpu-adam-fp32-shakespeare.bat`
Version: `v1.0.1+b4478463854f94e6f47694ddfc13afe1546a6488`
Settings: Temperature=0.8, GenerateLength=300, Dataset=shakespeare

**Prompt: `MENENIUS:`**
```
MENENIUS:aW
AFight and do the art,
Astee his lik'd Pruden, Waneless Cal,
JOsing ta Rodure: Go, sir, it
You andielded; nor you ward borns and tainy
That call would for blood with a King and woman, the life,
But my surely of my son, it is a gimful knew that
it not to go whither.

MENENIUS:
By yonder
```

## Notes

- Val loss is noisy in the 60–70K range (oscillates ±0.02), stabilises in the 78–85K range
- Noise is expected: single-threaded CPU, no parallelism; per-step variance is higher than GPU
- Best eval 1.5411 at 84K; final 1.5425 (+0.014 vs GPU FP32 1.5281) — platform overhead only
- Gap vs GPU FP32 (+0.014) used in paper Table 1 as platform isolation baseline
- Gap vs CPU BF16W isolates precision cost of BF16W scheme (pending BF16W solo run)
- Speed: 201.5 ms/sample = 12.3× slower than GPU (16.4 ms/sample) — single-thread vs CUDA

## Paper Reference

Table 1 row: `CPU Adam FP32 | 85,000 | 1.5425 | 55.3% | 201.5`
