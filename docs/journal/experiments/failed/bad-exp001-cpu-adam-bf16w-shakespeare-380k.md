> ⛔ **CRITICAL — EXCLUDED FROM PAPER**
> This experiment is invalid for two reasons:
> 1. **Wrong param count**: architecture embed=80/ff=240/vocab=256 = **277,920 params**, not 380K.
> 2. **Batch bug (BUG-001)**: trained with `--batch-size 16` (sequential Adam steps at lr/16). Additionally stopped early at 65K samples due to convergence failure — which may be partly attributable to the batch bug rather than BF16W precision alone.
> See [bugs/bug-001-trainbatch-sequential-adam-steps.md](../bugs/bug-001-trainbatch-sequential-adam-steps.md)

# EXP-001 — CPU Adam BF16W — Shakespeare char-level — 380K params ⛔ EXCLUDED

## Summary

| Property | Value |
|---|---|
| Model | 380K params, embedDim=80, heads=4, ff=240, layers=4, seqLen=128 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | CPU Adam BF16W (w=BF16, m=FP32, v=FP32), lr=0.003, linear decay, warmup=200 steps |
| Samples | ~65K (stopped early — convergence failure) |
| Eval loss @65K | 2.5434 |
| Train loss @65K | 2.5658 |
| Accuracy @65K | 27.93% |
| Speed | ~113 ms/sample |
| Status | ⛔ STOPPED — severe convergence penalty vs GPU FP32 |
| Checkpoint | exp001-cpu-adam-bf16w-shakespeare-380k.neuro |

## Comparison vs GPU FP32 at same sample count

| Variant | Eval @65K | Accuracy @65K |
|---|---|---|
| GPU Adam FP32 | **1.7729** | 47.81% |
| CPU Adam BF16W | 2.5434 | 27.93% |
| **Gap** | **+0.77** | **-20%** |

BF16W shows a **severe convergence penalty** on Shakespeare char-level. The curve barely moved from 2.62 → 2.54 over 50K samples while GPU FP32 reached 1.77 in the same window.

## Analysis

The BF16W appointment corpus result (near-zero penalty) does not generalise to Shakespeare char-level at lr=0.003. Likely causes:
- Char-level gradients are smaller in magnitude than word-level — BF16 precision insufficient to represent small weight updates
- Gradient updates underflow to zero in BF16, effectively stopping learning
- Word-level (appointment corpus) has larger gradient signal per token — less sensitive to precision loss

## Paper implication

BF16W is **not viable at this learning rate on char-level tasks**. This is an honest negative finding:
- FP32 Adam is required for reliable convergence at 380K params
- BF16W requires either: higher LR, gradient scaling, or loss scaling to compensate
- FPGA implication: FP32 moments are needed (~4.56 MB, borderline on ZCU102); or reduce model to ~360K params to fit within BRAM

## Raw training log (partial — stopped at ~65K)

```
Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (CPU Adam BF16 weights): seqLen=128 embedDim=80 heads=4 ff=240 layers=4

Dataset: shakespeare  BatchSize: 16  (log every 5,000 samples)
Samples    Train Loss     Eval Loss      Accuracy     ms/sample  LR        
--------------------------------------------------------------
4,992      4.2208         3.3246         14.62%       142.17     0.002978  
9,984      3.0610         2.8381         23.91%       121.24     0.002918  
14,976     2.7632         2.7037         26.12%       112.95     0.002857  
19,968     2.6817         2.6505         26.86%       113.04     0.002796  
24,960     2.6449         2.6129         27.60%       113.02     0.002735  
29,952     2.6225         2.5930         27.34%       113.06     0.002675  
34,944     2.6061         2.5774         27.53%       113.08     0.002614  
39,936     2.5928         2.5683         27.51%       113.11     0.002553  
44,928     2.5818         2.5611         27.71%       113.28     0.002493  
49,920     2.5780         2.5558         27.86%       113.12     0.002432  
54,912     2.5703         2.5537         27.93%       113.16     0.002371  
59,904     2.5688         2.5461         27.91%       113.11     0.002311  
64,896     2.5658         2.5434         27.93%       113.81     0.002250  
[STOPPED]
```
