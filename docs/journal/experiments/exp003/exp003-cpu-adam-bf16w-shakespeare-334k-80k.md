# EXP-003 — CPU Adam BF16W — Shakespeare char-level — 334K params — b=1 — 80K samples

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | CPU Adam BF16 weights, lr=0.003, linear decay, warmup=200 steps, b=1 |
| Samples | 80,000 |
| Final eval loss | **1.5480** |
| Best eval loss | **1.5477** @ 78K |
| Final train loss | 1.3581 |
| Final accuracy | 54.57% |
| Speed | 149.51 ms/sample avg |
| Total time | 11961.0s (~199 min / ~3.3 h) |
| Checkpoint | exp-cpu-adam-bf16w-shakespeare-334k-80k.neuro |
| Saved with | v1.0.1+cfcf5177972e54fa3981182160bbf990f4d15886 |
| Release | v1.0.2 (BUG-006 fix applied) |

## Notes

- Tuned to 80K samples (was 85K in exp002)
- BUG-006 fixed: Adam bias correction now uses `GlobalStep` (incremented once per training step)
- Best eval loss 1.5477 — improvement over exp002 BF16W best of 1.5547 at 85K
- Gap vs GPU FP32 (exp003): **+0.025** val loss (was +0.027 in exp002) — slight improvement
- BF16W is 7.7× slower than GPU FP32 on this hardware (149.51 vs 19.37 ms/sample)

## Comparison vs exp002

| Metric | EXP-002 BF16W (85K) | EXP-003 BF16W (80K) | Delta |
|---|---|---|---|
| Best eval loss | 1.5547 | **1.5477** | -0.007 |
| Final accuracy | 54.72% | **54.57%** | -0.15% |
| Speed (ms/sample) | 137.9 | 149.51 | +11.6 |
| BUG-006 fix | ❌ | ✅ | — |

## Gap vs GPU FP32 oracle (exp003)

| Variant | Best eval loss | Gap |
|---|---|---|
| GPU Adam FP32 | 1.5226 | — |
| CPU Adam BF16W | 1.5477 | **+0.025** |

## Demo output (CPU Adam BF16W, temperature=0.8, 80K samples)

**Prompt: `to be or not to be`**
```
to be or not to be,
No play him follow and and so ails
To call'd the bondain and the son' ta'en
Where best his art that that my morest
Was a of your practory.

CAMILLO:
Early of sit our badmentages;
Mother my to give must with seems thee loved
Dear with tyrnell'd was grow him: sit you may gone.

COMINIUS:
```

**Prompt: `MENENIUS :`**
```
MENENIUS :
if for my life, for a most rever in reason,
And would some a toward had I will bone 't!

Second Servant:
Who seem to York at thy lord sede, your longly:
Content, who never he cannot bounds end:
Who's much subject you me shall be short
About and a wife all quarried may may.

KING RICHARD I
```

**Prompt: `KING RICHARD III`**
```
KING RICHARD III:
By fear that thou daughter, O and for Wick,
That's men I will prove their must and tooks.

BUCKINGHAM:
And death I soul. What that I will now,
Then were his people of the king:
Of they lost be heard passage a face two be go;
And they with a lives men as the country,
The instremen of profe
```

## Raw training log

```
NeuronFabric TrainApp v1.0.1+cfcf5177972e54fa3981182160bbf990f4d15886  |  Apache 2.0 License  |  github.com/neuro-fabric

Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (CPU Adam BF16 weights): seqLen=128 embedDim=88 heads=4 ff=264 layers=4

Dataset: shakespeare  BatchSize: 1  (log every 1,000 samples)
Samples    Train Loss     Eval Loss      Accuracy     ms/sample  LR        
--------------------------------------------------------------
1,000      2.9750         2.5153         28.70%       138.03     0.002970  
2,000      2.4433         2.3478         32.87%       130.29     0.002932  
...
74,000     1.3664         1.5534         54.36%       172.40     0.000226  
75,000     1.3614         1.5511         54.48%       173.03     0.000188  
76,000     1.3613         1.5504         54.47%       172.72     0.000150  
77,000     1.3497         1.5498         54.54%       172.38     0.000113  
78,000     1.3643         1.5477         54.49%       172.16     0.000075  
79,000     1.3568         1.5479         54.57%       172.75     0.000038  
80,000     1.3581         1.5480         54.57%       172.46     0.000030  
--------------------------------------------------------------
Total time: 11961.0s  (149.51 ms/sample avg, 80,000 samples)
Checkpoint saved: C:\repos\_Neuro\neuro-fabric\run\results\exp-cpu-adam-bf16w-shakespeare-334k-80k.neuro
```
