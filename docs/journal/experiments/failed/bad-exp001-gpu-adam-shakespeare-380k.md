> ⛔ **CRITICAL — EXCLUDED FROM PAPER**
> This experiment is invalid for two reasons:
> 1. **Wrong param count**: architecture embed=80/ff=240/vocab=256 = **277,920 params**, not 380K. The 380K label applied to TinyStories (vocab=1501). The vocab-budget contrast table ("same params") is therefore false.
> 2. **Batch bug (BUG-001)**: trained with `--batch-size 16`, which runs 16 sequential Adam steps at lr/16 instead of true gradient accumulation — convergence is ~4× slower per sample.
> See replacement experiment: [exp002-gpu-adam-shakespeare-334k.md](exp002-gpu-adam-shakespeare-334k.md)

# EXP-001 — GPU Adam FP32 — Shakespeare char-level — 380K params ⛔ EXCLUDED

## Summary

| Property | Value |
|---|---|
| Model | 380K params, embedDim=80, heads=4, ff=240, layers=4, seqLen=128 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | GPU Adam FP32, lr=0.003, linear decay, warmup=200 steps |
| Samples | 250,000 |
| Final eval loss | **1.6152** |
| Final train loss | 1.4306 |
| Final accuracy | 52.69% |
| Train/eval gap | 0.18 (stable, no overfit) |
| Speed | 16.51 ms/sample avg |
| Total time | 4126.4s (~68 min) |
| Checkpoint | exp001-gpu-adam-shakespeare-380k.neuro |

**Vocab-budget contrast (paper Section 3):**

| Config | Vocab | P_reason | Eval @250K |
|---|---|---|---|
| 380K TinyStories | 1501 | ~260K | 2.8921 |
| 380K Shakespeare | 256 | ~360K | **1.6152** |

Same params, same optimizer — **1.28 loss gap purely from vocabulary tax**.

## Demo output (GPU Adam FP32, temperature=0.8, seed=42)

**Prompt: `HAMLET:`**
```
HAMLET:
Clarency, hard I po coure,
Than mocking I will him that of he savarel.

BRATUS:
I more with you blince office ovely,
Be shard child to the preed dry breathed,
Her love I will boy: for the more death it:
Why, not say nor there dead up, thy these it
unsuland the were he sold of him thy Angel
```

**Prompt: `KING LEAR:`**
```
KING LEAR:
Well, you king, fore it she know
Lest you readserse words, I with hate
And the shalf and to their with worms, do gracious
Of and Warwick bid youse, O would his sorrow;
Let the cannot both when securior:
And yet in you mile be be dike in a proud,
Anged a father to him which is on a funcious
```

**Prompt: `To be or not to be`**
```
To be or not to bear
will I shall hie has you, tell it with Clarence, Juliet
the comes is priciss of a tisturn me too.

CORIOLANUS:
A some tongue that I well.

CATESBY:
Many I can thee shame the remain with you night
The good aid shite a king, Richardley, or rest
That I confold, not thou are the no him,
```

**Quality assessment:** Correct Shakespeare structure (CHARACTER:\nSpeech, multi-character scenes, iambic-like line breaks). Words are plausible char-level constructions ("securior", "funcious") but semantically nonsensical — expected at 380K params. No UNK tokens (char-level). Model trained correctly.

*These outputs serve as GPU FP32 baseline for comparison against CPU FP32 and CPU BF16W runs.*

---

## Raw training log

PS C:\repos\_Neuro> cd "c:\repos\_Neuro\neuro-fabric\src\Neuro.Attention.TrainApp"; & "C:\Program Files\dotnet\dotnet.exe" run -c Release --no-build -- exp001-gpu-adam-shakespeare-380k.neuro 250000 --adam --dataset shakespeare --embed-dim 80 --num-heads 4 --ff-dim 240 --num-layers 4 --seq-len 128 --batch-size 16 --lr 0.003 --lr-schedule linear --warmup-steps 200 --log-every 5000
Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (GPU Adam FP32): seqLen=128 embedDim=80 heads=4 ff=240 layers=4

Dataset: shakespeare  BatchSize: 16  (log every 5,000 samples)
Samples    Train Loss     Eval Loss      Accuracy     ms/sample  LR        
--------------------------------------------------------------
4,992      3.2586         2.5101         28.19%       16.99      0.002978  
9,984      2.3783         2.2516         34.33%       16.70      0.002918  
14,976     2.1961         2.1315         37.48%       16.72      0.002857  
19,968     2.0735         2.0232         39.97%       16.80      0.002796  
24,960     1.9778         1.9679         41.94%       16.70      0.002735  
29,952     1.9064         1.9269         43.10%       16.58      0.002675  
34,944     1.8573         1.8877         44.15%       16.32      0.002614  
39,936     1.8059         1.8560         44.78%       15.93      0.002553  
44,928     1.7674         1.8400         45.80%       15.97      0.002493  
49,920     1.7382         1.8240         46.08%       16.21      0.002432  
54,912     1.7104         1.8011         46.57%       15.94      0.002371  
59,904     1.6834         1.7924         47.17%       15.85      0.002311  
64,896     1.6668         1.7729         47.81%       16.18      0.002250  
69,888     1.6416         1.7733         48.18%       16.14      0.002189  
74,880     1.6272         1.7575         48.19%       16.25      0.002129  
79,872     1.6174         1.7535         48.40%       16.47      0.002068  
84,864     1.6043         1.7508         48.24%       16.24      0.002007  
89,856     1.5879         1.7277         48.91%       16.17      0.001947  
94,848     1.5760         1.7276         49.30%       16.65      0.001886  
99,840     1.5722         1.7191         49.38%       15.97      0.001825  
104,832    1.5624         1.7088         49.77%       16.13      0.001765  
109,824    1.5549         1.7085         50.13%       16.54      0.001704  
114,816    1.5480         1.6974         49.88%       16.37      0.001643  
119,808    1.5407         1.6886         50.13%       16.04      0.001583  
124,800    1.5297         1.6859         50.42%       16.31      0.001522  
129,792    1.5256         1.6835         50.73%       16.39      0.001461  
134,784    1.5163         1.6762         50.52%       16.60      0.001401  
139,776    1.5157         1.6673         50.63%       16.44      0.001340  
144,768    1.5053         1.6686         51.07%       16.68      0.001279  
149,760    1.5021         1.6678         51.33%       16.82      0.001218  
154,752    1.5005         1.6690         51.28%       17.74      0.001158  
159,744    1.4929         1.6589         51.20%       16.49      0.001097  
164,736    1.4869         1.6623         51.00%       18.51      0.001036  
169,728    1.4837         1.6536         51.54%       19.99      0.000976  
174,720    1.4798         1.6467         51.70%       19.93      0.000915  
179,712    1.4746         1.6486         51.78%       19.98      0.000854  
184,704    1.4740         1.6407         52.08%       19.97      0.000794  
189,696    1.4656         1.6404         52.10%       18.77      0.000733  
194,688    1.4629         1.6369         52.14%       15.71      0.000672  
199,680    1.4619         1.6371         52.28%       15.88      0.000612  
204,672    1.4577         1.6296         52.50%       15.59      0.000551  
209,664    1.4517         1.6329         52.56%       15.11      0.000490  
214,656    1.4510         1.6237         52.48%       15.04      0.000430  
219,648    1.4435         1.6248         52.39%       15.06      0.000369  
224,640    1.4459         1.6239         52.36%       15.05      0.000308  
229,632    1.4380         1.6215         52.62%       15.09      0.000248  
234,624    1.4453         1.6207         52.75%       15.04      0.000187  
239,616    1.4376         1.6163         52.82%       15.12      0.000126  
244,608    1.4339         1.6168         52.81%       15.11      0.000066  
249,600    1.4331         1.6157         52.88%       15.06      0.000030  
250,000    1.4306         1.6152         52.69%       1.25       0.000030  
--------------------------------------------------------------
Total time: 4126.4s  (16.51 ms/sample avg, 250,000 samples)
Checkpoint saved: exp001-gpu-adam-shakespeare-380k.neuro
PS C:\repos\_Neuro\neuro-fabric\src\Neuro.Attention.TrainApp> 