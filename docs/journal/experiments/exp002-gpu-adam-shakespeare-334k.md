# EXP-002 — GPU Adam FP32 — Shakespeare char-level — 334K params — b=1

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | GPU Adam FP32, lr=0.003, linear decay, warmup=200 steps, **b=1** |
| Samples | 80,000 |
| Final eval loss | **1.5394** |
| Final train loss | 1.3090 |
| Final accuracy | 56.01% |
| Train/eval gap | 0.23 |
| Speed | 18.02 ms/sample avg |
| Total time | 1441.8s (~24 min) |
| Checkpoint | exp002-gpu-adam-shakespeare-334k.neuro |

## Key findings

### b=1 dramatically outperforms b=16 per sample (BUG-001)

| Run | Params | Batch | Samples | Eval loss |
|---|---|---|---|---|
| EXP-001 GPU b=16 | ~278K (Shakespeare) | 16 | 80,000 | 1.7535 |
| **EXP-002 GPU b=1** | **334K** | **1** | **80,000** | **1.5394** |

EXP-002 reaches **1.5394** at 80K vs EXP-001 still at **1.7535** at 80K with b=16.
Root cause: see [bugs/bug-001-trainbatch-sequential-adam-steps.md](../bugs/bug-001-trainbatch-sequential-adam-steps.md) — b=16 makes 16 sequential Adam steps at lr/16 rather than one true batch gradient step.

### Actual param count clarification

The "380K" label used in EXP-001 was based on the TinyStories config (vocab=1501 → 377,520 params). With Shakespeare vocab=256, the same architecture (embed=80, ff=240) is only **277,920 params**. This experiment uses a genuinely larger model (334K) on Shakespeare.

| Config | vocab | embed | ff | layers | Actual params |
|---|---|---|---|---|---|
| EXP-001 "380K" TinyStories | 1501 | 80 | 240 | 4 | **377,520** |
| EXP-001 "380K" Shakespeare | 256 | 80 | 240 | 4 | **277,920** |
| EXP-002 Shakespeare | 256 | 88 | 264 | 4 | **333,872** |

### Convergence curve (b=1, GPU Adam FP32)

EXP-002 converges to **1.54 eval loss in only 80K samples** — vs EXP-001 needing ~180K samples to reach 1.65 with b=16.

---

## Demo output (GPU Adam FP32, temperature=0.8, seed=42, 80K samples)

**Prompt: `HAMLET:`**
```
HAMLET:
But then die encountersed, when he art me and
departice's all the soon ta'en words hath her:
To scarce him your love war a neath:
That weep on for cry as for all of?
The Duke of not his stands up me?

Servant:
So but thee, we did him my young.

BRUTUS:
Nay, pray'd the gods of Juliet
I c
```

**Prompt: `KING LEAR:`**
```
KING LEAR:
What so not this not for---Favour hath
I have this courteminesion in band.o, go the Claudio;
For this supposed under us the crepts
Than a littless of this delightent the king's
Englew the wind of which he king it now of mean!

DUKE OF YORK:
And Jove call the king of you worse,
And sets upo
```

**Prompt: `To be or not to be`**
```
To be or not to be?

KING EDWARD IV:
And, do you in my limbs, my lord, hell you be out
From to him a touch'd mighting, then I come to
spounsy.

PARIS:
I hearing thee were him shall one that special as
with you perforce the slain and you come to
the monument of was most thou sentingomles,
Get the part of hi
```

**Quality assessment:** Correct multi-character Shakespeare structure. Real character names (BRUTUS, DUKE OF YORK, KING EDWARD IV, PARIS). Iambic line rhythms visible. Words are plausible ("departice's", "delightent", "sentingomles") — char-level constructions expected at this scale. Achieved in **only 80K samples** vs EXP-001's 250K samples with b=16.

---

## Raw training log

```
PS C:\repos\_Neuro\neuro-fabric\src\Neuro.Attention.TrainApp> & "C:\Program Files\dotnet\dotnet.exe" run -c Release --no-build -- exp002-gpu-adam-shakespeare-334k.neuro 80000 --embed-dim 88 --num-heads 4 --ff-dim 264 --num-layers 4 --dataset shakespeare --adam --lr 0.003 --lr-schedule linear --warmup-steps 200 --log-every 5000
Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (GPU Adam FP32): seqLen=128 embedDim=88 heads=4 ff=264 layers=4

Dataset: shakespeare  BatchSize: 1  (log every 5,000 samples)
Samples    Train Loss     Eval Loss      Accuracy     ms/sample  LR        
--------------------------------------------------------------
5,000      2.3489         2.0502         40.28%       15.89      0.002820  
10,000     1.8612         1.9189         44.35%       15.99      0.002632  
15,000     1.7133         1.8044         47.06%       16.01      0.002444  
20,000     1.6348         1.7694         48.91%       16.09      0.002256  
25,000     1.5843         1.7578         49.45%       16.26      0.002068  
30,000     1.5349         1.6807         50.12%       15.60      0.001880  
35,000     1.5109         1.6490         51.67%       19.11      0.001692  
40,000     1.4771         1.6518         51.56%       21.58      0.001504  
45,000     1.4565         1.6257         52.23%       21.08      0.001316  
50,000     1.4301         1.6191         53.05%       21.09      0.001128  
55,000     1.4098         1.5888         53.34%       16.99      0.000940  
60,000     1.3802         1.5764         54.54%       16.24      0.000752  
65,000     1.3677         1.5575         54.69%       16.72      0.000564  
70,000     1.3426         1.5531         54.85%       17.98      0.000376  
75,000     1.3292         1.5434         55.51%       20.91      0.000188  
80,000     1.3090         1.5394         56.01%       20.76      0.000030  
--------------------------------------------------------------
Total time: 1441.8s  (18.02 ms/sample avg, 80,000 samples)
Checkpoint saved: exp002-gpu-adam-shakespeare-334k.neuro
```
