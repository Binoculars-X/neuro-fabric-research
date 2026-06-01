# EXP-002b — CPU Adam BF16W — Shakespeare char-level — 334K params — 85K samples (post bug-004/005 fix)

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | CPU Adam BF16W (weights=BF16, moments=FP32), lr=0.003, linear decay, warmup=200 steps, b=1 |
| Samples | 85,000 |
| Final eval loss | **1.5547** |
| Best eval loss | **1.5545 @ 82K** |
| Final train loss | 1.3656 |
| Final accuracy | 54.72% |
| Speed | 137.90 ms/sample avg |
| Total time | 11721.8s (~3.26h) |
| Checkpoint | exp002-cpu-adam-bf16w-shakespeare-334k-85k.neuro |
| Log | exp002-cpu-adam-bf16w-shakespeare-334k-85k.neuro.log |
| Version | v1.0.1+b4478463854f94e6f47694ddfc13afe1546a6488 |

## Bugs fixed vs previous BF16W run (exp002)

- **BUG-004**: `_step` shadowed in BF16W derived classes → bias correction was resetting. Fixed: `private→protected` in base classes.
- **BUG-005**: FP32 master weight overwritten from BF16 decode on each step → sub-BF16 updates lost during warmup. Fixed: `w[i,j] = wf` (accumulate at full precision).

## Comparison vs GPU FP32 oracle

| Run | Optimizer | Eval loss | Accuracy | ms/sample |
|---|---|---|---|---|
| EXP-001 GPU FP32 | GPU Adam FP32 | **1.5281** | 56.20% | 16.4 |
| **EXP-002b CPU BF16W** | **CPU Adam BF16W** | **1.5547** | **54.72%** | **137.9** |

Gap vs GPU FP32: +0.0266 eval loss. BF16W converges to within 1.7% of GPU oracle.

## Demo output (CPU Adam BF16W, temperature=0.8, seed=42, 85K samples)

**Prompt: `MENENIUS :`**
```
MENENIUS :
And there is ever in the rather for
To call be in abondine leave and oath
This grace to him.

MAMILLIUS:
The love warm him you grief, as done he drunks
The pleasure, an but reason in the grief,
And loves-is' sendenger corray a gentlemen?--'Tis plucks
to high orsequalicy! What is bress
Of
```

**Prompt: `to be or not to be`**
```
to be or not to be by
both of his ports here nothiness of whence
Whose would gentleman in their heart.

JULIET:
Why, senter you we there thou wavest shall be not
The estruck's and to the king mafe.

CLEOMENES:
By then the queen of bless the be for unkey
To the teather be so been in as office,
And what do m
```

## Notes

- No `aW` garbage artifact (unlike CPU FP32 BUG-003) — BF16W quantization noise prevents memorization of zero-context prior
- Speed improved vs old BF16W run (184.52 → 137.90 ms/sample) — likely due to bug-005 fix reducing wasted compute during warmup
- This result is the paper1 CPU BF16W row
