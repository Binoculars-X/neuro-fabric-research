# EXP-001 — TinyStories GPU Adam FP32 vs BF16W Convergence

**Date:** 30/05/26  
**Status:** planned  
**Hypothesis:** BF16W (w=BF16, m=FP32, v=FP32) converges to the same held-out loss floor as GPU FP32 Adam on TinyStories vocab=1501 when trained with proper 90/10 eval split.

---

## Setup

**Dataset:** TinyStories (word-level, vocab=1501, 90/10 train/val split via `trainapp2`)  
**Config:** `embedDim=80, heads=4, ff=240, layers=4, seqLen=128` ≈ 380K params  
**Seed:** `Random(42)`  
**Batch size:** 4  
**Log interval:** every 4000 samples  
**Target samples:** 100K each run  
**Hardware:** RTX 4090

---

## Runs

### Run A — GPU FP32 Adam (reference)

**Command:**
```powershell
dotnet run --project src/Neuro.Attention.TrainApp -c Release -- `
  exp001-gpu-fp32.neuro 100000 --adam `
  --dataset tinystories --embed-dim 80 --num-heads 4 --ff-dim 240 --num-layers 4 `
  --batch-size 4 --log-every 4000
```

**Expected:** plateau ~1.85–1.90 held-out loss (Day 7 in-sample floor ~1.85; held-out expected slightly higher)

| Samples | Train Loss | Val Loss | Accuracy |
|---|---|---|---|
| | | | |

---

### Run B — CPU BF16W Adam (silicon reference)

**Command:**
```powershell
dotnet run --project src/Neuro.Attention.TrainApp -c Release -- `
  exp001-cpu-bf16w.neuro 100000 --adam-cpu-bf16w `
  --dataset tinystories --embed-dim 80 --num-heads 4 --ff-dim 240 --num-layers 4 `
  --batch-size 4 --log-every 4000
```

**Expected:** val loss within 0.05 of Run A at 100K samples (convergence parity claim)

| Samples | Train Loss | Val Loss | Accuracy |
|---|---|---|---|
| | | | |

---

## Success Criteria

| Criterion | Pass condition |
|---|---|
| BF16W convergence parity | Final val loss ≤ Run A + 0.05 |
| No divergence | Loss monotonically decreasing over first 40K |
| Demo quality | Coherent TinyStories output, <5 UNK/150 tokens |

---

## Results

*To be filled after runs complete.*

---

## Conclusion

*To be filled.*
