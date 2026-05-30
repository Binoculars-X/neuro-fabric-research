# EXP-002 — CPU Adam FP32 — Shakespeare char-level — 334K params — b=1

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | CPU Adam FP32, lr=0.003, linear decay, warmup=200 steps, b=1 |
| Samples | 80,000 |
| Final eval loss | **1.5407** |
| Final train loss | 1.3321 |
| Final accuracy | 55.19% |
| Speed | 190.32 ms/sample avg |
| Total time | 15225.4s (~4.2h) |
| Checkpoint | exp002-cpu-adam-fp32-shakespeare-334k.neuro |

## GPU vs CPU comparison @ 80K samples

| Run | Optimizer | Eval loss | Accuracy | ms/sample |
|---|---|---|---|---|
| EXP-002 GPU FP32 | GPU Adam | 1.5394 | 56.01% | 18.02 |
| **EXP-002 CPU FP32** | **CPU Adam** | **1.5407** | **55.19%** | **190.32** |
| EXP-002 CPU BF16W | CPU Adam BF16W | 1.5375 | 54.74% | 184.52 |

**CPU FP32 tracks GPU within 0.001 eval loss** — convergence parity confirmed. 10.6× slower in wall time.

## Raw training log

```
PS C:\repos\_Neuro\neuro-fabric\src\Neuro.Attention.TrainApp> & "C:\Program Files\dotnet\dotnet.exe" run -c Release --no-build -- exp002-cpu-adam-fp32-shakespeare-334k.neuro 80000 --embed-dim 88 --num-heads 4 --ff-dim 264 --num-layers 4 --dataset shakespeare --adam-cpu --lr 0.003 --lr-schedule linear --warmup-steps 200 --log-every 5000
Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (CPU Adam): seqLen=128 embedDim=88 heads=4 ff=264 layers=4

Dataset: shakespeare  BatchSize: 1  (log every 5,000 samples)
Samples    Train Loss     Eval Loss      Accuracy     ms/sample  LR        
--------------------------------------------------------------
5,000      2.4377         2.1259         38.06%       193.82     0.002820  
10,000     1.9653         2.0002         41.74%       174.72     0.002632  
15,000     1.8006         1.8570         45.45%       182.46     0.002444  
20,000     1.7018         1.7905         47.38%       159.72     0.002256  
25,000     1.6486         1.7672         48.02%       191.45     0.002068  
30,000     1.6001         1.7161         49.61%       168.93     0.001880  
35,000     1.5571         1.7162         50.65%       180.37     0.001692  
40,000     1.5266         1.6871         50.70%       161.66     0.001504  
45,000     1.4949         1.6396         51.42%       182.96     0.001316  
50,000     1.4618         1.6320         51.49%       226.02     0.001128  
55,000     1.4428         1.6160         53.44%       182.25     0.000940  
60,000     1.4198         1.5901         53.36%       211.21     0.000752  
65,000     1.3879         1.5751         53.89%       214.42     0.000564  
70,000     1.3718         1.5644         54.80%       205.51     0.000376  
75,000     1.3461         1.5494         54.63%       204.56     0.000188  
80,000     1.3321         1.5407         55.19%       204.13     0.000030  
--------------------------------------------------------------
Total time: 15225.4s  (190.32 ms/sample avg, 80,000 samples)
Checkpoint saved: exp002-cpu-adam-fp32-shakespeare-334k.neuro
```
