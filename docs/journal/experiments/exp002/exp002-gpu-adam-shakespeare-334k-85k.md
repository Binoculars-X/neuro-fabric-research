# EXP-002 — GPU Adam FP32 — Shakespeare char-level — 334K params — b=1 — 85K samples

## Summary

| Property | Value |
|---|---|
| Model | 334K params, embedDim=88, heads=4, ff=264, layers=4, seqLen=128, vocab=256 |
| Dataset | Shakespeare char-level, vocab=256, train=1,039,854 tokens, val=115,540 |
| Optimizer | GPU Adam FP32, lr=0.003, linear decay, warmup=200 steps, b=1 |
| Samples | 85,000 |
| Final eval loss | **1.5425** |
| Best eval loss | **1.5381** @ 84K |
| Final train loss | 1.2816 |
| Final accuracy | 55.38% |
| Speed | 18.86 ms/sample avg |
| Total time | 1603.0s (~26.7 min) |
| Checkpoint | exp-gpu-adam-shakespeare-334k-85k.neuro |
| savedWith | v1.0.1+b4478463854f94e6f47694ddfc13afe1546a6488 |

## Notes

- First clean run with `AppVersion` bug fixed (GPU `AdamTransformerBus.Save` now stamps version correctly)
- Still descending at 85K — not yet at minimum; sweet spot confirmed ~87-90K based on prior runs

## Demo output (GPU Adam FP32, temperature=0.8, seed=42, 85K samples)

**Prompt: `HORAZIO`**
```
HORAZIO:
Lords! Lare is the both return
My love I shall not of the gods imposs
A colour of courtesy, to the grace out of yours
Make away is a tremblerous and those of sir.

Second Murderer:
Our fear that what with womb the sing of grace,
Are nots their holds use he withing and so.

KING RICHARD I
```

**Prompt: `ALFONSO`**
```
ALFONSOM the not
becomes to child they the corcle, sir, it let me might,
And band usaking of the mother to't: I would send
Thou things thy life seem out with his undertake.

CORIOLANUS:
My lord, you have loved the present welcome,
I can those that wolse a wife all that which
That much be doth thing
```

**Prompt: `to be or not to be`**
```
to be or not to be me
daughter here--

HORTENSIO:
Perjury that thou first my of sons, marriage.

CATESBY:
I am thou temper him in days. I'll light it purpose.

QUEEN:
I hear me doth be my lord, in then lift deed,
And the that infanted death'd not of asdure
Intend not the grave of state, carge of Elbow?
```

## Raw training log

```
NeuronFabric TrainApp v1.0.1+b4478463854f94e6f47694ddfc13afe1546a6488  |  Apache 2.0 License  |  github.com/neuro-fabric

Dataset: Shakespeare, vocab=256, train=1,039,854, val=115,540
New model (GPU Adam FP32): seqLen=128 embedDim=88 heads=4 ff=264 layers=4

Dataset: shakespeare  BatchSize: 1  (log every 1,000 samples)
Samples    Train Loss     Eval Loss      Accuracy     ms/sample  LR        
--------------------------------------------------------------
1,000      2.9787         2.4882         28.66%       16.89      0.002972  
2,000      2.4185         2.3120         32.90%       17.08      0.002936  
3,000      2.2507         2.1696         36.06%       16.43      0.002901  
4,000      2.1217         2.1007         39.11%       17.21      0.002866  
5,000      2.0261         2.0052         41.74%       16.67      0.002830  
6,000      1.9505         2.0073         41.58%       16.43      0.002795  
7,000      1.8870         1.9930         42.63%       16.27      0.002759  
8,000      1.8537         1.9086         44.59%       17.44      0.002724  
9,000      1.8179         1.8963         43.64%       16.19      0.002689  
10,000     1.7827         1.8837         45.25%       16.02      0.002653  
11,000     1.7473         1.8722         45.01%       16.61      0.002618  
12,000     1.7209         1.8376         46.83%       17.04      0.002583  
13,000     1.7256         1.8256         46.42%       16.35      0.002547  
14,000     1.6960         1.8470         46.06%       16.25      0.002512  
15,000     1.6769         1.8041         47.72%       18.02      0.002476  
16,000     1.6502         1.7815         47.75%       16.79      0.002441  
17,000     1.6394         1.7981         47.06%       17.18      0.002406  
18,000     1.6321         1.7860         46.72%       17.34      0.002370  
19,000     1.6182         1.7808         47.75%       17.85      0.002335  
20,000     1.6154         1.7493         49.02%       17.04      0.002300  
21,000     1.5865         1.7711         48.16%       17.36      0.002264  
22,000     1.5907         1.7310         49.18%       17.73      0.002229  
23,000     1.5730         1.7461         48.93%       16.99      0.002193  
24,000     1.5698         1.7169         49.11%       16.86      0.002158  
25,000     1.5552         1.7509         49.43%       17.85      0.002123  
26,000     1.5516         1.7214         50.07%       17.63      0.002087  
27,000     1.5345         1.6850         50.67%       17.39      0.002052  
28,000     1.5316         1.7077         50.01%       17.29      0.002017  
29,000     1.5256         1.6942         50.02%       18.14      0.001981  
30,000     1.5302         1.7171         50.00%       17.50      0.001946  
31,000     1.5072         1.6822         50.93%       16.66      0.001910  
32,000     1.5236         1.6846         50.65%       17.73      0.001875  
33,000     1.4979         1.6801         50.92%       17.04      0.001840  
34,000     1.5073         1.6716         50.98%       17.23      0.001804  
35,000     1.4929         1.6778         51.05%       17.10      0.001769  
36,000     1.4806         1.6829         51.31%       17.71      0.001733  
37,000     1.4762         1.6802         50.94%       16.84      0.001698  
38,000     1.4778         1.6723         51.53%       17.97      0.001663  
39,000     1.4753         1.6391         52.03%       17.60      0.001627  
40,000     1.4660         1.6416         51.98%       16.26      0.001592  
41,000     1.4486         1.6737         51.78%       16.23      0.001557  
42,000     1.4457         1.6290         52.50%       16.17      0.001521  
43,000     1.4501         1.6548         51.82%       17.57      0.001486  
44,000     1.4475         1.6270         52.58%       16.26      0.001450  
45,000     1.4425         1.6155         52.53%       16.54      0.001415  
46,000     1.4301         1.6270         52.80%       16.34      0.001380  
47,000     1.4312         1.6181         52.76%       17.24      0.001344  
48,000     1.4316         1.5968         53.21%       16.64      0.001309  
49,000     1.4250         1.6332         51.86%       16.54      0.001274  
50,000     1.4125         1.6112         53.63%       17.31      0.001238  
51,000     1.4191         1.5954         53.55%       16.37      0.001203  
52,000     1.4070         1.6169         53.34%       16.68      0.001167  
53,000     1.3975         1.5966         53.54%       16.36      0.001132  
54,000     1.4306         1.6140         53.10%       17.14      0.001097  
55,000     1.3883         1.5836         54.22%       20.90      0.001061  
56,000     1.3841         1.5950         53.71%       19.76      0.001026  
57,000     1.3747         1.5864         53.48%       22.86      0.000991  
58,000     1.3792         1.5826         53.60%       22.03      0.000955  
59,000     1.3718         1.5954         53.74%       22.16      0.000920  
60,000     1.3839         1.5729         54.10%       22.57      0.000884  
61,000     1.3685         1.5884         54.31%       22.43      0.000849  
62,000     1.3718         1.5795         54.19%       22.31      0.000814  
63,000     1.3588         1.5835         54.22%       22.12      0.000778  
64,000     1.3708         1.5837         54.76%       22.26      0.000743  
65,000     1.3514         1.5713         54.52%       22.13      0.000708  
66,000     1.3508         1.5845         54.31%       22.25      0.000672  
67,000     1.3428         1.5817         54.39%       22.13      0.000637  
68,000     1.3512         1.5785         54.37%       22.67      0.000601  
69,000     1.3385         1.5621         54.70%       22.37      0.000566  
70,000     1.3296         1.5589         54.63%       21.95      0.000531  
71,000     1.3408         1.5696         54.70%       22.82      0.000495  
72,000     1.3225         1.5654         54.52%       22.87      0.000460  
73,000     1.3193         1.5638         54.76%       22.71      0.000425  
74,000     1.3126         1.5601         54.76%       21.92      0.000389  
75,000     1.3109         1.5595         54.69%       21.88      0.000354  
76,000     1.3106         1.5478         54.75%       21.61      0.000318  
77,000     1.3026         1.5533         54.83%       22.05      0.000283  
78,000     1.3033         1.5571         55.03%       22.46      0.000248  
79,000     1.2981         1.5507         55.32%       22.16      0.000212  
80,000     1.2903         1.5461         55.11%       21.80      0.000177  
81,000     1.2981         1.5412         55.32%       22.09      0.000142  
82,000     1.2981         1.5420         55.38%       21.99      0.000106  
83,000     1.2863         1.5395         55.40%       22.00      0.000071  
84,000     1.2895         1.5381         55.63%       21.70      0.000035  
85,000     1.2816         1.5425         55.38%       22.34      0.000030  
--------------------------------------------------------------
Total time: 1603.0s  (18.86 ms/sample avg, 85,000 samples)
Checkpoint saved: C:\repos\_Neuro\neuro-fabric\run\results\exp-gpu-adam-shakespeare-334k-85k.neuro
```
