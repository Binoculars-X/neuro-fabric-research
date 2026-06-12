# EXP-010 — MoE 4×200K, TinyStories, byte-level vocab=256, 250K samples

## Goal
Test Mixture-of-Experts routing with 4 experts × ~200K params each (~810K total params,
~400K active per token, topK=2). Compare against the dense 1M baseline (EXP-005 at 250K
samples) and the dense 200K single-chip model (EXP-006).

## Config
| Parameter | Value |
|---|---|
| Architecture | MoE (Mixture of Experts) |
| Experts | 4 |
| topK | 2 (active per token) |
| Params per expert | ~200K (EXP-006 chip size) |
| Total params | ~810K |
| Active params/token | ~400K |
| embedDim | 64 |
| heads | 2 (head_dim=32) |
| ff | 192 |
| layers | 4 |
| vocab | 256 (byte-level) |
| batchSize | 32 |
| samples | 250,000 |
| LR | 0.003 linear decay |
| warmup | 500 steps |
| dataset | TinyStories (22.5M tokens, 90/10 split) |
| hardware | GPU Adam FP32, RTX 4090 |

## Results

| Samples | BPC | Train Loss | Eval Loss | Accuracy | ms/sample |
|---|---|---|---|---|---|
| 4,992 | 3.851 | 3.4877 | 2.6690 | 27.34% | 24.46 |
| 49,920 | 1.784 | 1.2483 | 1.2368 | 62.62% | 24.90 |
| 124,800 | 1.496 | 1.0309 | 1.0368 | 67.98% | 22.41 |
| **249,984** | **1.330** | **0.9190** | **0.9218** | **71.80%** | **1.91** |

## Key observations
- Final eval loss **0.922, BPC 1.330** at 250K samples
- Outperforms dense 1M at same sample count (EXP-005@250K: eval loss 0.880, BPC 1.302) — close but not ahead
- Decisively outperforms the dense 200K single model (EXP-006: eval loss 1.060, BPC 1.530)
- ms/sample ~22 during training (vs ~1.4 for dense 1M) — ~15× slower per sample due to expert routing overhead
- MoE routing adds significant compute cost; benefit over dense at same total params is modest at this scale

## Comparison — 250K samples, TinyStories byte-level

| Experiment | Architecture | Total Params | Active Params | Eval Loss | BPC | Accuracy |
|---|---|---|---|---|---|---|
| EXP-007 | Dense | 110K | 110K | 1.157 | 1.669 | 65.33% |
| EXP-006 | Dense | ~200K | ~200K | 1.060 | 1.530 | 67.59% |
| EXP-010 | MoE 4× | ~810K | ~400K | **0.922** | **1.330** | **71.80%** |
| EXP-005 | Dense | ~1M | ~1M | 0.865 | 1.248 | 73.55% |

## Run folder
`neuro-fabric/run/gpu-fp32-moe4x-tinystories-1000k-b32-250k/`
