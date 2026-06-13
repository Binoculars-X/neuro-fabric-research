# FPGA Neural Network — Demo Design

## Goal

Demonstrate that a bus-driven parallel neuron core architecture on FPGA:

1. **Runs faster** (lower latency) than an equivalent GPU at single-sample inference
2. **Consumes less power** at realistic inference workloads
3. **Achieves competitive accuracy** on well-known public datasets
4. **Scales efficiently** — 10K parallel neuron cores vs ~1K GPU tensor cores active per sample

---

## Key Claims to Prove

| Claim | Metric | Target |
|---|---|---|
| Lower latency | µs per inference | FPGA <1µs vs GPU ~50–200µs |
| Lower power | Watts at load | FPGA ~5W vs GPU ~150W |
| Better efficiency | Inferences per watt | FPGA 10–20× better than GPU |
| Competitive accuracy | % on test set | Within 1–2% of GPU baseline |
| True parallelism | All cores fire per clock | Demonstrated via cycle simulation |

---

## Architecture Under Test

- **7 layers** (matches human neocortex depth)
- **10,000 neuron cores** across hidden layers
- **Bus-driven**: single control signal dispatches all cores simultaneously
- **Local weights**: each core holds weights in BRAM (no shared memory bus contention)
- **Fixed-point arithmetic**: Q8.8 or Q1.15 (FPGA-native, no FP32 overhead)

---

## Comparison Baseline

| Platform | Model | TDP | Tensor/CUDA Cores | Notes |
|---|---|---|---|---|
| **FPGA** | Xilinx Kintex-7 325T | ~5–15W | 10K neuron cores (parallel) | Our architecture |
| **GPU** | NVIDIA RTX 3080 | 320W (~150W at this load) | 272 tensor cores / 8,704 CUDA | ONNX Runtime + CUDA |
| **CPU** | Intel Core i9-13900K | ~125W | N/A | ML.NET baseline |

---

## Demo Stages

### Stage 1 — Correctness (IRIS Dataset)

**Dataset**: IRIS (`https://archive.ics.uci.edu/dataset/53/iris`)  
**Inputs**: 4 floats | **Classes**: 3 | **Samples**: 150

Goals:
- Train to >95% accuracy
- Verify forward pass produces correct predictions
- Verify backpropagation converges
- Verify weight save → load → inference round-trip is lossless

Pass criteria: accuracy ≥ 95%, loss curve decreasing, model file saves and reloads correctly.

---

### Stage 2 — Accuracy Benchmark (MNIST)

**Dataset**: MNIST (`https://yann.lecun.com/exdb/mnist/`)  
**Inputs**: 784 floats (28×28) | **Classes**: 10 | **Samples**: 70,000

Goals:
- Achieve ≥97% test accuracy
- Match published CPU/GPU accuracy baselines
- Demonstrate model export and reload at full accuracy

| Baseline | Accuracy |
|---|---|
| RTX 3080 (ONNX/CUDA) | ~99.3% |
| CPU (ML.NET) | ~98.5% |
| **FPGA target** | **≥97%** |

Pass criteria: test accuracy ≥97%, training completes in reasonable time on CPU prototype.

---

### Stage 3 — Latency & Power Benchmark (MNIST single-sample)

**Workload**: 100,000 single-sample inferences, no batching

Measure:
- Latency per inference (µs)
- Inferences per second
- Platform power draw (W)
- Inferences per watt

Expected results:

| Metric | FPGA (simulated) | RTX 3080 | CPU i9 |
|---|---|---|---|
| Latency (µs) | **<1** | ~65 | ~500 |
| Inf/sec | ~5–10M | ~15M | ~2M |
| Power (W) | **~5** | ~150 | ~125 |
| Inf/Watt | **~1–2M** | ~100K | ~16K |

Pass criteria: FPGA latency <5µs, inf/watt >10× better than GPU.

---

### Stage 4 — Real-Time Sensor Stream (UCI HAR)

**Dataset**: UCI Human Activity Recognition (`https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`)  
**Inputs**: 561 floats | **Classes**: 6 | **Samples**: 10,299

Goals:
- Simulate real-time sensor data stream (one sample per trigger)
- Demonstrate FPGA always-on inference at <1µs per sample
- Compare GPU overhead (kernel launch + PCIe) makes it impractical for single-sample streaming

Pass criteria: FPGA processes each sample before next sensor tick (target: <100µs budget per sample), GPU fails due to kernel launch latency.

---

## Why 10K FPGA Cores Beat 1K GPU Tensor Cores

```
GPU inference path (single sample):
  PCIe transfer       ~5–10µs
  CUDA kernel launch  ~5–20µs
  Warp scheduling     ~5–15µs  (waits for full warp to fill)
  Compute             ~1–5µs
  Memory round-trip   ~10µs    (weights in GDDR6, not on-core)
  ─────────────────────────────
  Total:              ~50–200µs per sample

FPGA inference path (single sample):
  Bus signal asserted ~1ns
  All 10K cores fire  1 clock cycle per layer × 7 layers = ~35ns at 200MHz
  No memory round-trip (weights in BRAM, on-core)
  ─────────────────────────────
  Total:              ~35–100ns = <1µs per sample
```

The GPU is **optimized for batches of thousands**. The FPGA is **optimized for one sample, right now**.

---

## Datasets Summary

| Dataset | Task | Inputs | Classes | Samples | Stage | URL |
|---|---|---|---|---|---|---|
| IRIS | Classification | 4 | 3 | 150 | 1 | https://archive.ics.uci.edu/dataset/53/iris |
| Breast Cancer Wisconsin | Binary | 30 | 2 | 569 | 1 | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| MNIST | Digit recognition | 784 | 10 | 70,000 | 2 & 3 | https://yann.lecun.com/exdb/mnist/ |
| Fashion-MNIST | Image classification | 784 | 10 | 70,000 | 2 | https://github.com/zalandoresearch/fashion-mnist |
| UCI HAR | Sensor / activity | 561 | 6 | 10,299 | 4 | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |

---

## C# Prototype Benchmark Structure

```
\neuro-fabric\
  ARCHITECTURE.md         ← system architecture
  DEMO.md                 ← this file
  src/
    NeuronCore.cs         ← individual core (weights, forward, backward)
    NeuronLayer.cs        ← layer of parallel cores
    NeuralBus.cs          ← bus controller (signal dispatch, model save/load)
    Activations.cs        ← ReLU, Sigmoid, Softmax
    LossFunctions.cs      ← MSE, CrossEntropy
  benchmarks/
    FpgaVsGpuBenchmark.cs ← latency + inf/watt comparison
    BenchmarkResult.cs    ← result record + console output
  datasets/
    IrisLoader.cs
    MnistLoader.cs
    HarLoader.cs
  models/
    trained_iris.bin      ← saved weight files (WeightWrite demo)
    trained_mnist.bin
```

---

## Success Definition

The demo is successful if it shows:

- [ ] Stage 1: IRIS accuracy ≥95%, model save/load verified
- [ ] Stage 2: MNIST accuracy ≥97%
- [ ] Stage 3: FPGA latency <5µs, >10× better inf/watt than RTX 3080
- [ ] Stage 4: Real-time sensor stream processed without latency budget breach
- [ ] Power story: FPGA draws <10W vs GPU minimum ~150W at this workload
