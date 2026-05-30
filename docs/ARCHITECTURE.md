# FPGA Neural Network Architecture

## Overview

Bus-driven parallel neuron core architecture, prototyped in C# for conversion to FPGA HDL.  
Each neuron is an isolated compute core with local weight storage (BRAM / NAND flash equivalent).

---

## Core Concepts

| Concept | Description |
|---|---|
| **Neuron Core** | Independent parallel execution unit with local weights |
| **Weight Storage** | Local BRAM per core (NAND flash in prototype) |
| **Main Bus** | Broadcasts `Forward`, `Backward`, `WeightRead`, or `WeightWrite` control signal to all cores |
| **Layer** | All cores in a layer execute concurrently on the same bus signal |
| **Connectivity** | Axon outputs → next layer synapse inputs only (no lateral interconnect needed) |

---

## Signal Flow

```
[Input]
   │
   ▼
[Bus: Forward signal]──────────────────────────────────┐
   │                                                   │
   ▼                                                   ▼
[Layer 0: N cores, parallel]      [Layer 0: N cores, parallel]
  Core 0: w·x + b → ReLU           (same, all fire simultaneously)
  Core 1: w·x + b → ReLU
  ...
   │
   ▼ axon outputs (float[])
[Layer 1: N cores, parallel]
   │
   ▼
[Output Layer → Loss]

────── Backward pass ──────

[Loss Gradient]
   │
[Bus: Backward signal]
   │
   ▼
[Layer N → Layer 0 (reverse)]
  Each core:
    delta = grad × ReLU'(z)
    weights -= lr × delta        ← BRAM write
    gradient_out = W^T × delta   ← passed to previous layer

────── Weight read (model export) ──────

[Bus: WeightRead signal]
   │
   ▼ all cores simultaneously
  Each core streams weights → bus → flat float[]
   │
   ▼
[NeuralBus.ExportModel() → byte[] → file / network]

────── Weight write (model import) ──────

[File / network → byte[] → float[]]
   │
[Bus: WeightWrite signal + payload slice per core]
   │
   ▼ all cores simultaneously
  Each core loads its weight slice → BRAM write
```

---

## Component Responsibilities

### `NeuronCore`
- Stores weights in local memory (BRAM on FPGA)
- **Forward**: computes `dot(inputs, weights) + bias → activation`
- **Backward**: computes delta, updates weights, returns gradient to previous layer
- Caches `z` (pre-activation) and output for backprop use

### `NeuronLayer`
- Contains N `NeuronCore` instances
- Receives bus signal and dispatches to all cores via `Parallel.For`
- **Forward**: collects per-core scalar outputs → `float[]` for next layer
- **Backward**: distributes per-core gradients, accumulates gradient sum for previous layer (adder tree on FPGA)

### `NeuralBus`
- Top-level controller
- Drives `Forward` pass: input → layer 0 → ... → output
- Drives `Backward` pass: loss gradient → last layer → ... → layer 0
- **`WeightRead`**: collects all layer weight dumps → serializes to `byte[]` → `SaveModel(path)`
- **`WeightWrite`**: deserializes `byte[]` → distributes weight slices to each layer/core → `LoadModel(path)`

---

## Activation Functions

| Function | Formula | Used At |
|---|---|---|
| ReLU | $\max(0, x)$ | Hidden layers (default) |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | Binary output |
| Softmax | $\frac{e^{x_i}}{\sum e^{x_j}}$ | Multi-class output |

---

## Weight Serialization (Model Copy)

| Operation | Signal | Description |
|---|---|---|
| Export / Save | `WeightRead` | Each core streams weights onto bus → flat `float[]` → `byte[]` → file |
| Import / Load | `WeightWrite` | File → `byte[]` → `float[]` → bus slices each core's BRAM |
| Format | — | Raw `float` array, layer-ordered, core-ordered within layer |
| FPGA | — | DMA burst read/write; all BRAM blocks addressed by core ID |

Enables trained model snapshots, transfer learning, and hot-swap of model weights without retraining.

---

## Backpropagation

1. **Forward pass** — cache `z` and `a` per core
2. **Loss gradient** — computed at output layer (`MSE` or `CrossEntropy`)
3. **Per-core delta** — `δ = grad × σ'(z)`
4. **Weight update** — `w -= lr × δ` (BRAM write)
5. **Propagate gradient** — `grad_out = W^T × δ` (passed back via bus)

---

## FPGA Mapping

| C# | FPGA |
|---|---|
| `NeuronCore` | DSP slice group + BRAM block |
| `float[]` weights | Local BRAM (18K / 36K blocks) |
| `Parallel.For` | Concurrent `always` blocks |
| `NeuronSignal` (enum) | 2-bit control wire, broadcast (`00`=Fwd, `01`=Bwd, `10`=WRead, `11`=WWrite`) |
| Gradient accumulation | Pipelined adder tree |
| `float` (32-bit) | Fixed-point `Q8.8` or `Q1.15` |

---

## Interconnect Decision

- **Same-layer cores**: no lateral interconnect required
- **Layer-to-layer**: axon outputs only (feedforward bus)
- **Recurrent / attention**: requires additional interconnect fabric (not in scope for feedforward prototype)

---

## Recommended Layer Count

Biological reference: the human neocortex has **6 cortical layers** with **7–9 processing stages** end-to-end including subcortical relay.

> **Design target: 7 layers** — matches cortical depth, fits mid-range FPGA (Artix-7 / Kintex-7), sweet spot before vanishing gradient requires residual connections.

| Layers | Use |
|---|---|
| 1 (input) | Raw feature input |
| 2–6 (hidden) | Feature extraction and abstraction |
| 7 (output) | Classification / regression |

---

## Demo Datasets

### Training & Validation Targets

| Dataset | Task | Inputs | Classes | Samples | Demo Stage |
|---|---|---|---|---|---|
| **IRIS** | Classification | 4 floats | 3 | 150 | Stage 1 — correctness |
| **Breast Cancer Wisconsin** | Binary classification | 30 floats | 2 | 569 | Stage 1 — correctness |
| **Wine Quality** | Classification | 11 floats | 10 | 6,497 | Stage 1 — correctness |
| **MNIST** | Digit recognition | 784 floats | 10 | 70k | Stage 2 — benchmark |
| **Fashion-MNIST** | Image classification | 784 floats | 10 | 70k | Stage 2 — benchmark |
| **UCI HAR** | Sensor activity | 561 floats | 6 | 10,299 | Stage 3 — power proof |

### Dataset Sources

```
MNIST         https://yann.lecun.com/exdb/mnist/
Fashion-MNIST https://github.com/zalandoresearch/fashion-mnist
IRIS          https://archive.ics.uci.edu/dataset/53/iris
Wine Quality  https://archive.ics.uci.edu/dataset/186/wine+quality
Breast Cancer https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
UCI HAR       https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
```

---

## Demo Benchmark Plan

### Stage 1 — Correctness
- Dataset: IRIS (4 inputs, 3 outputs, 7 layers)
- Goal: training converges, accuracy >95%, weights save/load round-trip verified

### Stage 2 — Accuracy Benchmark
- Dataset: MNIST (784 inputs, 10 outputs)
- Goal: match known CPU/GPU accuracy baselines, measure inference time per sample

### Stage 3 — Power Proof
- Dataset: UCI HAR (real-time sensor stream simulation)
- Goal: demonstrate FPGA inference power vs. GPU/CPU

### Power Comparison Targets

| Platform | Inference Power | MNIST Accuracy |
|---|---|---|
| NVIDIA RTX 3080 | ~320W | ~99.7% |
| Intel Core i9 (CPU) | ~125W | ~99.3% |
| Raspberry Pi 4 | ~5W | ~98.5% |
| **Xilinx Artix-7 (target)** | **~1–3W** | **~97–99% achievable** |

Key claims to demonstrate:
- **Power**: 40–100× less than GPU, 40–60× less than CPU
- **Latency**: all-layer inference in single clock pipeline (no memory bottleneck)
- **Parallelism**: all cores in a layer fire simultaneously — O(1) per layer regardless of width
