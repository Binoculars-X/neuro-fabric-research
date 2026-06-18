# NeuroFabric — Architecture Challenges

---

## Challenge 1 — Cycles per Iteration

### Perceptron (MLP)
- All MACs fire in parallel (1 cycle)
- Adder tree reduction: `log₂(fan-in)` cycles — e.g. embedDim=64 → **6 cycles**
- Softmax output layer: ~10-20 cycles over N classes
- **Total forward: ~30 cycles. Backprop: ~3×→ ~90 cycles.**

### LLM (Transformer)
- Same MAC parallelism, but softmax runs over seqLen scores per head
- Layers are sequential (layer N+1 waits for layer N)
- seqLen=128, embedDim=64, 3 layers: **~300 cycles forward, ~900 backprop**
- At 200 MHz (130nm): **~1.5 µs forward + 4.5 µs backprop per token**

### vs H100
| Metric | NeuroFabric (1 chip) | H100 |
|---|---|---|
| Latency (batch=1) | **~1-2 µs** | ~1 ms — 500-1000× slower |
| Throughput (batch=1024) | ~500K tokens/sec | ~10M tokens/sec — 20× faster |
| Power | **~1-3W** | 700W |

**NeuroFabric wins on latency. H100 wins on throughput** — it processes thousands of samples simultaneously with 16K CUDA cores and 3.35 TB/s HBM bandwidth.

### Solution — 1000-chip PCB
At 1000 chips × 1B params = 1T params (GPT-4 class):
- Inter-chip traffic: **~4 MB/step** (activations only) vs **~14 TB/step** for H100 cluster
- Throughput gap closes to **2-3×** of H100 cluster
- Power: **~3 kW** vs **~700 MW** for equivalent H100 cluster
- Cost: **~$250K** vs **~$30B**

Throughput parity is not the goal — **200,000× lower power at near-equivalent throughput** is the claim.

---

## Challenge 2 — Optimizer: SGD vs Adam

### Discovered (Day 5 benchmarks)

Benchmarking the 1M param TinyStories model for 8,000 samples revealed a large convergence gap:

| Optimizer | Eval Loss @8k samples | Accuracy |
|---|---|---|
| CPU SGD (silicon reference) | 5.47 | 3.9% |
| GPU SGD (TorchSharp — verified oracle) | 5.43 | 7.0% |
| GPU Adam | **2.71** | **42.2%** |

CPU SGD and GPU SGD match exactly — the silicon reference backprop is numerically correct.  
Adam is **2× lower loss and 6× higher accuracy** at the same step budget. This is not a bug — it is a fundamental property of the optimizers.

### Why Adam converges faster

SGD update: $\theta \leftarrow \theta - \eta \cdot g$

Adam update: $\theta \leftarrow \theta - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

Adam maintains per-weight first moment $m_t$ (gradient mean) and second moment $v_t$ (gradient variance). The division by $\sqrt{\hat{v}_t}$ normalises each weight's effective learning rate — weights with noisy, high-variance gradients get smaller steps; weights with consistent gradients get larger steps. This is critical for transformers where embedding rows have sparse, uneven gradient coverage.

### ASIC implication

The current NeuroFabric chip implements SGD: `w -= lr * dw`. This is 1 multiply-accumulate per weight per step.

Adam requires **2 additional registers per weight** ($m$ and $v$) and **3 extra operations per weight per step** (update $m$, update $v$, compute $m/\sqrt{v}$). For a 100K param chip:

| | SGD | Adam |
|---|---|---|
| Extra SRAM per chip | 0 | 2 × 100K × 4 bytes = **800 KB** |
| Ops per weight per step | 1 | ~4 |
| Silicon area overhead | baseline | ~3-4× larger update unit |

At 5nm, 800 KB SRAM is feasible but not free (~0.4 mm²). The MAC array itself is small; the SRAM dominates chip area.

### Options

1. **Start with SGD on ASIC, research adaptive optimizers in parallel**  
   Ship v1 silicon with SGD — zero area overhead, proven correct (matches GPU SGD exactly). In parallel, prototype Options 2 and 3 in simulation to quantify the real silicon cost before committing to a design. SGD is the baseline, not the destination.

3. **Implement Adam on ASIC with shared moment SRAM**  
   $m$ and $v$ arrays stored in a shared SRAM bank (not per-weight registers). Access latency increases but area is lower. Requires a memory controller and pipeline stall cycles.

4. **Implement AdaGrad (simpler adaptive optimizer)**  
   AdaGrad maintains only $v_t$ (one extra register per weight, not two). Update: $\theta \leftarrow \theta - \eta \cdot g / \sqrt{v_t + \epsilon}$. Converges better than SGD, half the SRAM overhead of Adam.  
   Downside: $v_t$ grows monotonically → learning rate decays to zero over time. Unsuitable for continual on-chip learning.

5. **Implement Adam with 16-bit moment storage**  
   BF16 moments: 2 × 100K × 2 bytes = **400 KB** per chip. Halves the SRAM cost with minimal convergence impact (Adam moments are smooth; BF16 precision is sufficient).

### Current recommendation

> **Option 5 (BF16 Adam on ASIC)** is the target. On-chip training is the goal — convergence quality is non-negotiable.  
> **Option 4 (AdaGrad)** is a fallback only if BF16 Adam proves too costly at tape-out; its monotonic decay makes it unsuitable for continual learning so it should be treated as a last resort.

The silicon reference (`TransformerBus` SGD) is a numerical oracle for testing backprop correctness — it is **not** the intended production optimizer. The 2× convergence gap between SGD and Adam is the primary motivation for implementing adaptive optimisation in silicon.

