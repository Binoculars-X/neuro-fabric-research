# MoE Scaling Considerations
<!-- Written: 12/06/26, based on EXP-010 results and architecture analysis -->

---

## Weight Split Across Expert Chips

Each expert is a fully independent `AdamTransformerBus` with its own weights, Adam moments, and optimizer.

**Per-expert chip (embed=64, heads=2, ff=192, layers=4, vocab=256):**
| Component | Params |
|---|---|
| Embedding (256×64) | 16,384 |
| 4× Attention layers (QKV + proj) | ~65,536 |
| 4× FF layers (64→192→64) | ~98,304 |
| Unembedding (64×256) | 16,384 |
| **Total per expert** | **~196K** |

**Coordinator chip (shared):**
| Component | Params |
|---|---|
| `_coordEmbed` (256×64) | 16,384 |
| `_router` linear (64→4) + softmax | ~260 |
| `_coordUnembed` (64×256) | 16,384 |
| **Total coordinator** | **~33K** |

**Data crossing chip boundaries:**
- Forward: coordinator embeds token → `[seqLen, embedDim]` → sent to top-k active experts → weighted sum of expert outputs → logits
- Backward: each active expert receives only its `dHidden` gradient slice — no inter-expert gradient flow

---

## Why Top-2 Routing?

Top-k=2 is a deliberate tradeoff, not a hardware constraint:

- **Not top-1:** Router collapse risk — router quickly learns to always pick the same expert; other experts stop receiving gradients and die.
- **Not top-all:** Defeats the purpose — full compute cost with no parallelism benefit.
- **Top-2 benefits:**
  - Each token gets two expert "opinions" → richer representation
  - Exactly 2 chips active simultaneously → can run in parallel on real FPGA hardware
  - Active power = 50% of total chip array per token
  - All experts trained roughly equally over many tokens → no collapse
  - Same choice as Shazeer et al. 2017 ("Outrageously Large Neural Networks") for identical reasons

---

## Ideal top-k as a Function of N Experts

Research consensus: **top-k ≈ √N**, with a floor of 2.

| N experts | √N | Practical top-k | Active% |
|---|---|---|---|
| 4 | 2 | **2** | 50% |
| 9 | 3 | **3** | 33% |
| 16 | 4 | **4** | 25% |
| 64 | 8 | **6–8** | 9–12% |
| 100 | 10 | **8–10** | 8–10% |
| 256 | 16 | **12–16** | 5–6% |
| 1024 | 32 | **16–32** | 2–3% |

**Hard lower bound:** each expert must see ≥2–3% of all tokens (= k/N) to receive enough gradient signal. Below this, experts die without an auxiliary load-balancing loss.

**Hard upper bound:** above ~25% active chips, the sparse MoE advantage disappears.

---

## Does MoE Prove Infinite Scale?

**No — but it proves parameter scaling decoupled from active compute**, which is a fundamentally different scaling axis than dense models.

### What EXP-010 proves:
- 4 isolated buses cooperating via a router match near-1M quality at ~400K active params/token
- The isolated-bus constraint (no weight sharing, tensor-only boundaries) works in practice
- Adding chips improves quality without proportionally increasing per-token FLOPs

### Practical scaling limits:

1. **Router bottleneck** — coordinator chip routes every token. At 1000+ experts, the router itself becomes a dense model and single point of failure. Hierarchical routing needed (route to cluster, then within cluster).

2. **Load balancing degrades** — with many experts and sparse top-k, some experts get starved. Requires auxiliary loss or capacity caps, complicating the "simple isolated chip" story.

3. **Coordinator bandwidth** — at N=1000, top-k=32: 32 tensor transfers per token = 32 × seqLen × embedDim × 4 bytes inter-chip traffic per step on real FPGA.

4. **Diminishing returns** — MoE scaling plateaus empirically. Each new expert specialises in increasingly rare token patterns; BPC improvement per added expert shrinks toward zero.

### The defensible paper claim:
> MoE enables parameter count to scale with the number of chips while keeping per-token active compute constant — a hardware-friendly scaling law. Practical limits exist at the router and interconnect, but these are engineering problems, not fundamental barriers.

---

## Large-Scale Example: 1T Params on 400M-per-Chip Hardware

**The setup:**
- 1T total params ÷ 400M params/chip = **2,500 chips**
- √N rule: top-k ≈ √2500 = **50 chips active per token**
- Active params per token: 50 × 400M = **20B active params/token** — GPT-3 class quality

**Hierarchical routing (required at this scale):**
```
Token → Level-1 router → pick 5 clusters (of 50 chips each)
       → Level-2 router → pick 10 chips within each cluster
       → 50 chips active total
       Router complexity: O(√N) not O(N)
```
This eliminates the 2500-way softmax bottleneck and the single-coordinator failure point.

**Remaining engineering challenges at 2500 chips:**

| Problem | Severity | Solution |
|---|---|---|
| Flat router (2500-way softmax) | ❌ Bottleneck | Hierarchical routing tree |
| Top-50 inter-chip bandwidth | ⚠️ Heavy | 50 × seqLen × embedDim tensor transfers per step |
| Load balancing across 2500 experts | ⚠️ Hard | Auxiliary loss + capacity factor per chip |
| Expert gradient staleness (2% token exposure) | ⚠️ Hard | Very large batches (millions of tokens/step) |
| Single coordinator SPOF | ❌ Fatal | Distributed coordinator tree |

---

## The 400M/Chip Limit Is Not a Capability Ceiling

The per-chip parameter limit is a **storage constraint**, solved by adding more chips via MoE. The remaining gaps to GPT-4-class models are **training methodology problems**, independent of chip architecture:

| Problem | Chip topology relevant? | What solves it |
|---|---|---|
| RLHF / alignment | ❌ No | Human feedback pipeline, reward model |
| 128K context window | ❌ No | Attention algorithm (Flash Attention, sliding window) |
| Multimodal (vision) | ❌ No | Vision encoder upstream of transformer |
| Expert gradient staleness | ⚠️ Partially | Larger batch size per step |
| Training stability at 2500 chips | ⚠️ Partially | Load balancing loss, capacity factor |

**NeuronFabric chips have no fundamental parameter ceiling.** The 400M/chip limit is overcome by MoE across N chips. The remaining gaps to GPT-4-class models are training methodology problems (RLHF, long context) that apply equally to GPU clusters — they are not unique to this architecture, and the chip design is not blocked by them.
