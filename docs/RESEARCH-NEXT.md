## FUTURE STEPS 

1. Mixture of Experts (MoE) Transformer

> **Goal**: scale effective reasoning capacity to ~1M params using a shared-embedding MoE topology: one 200K gating chip handles embedding + routing + its own transformer stack; nine 100K expert chips are pure transformer layers with no embedding overhead. Only one expert chip activates per token — sparse at inference, full utilisation at training.

### Why naive MoE (N full TransformerBus instances) fails for large vocab

If each expert is a full `TransformerBus` with its own embedding, the vocab tax applies N times:

- TinyStories vocab=1501: $64 \times 1501 = 96K$ per expert → only ~4K reasoning params per expert
- 10 experts × 4K = 40K total reasoning — marginally better than single model's 19K, not worth the complexity

**Fix**: shared embedding on a dedicated gating chip. Embedding cost paid **once**.

### Architecture — 200K gating chip + 9× 100K expert chips

```
[Token]
    ↓
[Gating chip — 200K]
  ├─ Shared embedding: 64×vocab  (96K for TinyStories)
  ├─ Transformer layers: ~103K   (gating chip's own reasoning stack)
  ├─ Gate: Linear(64→9) + Softmax  (~0.5K)
  └─ Output projection: weight-tied to embedding (free)
         ↓ 64-dim hidden state
[Expert chip i — 100K]           ← selected chip only; others idle
  └─ Pure transformer layers: 100K  (no embedding)
         ↓ 64-dim hidden state back to gating chip
[Output projection on gating chip → vocab logits]
```

**Param budget breakdown:**

| Component | Params |
|---|---|
| Gating chip: shared embedding (vocab=1501) | 96K |
| Gating chip: transformer stack | ~103K |
| Gating chip: gate weights | ~0.5K |
| 9× expert chips (pure transformer) | 9 × 100K = 900K |
| **Total** | **~1.1M** |
| **Active per token** | ~203K reasoning + 96K embedding |

### Why this helps for TinyStories

| Model | Reasoning params | Est. loss | Est. accuracy |
|---|---|---|---|
| Single 115K (Day 3 baseline) | 19K | ~2.9 | ~37% |
| Naive 10× full TransformerBus MoE | 40K | ~2.6 | ~42% |
| **200K gating + 9×100K experts** | **~203K active** | **~1.8–2.2** | **~55–65%** |
| Human-level small LM (reference) | — | ~1.5 | ~70%+ |

The single-model TinyStories failure was not marginal — 19K reasoning params is so small the model cannot form coherent patterns. Going to 203K active reasoning params crosses a real threshold. Whether it reaches fully coherent story generation is uncertain without running it; the improvement over the baseline should be large and measurable.

### Training

- Loss: $L_{total} = L_{task} + \alpha \cdot L_{balance}$
- $L_{balance}$ penalises uneven expert utilisation — prevents expert collapse
- $\alpha = 0.01$
- All experts train in the same sequential loop; each expert receives gradient only on steps where it was selected
- Gating chip trains on every step (embedding + gate gradients always flow)

### Implementation plan

> **Principle**: Day 1–3 codebase (`Neuro.Attention`, `Neuro.Cpu.Optimized`, all tests) stays untouched. MoE lives in a new project `Neuro.Moe` that references `Neuro.Attention` and wraps/extends its types.

- [ ] New project `Neuro.Moe` in `neuro-fabric/src/` — references `Neuro.Attention` and `Neuro.Infrastructure`
- [ ] `GatingChip` class — subclasses or wraps `TransformerBus`; adds shared embedding, N-way gate (`float[,] _gateW`, shape `embedDim×numExperts`), output projection (weight-tied)
  - `float[] Embed(int tokenId)` — returns 64-dim embedding vector
  - `int Route(float[] hiddenState)` — runs gate, returns expert index
  - `float[] Project(float[] hiddenState)` — returns vocab logits
- [ ] `ExpertChip` class — wraps `TransformerBus` but overrides forward to accept a 64-dim hidden state directly (bypasses embedding lookup)
  - `float[] Forward(float[] hiddenState)` — transformer layers only, no embed/project
- [ ] `MoESystem` — composes one `GatingChip` + N `ExpertChip` instances
  - `TrainStep(int[] tokens, int[] targets, float lr)` — embed → gate chip transformer → route → expert forward → project → cross-entropy + balance loss → backprop
  - `Sample(int seed, int length, float temperature)` — autoregressive generation
  - `Save(string dir)` / `Load(string dir)` — saves gating chip + each expert chip as separate `.neuro` files
- [ ] Load balance loss in `MoESystem`: accumulate $f_i$ over batch, compute $L_{balance} = N \cdot \sum f_i \cdot p_i$
- [ ] New `Neuro.Moe.TrainApp` (or extend existing `TrainApp` with `--moe` flag + `--num-experts N`)
- [ ] Test project `Neuro.Moe.Tests` — `MoE_ReducesLoss_FasterThanSingle`, `ExpertLoad_IsBalanced`, `Save_Load_Roundtrip`
- [ ] Benchmark: train on TinyStories 20K steps, compare loss/accuracy vs single 115K baseline from Day 3
- [ ] Demo: compare generated text quality vs Day 3 TinyStories output

### FPGA mapping

```
[Host controller = Gating chip (200K)]
  ↓ selects expert i
[Expert chip 0] [Expert chip 1] ... [Expert chip 8]
  ← only chip i activates; others draw zero compute power →
```

At inference: gating chip embeds token → runs its transformer → gate selects chip i → hidden state sent to chip i → chip i runs its transformer → hidden state returned → gating chip projects to logits. One bus transfer each way, N-1 chips idle.

---
