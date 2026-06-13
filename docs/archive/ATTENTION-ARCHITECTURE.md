# FPGA Attention Core Architecture

Extension of the feedforward bus architecture to support transformer-style self-attention.  
Same design principles: bus-driven, parallel cores, local BRAM weights.

---

## Key Difference from Feedforward

| | Feedforward | Attention |
|---|---|---|
| Core input | single float[] | sequence of float[] (tokens) |
| Core output | scalar | attention-weighted vector |
| Interconnect | axon → next layer only | all cores read full sequence (lateral read) |
| Bus signals | 2-bit (Fwd/Bwd/WRead/WWrite) | +1 bit for `Encode` pass (positional encoding) |

Lateral read is **read-only** — cores never write to each other's state.  
On FPGA: sequence stored in shared read-only BRAM block, all cores address it simultaneously.

---

## Components

### `EmbeddingLayer`
- Maps token index → dense float vector (embedding dim `d`)
- Weights: vocabulary × d matrix in BRAM
- One lookup per token, fully parallel across sequence positions

### `PositionalEncoding`
- Adds sinusoidal position vector to each token embedding
- Stateless — computed on-the-fly, no BRAM required
- FPGA: combinational logic, zero latency

### `AttentionCore`
- Holds three weight matrices: **Q**, **K**, **V** (each `d × d_head` in BRAM)
- **Forward**:
  1. Project each token: `q_i = Q·x_i`, `k_i = K·x_i`, `v_i = V·x_i`
  2. Score: `a_ij = dot(q_i, k_j) / √d_head`
  3. Softmax over j → attention weights
  4. Output: `∑_j a_ij · v_j`
- One core = one attention head
- All heads fire in parallel on the same sequence

### `AttentionLayer`
- N `AttentionCore` instances (N heads) — all read same sequence, fire concurrently
- Concatenates head outputs → linear projection (W_O matrix)
- Bus signal dispatches identically to `NeuronLayer`

### `FeedForwardLayer`
- Standard `NeuronLayer` applied position-wise after attention
- ReLU hidden, Linear output
- Identical to existing implementation

### `TransformerBus`
- Extends `NeuralBus` with sequence management
- Pass order per token sequence:
  1. `Encode` — embedding + positional encoding
  2. `Forward` — attention layers → feedforward layers → output projection
  3. `Backward` — gradient flows in reverse through same layers
- `WeightRead` / `WeightWrite` unchanged — all BRAM blocks addressable by core ID

---

## Signal Bus (3-bit extension)

| Signal | Bits | Description |
|---|---|---|
| `Encode`      | 000 | Embedding lookup + positional add |
| `Forward`     | 001 | Inference pass |
| `Backward`    | 010 | Backpropagation |
| `WeightRead`  | 011 | Export all weights |
| `WeightWrite` | 100 | Import all weights |

---

## Prototype Parameters

| Parameter | Value | Reason |
|---|---|---|
| Sequence length | 32 tokens | Fixed at compile time — fits in BRAM |
| Embedding dim `d` | 64 | Small enough for Artix-7 BRAM budget |
| Heads | 2 | `d_head = 32` each |
| Feedforward dim | 128 | Standard 2× multiplier |
| Vocabulary | 256 | Byte-level (all ASCII) — no tokeniser needed |
| Layers | 2 | Sufficient for character-level prediction |

---

## Demo Task

**Next-character prediction** on a small plain-text corpus (e.g. Shakespeare, ~1 MB).

| Metric | Target |
|---|---|
| Task | Predict next character (256-class classification) |
| Loss | Cross-entropy, Softmax output |
| Train perplexity | < 5.0 after 20 epochs |
| Inference | Single 32-token context window → next token |

---

## C# Project Structure

```
Neuro.Attention/
  AttentionCore.cs         ← Q/K/V projection + scaled dot-product attention
  AttentionLayer.cs        ← N heads parallel, output projection
  PositionalEncoding.cs    ← sinusoidal, stateless
  EmbeddingLayer.cs        ← token index → dense vector
  TransformerBus.cs        ← extends NeuralBus with Encode pass
  AttentionSignal.cs       ← 3-bit bus enum
```

`Neuro.Core` is unchanged — `FeedForwardLayer` reuses `NeuronLayer` directly.

---

## FPGA Mapping

| C# | FPGA |
|---|---|
| `AttentionCore` Q/K/V matrices | Three 18K BRAM blocks per head |
| Sequence BRAM | Single shared read-only 36K block |
| Parallel heads | Concurrent `always` blocks, one per head |
| Softmax over sequence | Pipelined across sequence length (fixed latency) |
| `PositionalEncoding` | Combinational ROM (LUT-based) |
| 3-bit bus | 3 control wires broadcast to all cores |
