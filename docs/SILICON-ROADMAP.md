# NeuronFabric Silicon Roadmap
## From 1M-Param Proof to Brain-Scale Analogue CIM

---

## 1. The Core Architectural Insight

GPUs waste most of their energy **moving weights**, not computing with them.

Every forward pass, a GPU must stream all weights from HBM (off-chip DRAM) to compute cores:
- 100B param model × 2B (BF16) = 200 GB to stream
- HBM bandwidth: ~3 TB/s → **67ms just to read weights**
- Power profile: ~700W, mostly memory bandwidth

**NeuronFabric approach:** weights live permanently in on-chip SRAM, adjacent to compute units.
No streaming. No memory wall. Pure multiply-accumulate.

> *GPUs waste most of their energy moving weights. Our chip never moves them.*

---

## 2. Single-Die Capacity by Process Node

SRAM density scales with node shrink. A 400mm² die holds:

| Node | SRAM density | Params (BF16 Adam) | Est. power |
|---|---|---|---|
| 12nm | ~0.5 MB/mm² | ~33M | ~5W |
| 7nm | ~0.9 MB/mm² | ~66M | ~8W |
| 5nm | ~1.5 MB/mm² | ~100M | ~12W |
| 3nm | ~2.5 MB/mm² | ~170M | ~12W |
| **2nm** | **~3.5 MB/mm²** | **~250M** | **~12W** |
| 1.4nm | ~5.5 MB/mm² | ~370M | ~12W |

BF16 Adam memory per param: weights (2B) + moment m (2B) + moment v (2B) = **6B/param**.

The architecture improves automatically with every node shrink — no redesign required.

---

## 3. Pipeline Parallelism — Scaling Depth

A transformer pipeline maps naturally to a chip chain:

```
[Chip 1: Embedding + Layers 1–100] → [Chip 2: Layers 101–200] → ... → [Chip N: Layers + LM Head]
```

- Only **activations** pass between chips: `[seqLen × embedDim]` tensor per hop
- At embedDim=128, seqLen=128: **32 KB per token** between stages — trivial bandwidth
- Weights never move — they stay in each chip's SRAM permanently
- **All chips are identical silicon** — same mask, same fab run, configured by weight flash at boot

**Hard upper bound:** pipeline depth = number of transformer layers (1 chip per layer maximum).

**Latency per pipeline:**
- Die-to-die (UCIe, same package): ~10–50 ns/hop
- 1,000 hops × 50 ns = **50 µs total** — 200× faster than the brain's ~10 ms

---

## 4. Parallel Replicas — Scaling Throughput

Identical pipelines run in parallel for concurrent requests:

```
Pipeline lane 0: [C1]→[C2]→...→[CN]   ← request A
Pipeline lane 1: [C1]→[C2]→...→[CN]   ← request B
Pipeline lane K: [C1]→[C2]→...→[CN]   ← request K
```

This gives a **two-axis scaling model**:
- **Pipeline depth** (series) → more layers → deeper reasoning
- **Parallel replicas** (width) → more throughput → more concurrent users

Both axes scale **linearly** with chip count. No quadratic memory bandwidth wall.

---

## 5. Product SKU Ladder (5nm, 100M params/chip)

| Product | Pipeline depth | Replicas | Total chips | Total params | Power | Market |
|---|---|---|---|---|---|---|
| Edge module | 4 | 1 | 4 | 400M | ~50W | Car, phone base station |
| Enterprise node | 10 | 4 | 40 | 1B | ~500W | Hospital, factory |
| Datacenter blade | 10 | 16 | 160 | 1B | ~2kW | Cloud inference |
| Full rack | 10 | 64 | 640 | 1B | ~8kW | Large-scale serving |
| 1T system | 1,000 | 10 | 10,000 | 1T | ~120kW | Frontier AI |

**vs GPU equivalent for 1T params:** ~6,000 H100s × 700W = **~4 MW**
**NeuronFabric digital SRAM mesh:** ~120 kW — **33× less power**

---

## 6. Latency vs the Brain

The brain processes a full cortical inference cycle in ~10 ms.

| System | Token latency | vs brain |
|---|---|---|
| Human brain | ~10,000 µs | baseline |
| GPU H100 (DRAM bottleneck) | ~500 µs | 20× faster |
| NeuronFabric 5nm mesh (1T) | ~50 µs | **200× faster** |
| NeuronFabric 2nm mesh (1T) | ~20 µs | 500× faster |

**Latency is already solved.** The remaining race is purely **watts per synapse**.

---

## 7. Analogue CIM — The Next Frontier

**Compute-in-Memory (CIM):** instead of reading weights then multiplying, the MAC happens
*inside* the SRAM bitcell — input voltage × stored weight = output current. Kirchhoff's law
does the accumulation. Zero data movement, zero clock cycles for the multiply.

| Technology | Energy/MAC | Status |
|---|---|---|
| GPU (H100) | ~100 pJ | Production |
| Digital SRAM (NeuronFabric today) | ~1 pJ | Buildable now |
| Analogue SRAM CIM (forward pass) | ~0.01 pJ | Silicon proven (ISSCC 2023–25) |
| **Analogue CIM + analogue Adam** | **~0.001 pJ** | Research (3–5 years) |

### Analogue Adam viability

Adam's three operations per weight:
```
m  = β1·m + (1-β1)·grad          ← linear — analogue friendly ✅
v  = β2·v + (1-β2)·grad²         ← squaring — needs extra circuit ⚠️
w -= lr · m / (√v + ε)           ← sqrt + division — hybrid digital ⚠️
```

The sqrt/divide is ~5% of total Adam operations. 95% is linear — fits analogue CIM perfectly.

**Approximate analogue Adam** (active research 2024–25):
- Replace `√v` with `|grad|` → fully analogue (sign-gradient methods)
- β values baked into capacitor ratios
- 6-bit moment precision — NeuronFabric BF16 results justify this is sufficient

Each weight becomes an **RC cell**: capacitors store m and v, charge/discharge rates = β1, β2,
input current = gradient, output voltage = updated weight. No digital logic, no clock.

### Bridge from Paper 1 to analogue

> Paper 1 proves BF16 Adam (7-bit mantissa) matches FP32 Adam quality.
> → Justifies 6-bit analogue Adam precision.
> → Justifies the analogue CIM Adam cell design.
> → Which is Paper 4.

---

## 8. Brain-Scale Power Projections

### Digital SRAM mesh

| Params | NeuronFabric digital | GPU cluster |
|---|---|---|
| 1T | 120W | 4,000W |
| 10T | 1,200W | 40,000W |
| 100T | 12,000W | 400,000W |

### Analogue CIM mesh

| Params | Analogue CIM | Human brain |
|---|---|---|
| 1T | **2W** | — |
| 10T | **20W** | — |
| **100T** | **200W** | **20W (brain)** |

**100T params (= human brain synapse count) at 200W** — a laptop charger.

---

## 9. Brain Comparison at 100T Params

| Metric | Human brain | NeuronFabric analogue CIM |
|---|---|---|
| Synapses / params | ~100T | 100T |
| Power | 20W | ~200W |
| Inference speed | ~10 ms | ~50 µs (200×) |
| Continuous learning | Yes | Yes (on-chip Adam) |
| Physical volume | 1.2 litres | ~20 litres (rack) |

**Remaining gap:** 10× on power — closed by analogue RC Adam cells (Paper 4).
Reaching 20W at 100T matches the brain on **every metric simultaneously**.

---

## 10. Chip Count at 100T (2nm, 250M params/chip)

```
100T / 250M = 400,000 chips
Pipeline per column: 10,000 layers → 10,000 chips
Parallel columns: 40
Total: 400,000 chips
Fab output: ~400 wafers (= 1 week of modern fab production)
Physical form: server rack (3D stacked chiplets, UCIe interconnect)
```

---

## 11. Research Paper Roadmap

| Paper | Topic | Key claim |
|---|---|---|
| **Paper 1** (current) | BF16 Adam on silicon | BF16 Adam matches FP32 quality; `ApplyUpdate` = SRAM update unit |
| **Paper 2** | Depth vs width scaling | Deep narrow networks on multi-chip pipelines |
| **Paper 3** | Multi-chip mesh scaling | Pipeline × replica topology; 33× power saving vs GPU at 1T params |
| **Paper 4** | Analogue CIM Adam | RC cell weight update; 0.001 pJ/MAC; 200W at 100T params |

---

## 12. Concurrent Workload Observation

During development, CPU BF16 Adam and CPU SGD ran simultaneously on the same R9 9900x
with no measurable interference. Both workloads are memory-bound (not compute-bound),
access separate memory regions, and use different execution unit mixes.

**Silicon implication:** a multi-chip SRAM fabric can run multiple model instances
concurrently — inference, training, fine-tune — without interference. The on-chip memory
hierarchy naturally partitions workloads. This is concrete multi-tenancy support with
no hypervisor or memory controller overhead.

---

*Last updated: May 2026. Training runs active: CPU BF16 Adam → 100k samples, CPU SGD plateau run.*
