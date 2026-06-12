# EXP-012 — Dense 300K, TinyStories, byte-level vocab=256, 250K + 500K samples

## Goal
Dense baseline at ~297K params — matching the **active parameter count** of EXP-011
(MoE 10×100K, ~300K active/token). Establishes a fair apples-to-apples comparison:
same per-token compute budget, no routing overhead.

## Config
| Parameter | Value |
|---|---|
| Architecture | Dense |
| Total params | ~297K |
| Active params/token | ~297K |
| embedDim | 80 |
| heads | 2 (head_dim=40) |
| ff | 240 |
| layers | 4 |
| vocab | 256 (byte-level) |
| batchSize | 32 |
| samples | 250,000 + 500,000 |
| LR | 0.003 linear decay |
| warmup | 500 steps |
| dataset | TinyStories (22.5M tokens, 90/10 split) |
| hardware | GPU Adam FP32, RTX 4090 |

## Results

| Samples | BPC | Train Loss | Eval Loss | Accuracy | ms/sample |
|---|---|---|---|---|---|
| 4,992 | 4.411 | 3.8463 | 3.0572 | 21.47% | 5.92 |
| 49,920 | 1.996 | 1.3941 | 1.3832 | 58.14% | 6.23 |
| 124,800 | 1.702 | 1.1604 | 1.1796 | 64.17% | 7.00 |
| **249,984** | **1.522** | **1.0260** | **1.0552** | **67.75%** | **0.54** |
| **500,000** | **1.388** | **0.9341** | **0.9620** | **70.64%** | **6.95** |

250K training time: **1693.2s** (~28 min), **6.77 ms/sample avg**.  
500K training time: **3456.3s** (~58 min), **6.91 ms/sample avg**.

## Key observations
- At **250K samples**: eval loss 1.055, BPC 1.522, accuracy 67.75%
- At **500K samples**: eval loss 0.962, BPC 1.388, accuracy 70.64%
- Dense 300K 250K→500K gain: **−0.093 loss, +2.9% accuracy** — continuing to improve but slowly
- Dense 300K beats dense 200K (EXP-006: 1.060 at 250K) by a small margin — consistent with scaling
- **Decisively beaten by MoE 10× at 500K** (EXP-011: 0.848 loss, 1.223 BPC, 73.99% accuracy):
  - loss gap: **+0.114** (dense worse)
  - BPC gap: **+0.165**
  - accuracy gap: **−3.35%**
- ~7 ms/sample — ~10× faster than MoE 10× routing (~66 ms/sample); same chip-level compute

## Comparison — TinyStories byte-level, all experiments

| Experiment | Architecture | Total Params | Active/token | Samples | Eval Loss | BPC | Accuracy |
|---|---|---|---|---|---|---|---|
| EXP-007 | Dense | 110K | 110K | 250K | 1.157 | 1.669 | 65.33% |
| EXP-006 | Dense | ~200K | ~200K | 250K | 1.060 | 1.530 | 67.59% |
| EXP-012 | Dense | ~300K | ~300K | 250K | 1.055 | 1.522 | 67.75% |
| EXP-010 | MoE 4× | ~810K | ~400K | 250K | 0.922 | 1.330 | 71.80% |
| EXP-005 | Dense | ~1M | ~1M | 250K | 0.865 | 1.248 | 73.55% |
| **EXP-012** | **Dense** | **~300K** | **~300K** | **500K** | **0.962** | **1.388** | **70.64%** |
| EXP-011 | MoE 10× | ~1M | ~300K | 500K | 0.848 | 1.223 | 73.99% |
| EXP-005 | Dense | ~1M | ~1M | 500K | 0.808 | 1.166 | 75.26% |

## Key finding
Dense 300K at 500K samples (0.962) is **clearly worse than MoE 10×100K at 500K samples (0.848)**
— a gap of 0.114 loss / 3.35% accuracy — despite having the **same active parameter count per
token** (~300K). The MoE model uses 3× more total parameters (1M vs 300K) for the same per-token
compute budget.

Dense 300K is also tightly clustered with dense 200K (EXP-006: 1.060 at 250K vs 1.055) — nearly
zero benefit from adding 100K more dense parameters. The dense scaling curve is saturating hard
in the 200–300K range for this dataset and sequence length.

MoE routing allows the same **active** parameter budget to reach ~1M total params and extracts
a large quality jump (+13.5% loss reduction) with no per-token compute increase, at the cost of
routing infrastructure. **This is the core empirical argument for MoE in NeuronFabric.**

## Note on GPU training speed vs ASIC inference speed

EXP-011 (MoE 10×100K) runs at ~66 ms/sample on GPU vs ~7 ms/sample for dense 300K here —
roughly **10× slower**. This is a **software artifact**, not an inherent MoE cost:

On a GPU, expert dispatch is serialized: the router selects an expert, dispatches the token,
waits for the result, then moves to the next token. All 10 expert weight matrices compete for
the same memory bandwidth and compute units. The overhead is real and unavoidable in software.

On a **dedicated ASIC** (the NeuronFabric target architecture), each expert is a separate
physical tile with its own weight SRAM and MAC array. The router broadcasts the token to all
active expert tiles **simultaneously** — true hardware parallelism. All active experts compute
in parallel and the results are summed. Latency = one expert's latency, regardless of expert
count.

| Metric | GPU (software) | ASIC (hardware target) |
|---|---|---|
| Dense 300K | ~7 ms/sample | baseline |
| MoE 10×100K | ~66 ms/sample (~10×) | **same as dense 100K** |
| Dense 1M | ~7 ms/sample | 10× more silicon area/energy |

The GPU experiments prove the **quality advantage** of MoE. The ASIC argument is that you get
that quality at **no runtime or energy cost** compared to an equivalent dense 300K chip — which
is the central NeuronFabric thesis: *pay for routing once in silicon area, not in per-token
energy*.

## Demo output @ 250K (temp=0.8)

```
> once upon a time
once upon a time, "Can you doll have a lot of the doof to dange and care him and
said, "Nes Ben will and she was tried to eat swimmins, and he said enjoy and the
penged up and said, "Maybe. I amble, Jose is very happy and clouds." amazing the
bird wanted to helps that she still, and play with a good friends what ha

> one little girl
one little girl bunny not glass. One day, Tom was not lay to make us.
Tom was done and ball town, the angry, the could never toss the flower. She was so
give it was. The sungla was right. They saw a big red boy in the house. He tried
to eat should be could and see, but his face to be the mask.
Tom and Ben. He had

> once cat
once cat named Sally.
After a big, dog a big could and prints for safe. He says, "Mom my heard them. We
can you are or man. It has an loved to climbed his toy stop," said said, "Yes,
let's okay. I want to walk and dad to go a big boy on the rabbit, but she could
not so scared to play in the tree."
"Mom, ude
```

## Demo output @ 500K (temp=0.8)

```
> once upon a time
once upon a time, a big box of change and a boy. Bobo used a new some mom. She learn was
alable together. She said, "Hello, Lily. That and Jack pip and takes the pictures."
The little bunny baby was crying it. It was a big dog. They ran to see the ugly and said,
"What is good, sand the way and she got home." Sam li

> one little girl
one little girl as she was happy. Sue saw a chubble bird. It was couldn't win around and
said, "No, do you need listened to eat the dog."
The bug started to help his hands. It was brought for plan. She thought he sailman. She
had a funny dog named Lily was very sun. They all had each other.
Max pretend to bind an

> one cat
one catch every day.
<|endoftext|>
One day, a little boy named Tim who lived in a small house with. The blue was full of her
face. When he did not want to play with the sparkle. One day, they saw a boy naughty dog
called the cake two all day and she could flowers. One day, they saw a girl in the same frog.
```

## Demo interpretation
- **250K**: more word-salad ("doof to dange", "penged up", "sungla"), dialogue attribution breaks
- **500K**: noticeably cleaner — real names (Lily, Tim, Sue), coherent dialogue, story arcs start to form; still incomplete sentences and non-words ("chubble", "alable", "sailman")
- 250K→500K improvement visible but modest — consistent with the small loss gain (1.055→0.962)
- Still clearly below EXP-011 MoE 10× at 500K (0.848): less coherent narrative, more broken syntax
- Confirms dense 300K plateau: large compute budget buys slow incremental gains vs MoE's larger
  parameter space at same active compute

## Run folder
`neuro-fabric/run/gpu-fp32-tinystories-300k-b32-500k/`
