# Related Work — Mixture-of-Experts Scaling

**Status:** Draft v0.1

**Purpose**

This document summarizes the evolution of Mixture-of-Experts (MoE) architectures with emphasis on expert granularity and discusses the positioning of NeuroFabric within this research direction.

---

# Executive Summary

Since the introduction of modern sparse Mixture-of-Experts architectures, research has evolved through three major stages:

1. **Large sparse experts (2017–2021)**
2. **Trillion-parameter sparse language models (2021–2023)**
3. **Fine-grained experts (2024–present)**

The most recent trend is particularly relevant to NeuroFabric. Rather than increasing the size of individual experts, several recent works argue for increasing the number of experts while making each expert smaller and more specialized.

NeuroFabric follows this trend but introduces an additional hardware constraint:

> Experts must be sufficiently small to support complete local training inside future FPGA/ASIC accelerators without external optimizer memory.

This objective differs fundamentally from existing GPU-oriented MoE systems.

---

# Evolution of Mixture-of-Experts

## 1. Sparsely-Gated Mixture-of-Experts (2017)

**Paper**

Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

**Contribution**

Introduced modern sparse expert routing.

Key ideas:

* sparse activation
* gating network
* only a small subset of experts is evaluated
* trillion-parameter models become theoretically possible

**Importance**

Foundation of all modern MoE architectures.

---

## 2. Switch Transformer (2021)

**Contribution**

First successful trillion-scale sparse Transformer.

Characteristics:

* Top-1 routing
* one expert per token
* simplified routing
* dramatically reduced communication overhead

Demonstrated models approaching **1.6 trillion parameters**.

The work proved that sparse activation can scale language models by more than an order of magnitude compared with dense Transformers.

---

## 3. GLaM (2021)

Generalist Language Model.

Characteristics:

* Top-2 routing
* 64 experts
* approximately 1.2 trillion parameters

The authors demonstrated that sparse activation can outperform dense models while using substantially less inference computation.

---

## 4. Mixtral (2024)

Open-weight production-quality MoE.

Characteristics:

* 8 experts
* Top-2 routing
* shared attention
* expert FFN layers

Although commonly described as "8×7B", the expert itself represents only the feed-forward network rather than an independent 7B language model.

Mixtral demonstrated that sparse models could achieve state-of-the-art quality while remaining practical for deployment.

---

## 5. DBRX (2024)

Databricks production MoE.

Characteristics:

* 16 experts
* Top-4 routing

DBRX focuses primarily on improving inference quality while maintaining computational efficiency.

---

## 6. DeepSeekMoE (2024)

One of the most influential recent MoE papers.

Primary contribution:

> Fine-grained experts.

Instead of increasing expert size, DeepSeekMoE proposes dividing knowledge among a much larger number of smaller specialists.

This represents an important conceptual shift compared with previous generations.

---

## 7. DeepSeek-V3 (2024)

Large-scale production MoE.

Characteristics:

* hundreds of routed experts
* shared experts
* advanced routing algorithms

The architecture further reinforces the trend toward increasing expert granularity.

---

## 8. PEER — Mixture of a Million Experts (2024)

Perhaps the strongest conceptual support for NeuroFabric.

Core idea:

Instead of a few large experts,

use

millions of extremely small experts.

The paper argues that increasing expert granularity may provide better scalability than continuously increasing expert size.

---

# Approximate Comparison

| Model              | Year | Total Parameters |             Experts | Active Experts | Main Goal                   |
| ------------------ | ---- | ---------------: | ------------------: | -------------: | --------------------------- |
| Sparsely-Gated MoE | 2017 |         research |                many |         sparse | Introduce sparse experts    |
| Switch Transformer | 2021 |       up to 1.6T |                many |          Top-1 | Trillion-scale training     |
| GLaM               | 2021 |             1.2T |                  64 |          Top-2 | Efficient sparse scaling    |
| Mixtral 8x7B       | 2024 |            46.7B |                   8 |          Top-2 | Practical open MoE          |
| DBRX               | 2024 |             132B |                  16 |          Top-4 | Production inference        |
| DeepSeekMoE        | 2024 |          various |                many |         sparse | Fine-grained experts        |
| DeepSeek-V3        | 2024 |             671B | 256 routed + shared |          Top-k | Large-scale sparse training |
| PEER               | 2024 |         research |           1,000,000 |         sparse | Extreme expert granularity  |

---

# Trends

The literature suggests several consistent trends.

## Increasing total model size

2017:

Millions to billions.

2021:

Trillion-parameter models.

2024:

Hundreds of billions using sparse activation.

---

## Increasing expert specialization

Early work focused primarily on reducing computation.

Recent work increasingly focuses on encouraging specialization among experts.

---

## Increasing expert granularity

Recent papers consistently move toward

* more experts
* smaller experts
* better routing

rather than simply increasing the size of each expert.

This trend appears in:

* DeepSeekMoE
* DeepSeek-V3
* PEER

---

# Positioning of NeuroFabric

NeuroFabric investigates a different optimization objective.

Existing work asks:

> How can sparse experts maximize model quality on GPU/TPU clusters?

NeuroFabric asks:

> How small can experts become while remaining fully trainable using completely local optimization inside FPGA/ASIC hardware?

This introduces hardware constraints not considered in previous MoE literature.

Target expert sizes are approximately

**20M–400M parameters**

which are motivated by accelerator memory capacity rather than cloud-scale compute efficiency.

---

# Comparison with Existing MoE

| Property          | Conventional MoE                 | NeuroFabric                   |
| ----------------- | -------------------------------- | ------------------------------ |
| Hardware          | GPU / TPU clusters               | FPGA → ASIC                    |
| Optimizer         | Centralized                      | Local Adam                     |
| Optimizer state   | External                         | Local                          |
| Primary objective | Compute efficiency               | Local trainability             |
| Expert size       | Typically billions of parameters | Target 20–400M                 |
| Memory scaling    | Cluster memory                   | On-chip memory                 |
| Research question | Sparse scaling                   | Hardware-local sparse training |

---

# Preliminary Conclusion

NeuroFabric should not be viewed as competing directly with existing MoE systems.

Instead, it explores a different point in the design space.

Current literature increasingly favors smaller, more specialized experts.

NeuroFabric extends this direction by introducing hardware-driven constraints that require experts to remain sufficiently small for complete local optimization without centralized optimizer memory.

This makes NeuroFabric complementary to existing MoE research rather than an alternative to it.

---

# References

## Foundational

1. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017.

2. Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR 2022.

3. Du et al., *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts*, ICML 2022.

---

## Recent Large MoE

4. Jiang et al., *Mixtral of Experts*, 2024.

5. Databricks, *DBRX Technical Report*, 2024.

6. Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, 2024.

7. DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024.

8. PEER: *Mixture of a Million Experts*, 2024.
