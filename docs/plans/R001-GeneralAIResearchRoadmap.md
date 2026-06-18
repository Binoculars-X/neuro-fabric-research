# General AI Research Roadmap

## Status

Future research track.

This work is intentionally separated from the core NeuroFabric validation effort.

Current priority remains:

1. Functional Transformer implementation.
2. FPGA validation.
3. Performance and scaling evaluation.
4. Publication and peer review.

Only after the core architecture is validated should these research directions be explored.

---

# Motivation

Modern LLMs demonstrate impressive capabilities but differ significantly from biological learning systems.

Humans appear capable of:

* Continuous lifelong learning.
* One-shot and few-shot learning.
* Rapid memory formation.
* Long-term knowledge consolidation.
* Learning without global backpropagation.

The goal is not to reproduce biological mechanisms exactly, but to identify useful architectural principles that may improve machine intelligence.

---

# Research Theme 1: Episodic Memory

## Hypothesis

Not all knowledge should be stored in weights.

A dedicated episodic memory system may allow immediate storage and retrieval of new experiences without requiring weight updates.

## Questions

* How much knowledge can remain outside model weights?
* Can episodic memory reduce retraining requirements?
* Can recent experiences be stored with near-zero latency?

## Success Criteria

Model successfully uses new information immediately after insertion into memory.

---

# Research Theme 2: Sleep Consolidation

## Hypothesis

Learning may occur in two stages:

### Day Phase

* New experiences stored in episodic memory.
* No weight updates.

### Night Phase

* Replay important experiences.
* Train long-term model using Adam.
* Consolidate useful knowledge into weights.

## Questions

* Can replay reduce catastrophic forgetting?
* Can offline consolidation outperform continuous online training?
* What replay schedule is optimal?

## Success Criteria

Knowledge remains available after episodic memory removal.

---

# Research Theme 3: Importance-Based Replay

## Hypothesis

Not all experiences should receive equal replay effort.

Important experiences may receive more consolidation.

## Candidate Importance Signals

* Prediction error.
* Novelty.
* Rarity.
* Reward.
* User feedback.

## Success Criteria

Improved retention with lower replay cost.

---

# Research Theme 4: Local Learning

## Hypothesis

Global optimization may not be required everywhere.

Some learning may occur using local optimization domains.

## Questions

* Can local Adam updates approximate global training?
* Can learning be partitioned into independent regions?
* How much communication is actually required?

## Success Criteria

Reduced communication with minimal quality degradation.

---

# Research Theme 5: Lifelong Learning

## Hypothesis

An intelligent system should continuously learn without full retraining.

## Experiment

Long-running training simulation:

* Daily data ingestion.
* Episodic memory.
* Nightly consolidation.
* Periodic evaluation.

Duration:

* Months rather than hours.

## Success Criteria

Knowledge accumulation without catastrophic forgetting.

---

# Research Theme 6: Hybrid Memory Architecture

## Hypothesis

Intelligence may emerge from multiple memory systems rather than a single weight store.

### Candidate Layers

* Working memory.
* Episodic memory.
* Semantic memory (weights).
* Long-term archive.

## Questions

* What information belongs in each layer?
* How should information migrate between layers?

## Success Criteria

Improved learning efficiency compared to weight-only models.

---

# Guiding Principle

The objective is not to copy the brain.

The objective is to identify successful architectural principles from biological systems and combine them with modern optimization techniques such as Adam, transformers, retrieval systems, and future NeuroFabric hardware.

Biology is inspiration, not a constraint.
