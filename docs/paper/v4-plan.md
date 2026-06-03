# v4-draft Plan — Scientific Hardening for arXiv Submission

Goal: prior-art disclosure + reproducible software reference. Not a benchmark claim, not a hardware revolution paper.

---

## Framing (must change)

**Remove:** "future revolutionary AI hardware" positioning  
**Replace with:** "software proof-of-concept + architectural prior art"

Exact positioning to defend:
1. Reproducible software reference architecture for local-update transformer training
2. Internal consistency demonstration (CPU BF16W ≈ GPU FP32 convergence)
3. Memory-budget analysis for FPGA feasibility
4. Prior-art disclosure for local Adam transformer training — explicitly scoped as *one concrete implementation*, not a claim of optimality

---

## Specific Changes — Review Each

### 1. Overclaiming novelty / industry state
- [ ] **Replace:** "Every neural chip we are aware of..."  
  **With:** "To our knowledge, publicly documented accelerator architectures generally separate training compute from optimizer state updates or rely on external memory/host orchestration."  
  *Reason: one obscure counterexample kills the claim*

### 2. Brain-energy comparison
- [ ] **Remove entirely:** "A human brain learns continuously at roughly 20W..."  
  *Reason: drags in neuroscience/energy debates we don't measure. Weakens paper.*

### 3. STDP claim
- [ ] **Replace:** "STDP cannot reproduce transformer training results."  
  **With:** "STDP-based learning systems have not demonstrated transformer-scale gradient training comparable to backpropagation-based LLM training."  
  *Reason: current wording is too absolute*

### 4. Text-generation quality claim
- [ ] **Replace:** "The model generates named Shakespearean characters, metered dialogue structure, and contextually coherent responses."  
  **With:** "The model produces recognizable Shakespeare-style dialogue structure and named-character patterns."  
  *Reason: "contextually coherent" is subjective and unverifiable*
- [ ] **Add BPC comparison:** Report eval loss as BPC (= loss / ln(2)). GPU b8 80k → eval loss 1.5316 → **2.21 BPC**. Compare honestly against Karpathy char-rnn (2015): ~1.3 BPC at ~3M params. Frame as: "not SOTA, but convergence validation of local-update scheme on a known benchmark."  
  *This converts a weakness into scientific honesty — reviewers respect this*  
  - [ ] **TODO code:** Add BPC column to TrainApp log output (evalLoss / Math.Log(2)) — so paper can cite directly from log ✅ done

### 5. FPGA projections — MOST DANGEROUS SECTION
- [ ] **Remove:** "~15W", "15–30 ms/sample" — no synthesis data to back these up  
- [ ] **Keep but reframe:** 150–200 MHz as a *target clock domain assumption*, not a measured result  
- [ ] **Reframe DSP/FMA unit counts** as: architectural analysis of *parallel DSP block utilization* — how many BF16 FMA ops can execute concurrently given non-blocking local SRAM access, at an assumed 150–200 MHz clock  
  *This is defensible as: "architectural throughput analysis under stated assumptions", not a claim of measured performance*  
  *Frame it as: "local SRAM eliminates memory-bandwidth bottleneck that would otherwise serialize DSP utilization"*  
- [ ] **Keep:** memory-budget arithmetic (exact byte counts — pure math, fully defensible)

### 6. Manifesto language
- [ ] **Replace:** "The ultimate vision..."  
  **With:** "A possible long-term architecture is a network of chips exchanging activations rather than optimizer state."

---

## What We Can Claim (defensible)

- ✅ "We implemented a self-contained transformer training system with local Adam updates."
- ✅ "The system converges on Shakespeare character-level training."
- ✅ "This paper is a software reference implementation and architectural disclosure, not a benchmark claim."
- ✅ Memory-budget arithmetic (w=BF16 + m=FP32 + v=FP32 = exact byte counts)

## What We Cannot Claim

- ❌ "Fully local Adam updates are numerically viable" — no comparison to published metrics
- ❌ Efficiency superiority over any existing hardware
- ❌ Scalability claims
- ❌ ASIC timing/power numbers

---

## Open Questions — Decide Before Writing

1. **FPGA section** — reframe as parallel DSP throughput analysis under 150–200 MHz assumption + non-blocking local SRAM. ✅ Agreed direction.

---

## Summary Verdict (from review)

Absolutely publishable as arXiv preprint **if**:
- speculative language pruned
- universal claims scoped carefully  
- FPGA projection numbers removed
- positioned as "software reference + prior art", not "hardware revolution"
