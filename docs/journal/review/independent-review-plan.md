# Independent Review Plan — NeuroFabric Paper 1

**Goal:** Before arXiv submission, get independent verification of every claim that could embarrass if wrong.

---

## What Needs Reviewing

| Area | Risk if wrong | Priority |
|---|---|---|
| Adam convergence numbers (1.5226 / 1.5477) | Core empirical claim | Critical |
| BF16W vs FP32 gap (+0.025) | Core contribution | Critical |
| FPGA SRAM arithmetic (3.34 MB fits) | Hardware claim | High |
| FPGA cycle/timing projections (150–200 MHz) | Feasibility claim | High |
| Paper Adam formulas (bc1, bc2) | Correctness | High |
| Demo output coherence ("coherent text") | Subjective claim | Medium |
| Vocabulary-budget table (3 domains) | Supporting evidence | Medium |

---

## Track 1 — Self-Verification (before external review)

Do this first so reviewers aren't finding basic arithmetic errors.

- [ ] Re-derive Adam bias correction formula independently; verify matches code
- [ ] Re-count 334K parameters from code; verify SRAM = 334K × 10 = 3.34 MB
- [ ] Verify seqLen=128 is consistent across paper, code, and logs
- [ ] Re-check vocabulary-budget table numbers from actual experiment logs
- [ ] Compare Shakespeare eval loss 1.5226 vs nanoGPT char-level published baseline (~1.47 for 10M params — our 334K should be higher, ~1.5 is plausible ✓)
- [ ] Verify train/val split is clean: no val tokens in training data

---

## Track 2 — FPGA Expert Review

**Target:** the FPGA developer who already reviewed (gave feedback on clock speed, LUT, SPI).

**Ask:** formal review of Section 3 (Architecture hardware mapping) and Section 4 (FPGA Training Target).

Specific questions for reviewer:
1. Is the 256-entry softmax LUT sufficient or does it need range decomposition?
2. Is 150–200 MHz realistic for this control complexity?
3. Is the BRAM block count estimate correct? (3.34 MB ÷ 36 Kb/block = ~755 blocks; ZCU102 has 912 — does this hold?)
4. Is the DSP48 utilisation estimate for BF16 FMA defensible?
5. Is activation recomputation feasible in this architecture (no extra SRAM for full activation store)?

**Deliverable:** written responses → update paper or add explicit caveats.

---

## Track 3 — ML Practitioner Community Review

**Channels (post simultaneously):**
- **Hacker News** — "Ask HN: feedback on small transformer training paper before arXiv"
- **r/MachineLearning** — "Pre-submission feedback request: 334K transformer, BF16W, local Adam"
- **Papers With Code Discord** — post draft link

**What to ask:**
- Is the Shakespeare eval loss plausible for 334K params, byte-level, 80K samples?
- Is the BF16W convergence result (+0.025 gap) surprising or expected?
- Any obvious bugs in the Adam implementation visible from the code?
- Is the vocabulary-budget analysis novel or well-known?

**Draft post:** see `review-post-draft.md`

---

## Track 4 — arXiv Endorser as Reviewer

**Status:** Prof. Cheung (CityU) — not eligible to endorse. Prof. Shi (UW) — awaiting reply. Dr. Ang Li (UW) — email sent.

**Strategy:** When someone agrees to endorse, ask in the same exchange:
> *"Would you also be willing to give brief technical feedback on the FPGA feasibility section? Even 10 minutes would be valuable."*

This costs nothing extra and sometimes converts an endorser into a real reviewer.

**Backup:** cold outreach to 3–5 arXiv cs.AR authors with recent FPGA ML papers.

---

## Success Criteria

Paper is ready to submit when:
- [ ] Self-verification complete (Track 1) — no arithmetic errors found
- [ ] At least one FPGA expert has reviewed Section 4 and confirmed or flagged claims
- [ ] Shakespeare loss number benchmarked against at least one published baseline
- [ ] arXiv endorsement obtained

---

## Status

| Track | Status |
|---|---|
| Self-verification | ⏳ Not started |
| FPGA expert review | ⏳ Awaiting response |
| Community review | ⏳ Not posted |
| Endorser review | ⏳ Awaiting replies (UW) |
