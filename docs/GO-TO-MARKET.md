# NeuronFabric — Go-To-Market Strategy

---

## Team Strengths

| Skill | Status |
|---|---|
| C# / .NET architecture | ✅ In-house |
| AI-assisted development | ✅ Active |
| FPGA engineering (10+ years) | ✅ In-house |
| HDL (VHDL / SystemVerilog) | ✅ In-house |
| Open source community building | Target |

HDL blocker — the single most common failure point for hardware startups at this stage — is already resolved.

---

## Architecture Differentiators (Three Unique Claims)

| Claim | Detail | Competitor status |
|---|---|---|
| **On-chip backpropagation** | Training runs on the chip itself — milliwatts, not megawatts | No competitor ships this |
| **Zero weight traffic** | Weights never leave their compute unit — no NVLink, no InfiniBand, no HBM | Groq/ANE inference-only; still have weight traffic |
| **Persistent Flash memory** | Weights survive power-off — droid learns permanently, no cloud retraining | Mythic AI has Flash inference; nobody adds backprop |

**The droid pitch:** *"A 70B parameter brain that fits in a chest cavity, runs 50 hours on a 10 kWh battery, learns your family's habits permanently, and never phones home."*

---

## Strategic Choice: Open Source First

Delay VC investment as long as possible. Build community and credibility first. Raise from the people who understand the technology.

### Precedents

| Project | Model | Outcome |
|---|---|---|
| RISC-V | Open ISA, Berkeley research | SiFive ($200M+), Esperanto, Western Digital built on top |
| LLVM | Open compiler infrastructure | Apple/Google/Nvidia adopted it; original team funded/acquired |
| Linux | Free kernel | Red Hat sold for $34B; $50B ecosystem |
| RISC-V International | Community consortium | 10B+ RISC-V cores shipped by 2023 |

Open source silicon gets **OpenMPW priority** — Google prefers open designs for their free shuttle program. This directly accelerates Phase 3.

---

## Funding Path

| Stage | Amount | Source | Trigger |
|---|---|---|---|
| Now | $0 | Own time | — |
| Phase 1 | $500–$2K | Own pocket | Buy FPGA board |
| Phase 2 | $0 | Own time + AI tooling | — |
| Phase 3 | $0–$50K | Own capital + community | OpenMPW shuttle (free) or MOSIS fallback |
| **Community raise** | **$50–$200K** | Crowd Supply / Wefunder / Republic.co | After Phase 1 FPGA benchmark published |
| Series A | $5–20M | VCs / angels | After Phase 3 working silicon |
| Series B/C | $50–200M | Institutional | After commercial traction |

**No VC money until working silicon exists.** Community funding covers the gap between Phase 3 and Series A.

### Community Investment Platforms

| Platform | Type | Limit | Best for |
|---|---|---|---|
| **Crowd Supply** | Hardware pre-orders / crowdfunding | No limit | Dev board campaign after FPGA demo |
| **Wefunder** | Equity crowdfunding (Reg CF) | $5M | Technical community investors |
| **Republic.co** | Equity crowdfunding (Reg CF) | $5M | Broader tech audience |
| **GitHub Sponsors** | Recurring donations | No limit | Ongoing open source maintenance |
| **Hackaday.io** | Community + audience building | $0 raised but drives Crowd Supply | Pre-launch audience |

---

## Open Source Execution Plan

### Step 1 — Polish GitHub repo (now)
- Clean README with architecture diagram
- Link to ROADMAP.md and benchmark results
- 97% MNIST accuracy result prominently featured
- Clear "what makes this different" section (training power claim)

### Step 2 — Hackaday.io project page (after Step 1)
- Plain-English explanation of the training power claim
- "Train a transformer at milliwatts, not megawatts"
- Links to GitHub, invites contributors
- Builds audience before Crowd Supply launch

### Step 3 — Publish Phase 1 FPGA benchmark (after Phase 1)
- Measured training power on FPGA vs GPU
- Video of the board running, multimeter showing watts
- This is the moment the community becomes invested

### Step 4 — Crowd Supply campaign (after Phase 1 benchmark)
- Sell NeuronFabric FPGA dev kits (Kintex-7 + interface board)
- Target: $50–200K
- Backers become community members, bug reporters, early customers
- Funds Phase 3 silicon without diluting equity

### Step 5 — OpenMPW application (after Phase 2)
- Submit open HDL design to Google shuttle
- Open designs get preference
- Community credibility strengthens the application
- 5,000 GitHub stars > academic affiliation for shuttle acceptance
- **Include external SPI Flash interface on PCB** — weights survive reboot, demonstrates persistent learning claim

---

## Why Open Source Doesn't Kill the Business

| Concern | Reality |
|---|---|
| Someone copies the architecture | They'd need the FPGA engineer, the software stack, and the community trust you built. Code alone isn't the moat — execution is. |
| Nvidia forks it | That's validation worth more than any patent. You'd be in every press release as the origin. |
| No revenue before Phase 4 | Community funding + dev kit sales cover costs through Phase 3 |
| IP exposure | File a provisional patent ($1,500) before publishing. Locks your priority date. |

---

## Series A Pitch (Post-Silicon)

With working silicon and open source traction the pitch becomes:

> *"We built a chip that trains a transformer at 50 mW, survives power-off via on-chip Flash, and requires zero NVLink or InfiniBand to scale. No competitor does any of this — Groq, Cerebras, ANE, TPU are all inference-only and cloud-dependent. We have 5,000 GitHub stars, 200 contributors, 3 universities running our architecture, and a free silicon proof from Google's OpenMPW program. Put 70 of our Phase 4 chips in a droid and you get a 70B parameter brain that learns permanently on a 10 kWh battery. We need $10M to tape out the 5nm production chip."*

That pitch is fundable. The same pitch without the silicon and community is not.

---

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| OpenMPW shuttle rejection | Low | Apply every round (4–6/year); MOSIS fallback |
| Fixed-point convergence fails | Medium | Discovered in Phase 2b before any cash at risk |
| Community doesn't grow | Medium | Hackaday + Crowd Supply have built-in audiences for open hardware |
| Nvidia announces on-chip training | Medium | 3–5 year window; our $250 chip vs their $30K H100 is structural advantage |
| Series A market conditions | Low | Community traction + silicon proof is fundable in any market |
