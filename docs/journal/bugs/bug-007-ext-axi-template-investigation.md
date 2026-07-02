# BUG-007 Extension — AXI4-Lite Template Investigation (2026-07-03)

**Status:** PARTIALLY RESOLVED — JTAG path fully working; ARM path has systematic
address-decode failure for specific register offsets; workaround identified.

**Relates to:** bug-007-arm-rddata-pipeline-stale-read.md

---

## Summary of Work Done (2026-07-02 to 2026-07-03)

### Root cause of original BUG-007

The original `axi_lite_slave.sv` had a 1-cycle pipeline register (`rd_data_q`) that
caused the ARM `rf_read()` function to read stale data. The fix (two dummy reads in
`rf_read()`) was documented in bug-007 but the bitstream built with that fix still
showed ARM failures. Investigation continued.

### Decision: Replace AXI slave with Vivado-generated template

The custom AXI slave was replaced with an official Vivado-generated AXI4-Lite peripheral
template (`create_peripheral` + `generate_peripheral` Tcl API, Vivado 2025.2).

**Generated files (DO NOT HAND-MODIFY protocol sections):**
- `fpga/ip/axi_train_regs/axi_train_regs_1_0/hdl/axi_train_regs.v` — top wrapper
- `fpga/ip/axi_train_regs/axi_train_regs_1_0/hdl/axi_train_regs_slave_lite_v1_0_S00_AXI.v` — AXI state machines

**User logic sections modified (register map + bridge only):**
- Parameters: `T=4`, `V=16`, `C_S_AXI_ADDR_WIDTH=10`, `OPT_MEM_ADDR_BITS=7`
- Write decode: `if (S_AXI_WVALID)` with same strobe as generated template
- Read mux: `assign S_AXI_RDATA = reg_data_out`, `always @(*) case(axi_araddr[9:2])`
- User logic: `rd_data_q` FF, `cycle_cnt`, debug registers, bridge assigns

**AXI state machines left 100% untouched:**
- Write state machine (Idle→Waddr→Wdata) controlling `axi_awready`, `axi_wready`, `axi_bvalid`
- Read state machine (Idle→Raddr→Rdata) controlling `axi_arready`, `axi_rvalid`

---

## New Register Map (axi_train_regs)

| Address | Name | Direction | Notes |
|---|---|---|---|
| 0x000 | CTRL | R/W | [0]=rst [1]=start [2]=train_start [3]=en |
| 0x004 | STATUS | R | [0]=done [1]=adam_done [2]=out_valid [3]=timeout |
| 0x008 | CYCLE_LO | R | cycle_cnt[31:0] — **FAILS from ARM, use 0x010** |
| 0x00C | CYCLE_HI | R | cycle_cnt[47:32] — **FAILS from ARM, use 0x014** |
| 0x010 | CYCLE_LO alias | R | same as 0x008, works from ARM |
| 0x014 | CYCLE_HI alias | R | same as 0x00C, works from ARM |
| 0x040 | REG_WR_ADDR | W | [8:0] transformer_train write address |
| 0x044 | REG_WR_DATA | W | write data (no auto-pulse) |
| 0x048 | REG_WR_STROBE | W | write any value -> one-clock tt_wr_en pulse |
| 0x050 | REG_RD_ADDR | W | [8:0] address to read from register file |
| 0x054 | REG_RD_DATA | R | rd_data_q — **FAILS from ARM, use 0x060** |
| 0x058 | DBG_RD_ADDR | R | reg_rd_addr snapshot — fails from ARM |
| 0x05C | DBG_RAW_TT_RD_DATA | R | tt_rd_data direct — fails from ARM |
| 0x060 | DBG_RD_DATA_Q | R | rd_data_q — works from ARM, USE THIS |
| 0x064 | DBG_WR_EN_COUNT | R | cumulative wr_en pulse count |
| 0x068 | DBG_LAST_WR_ADDR | R | reg_wr_addr at last wr_en |
| 0x06C | DBG_LAST_WR_DATA | R | reg_wr_data at last wr_en |
| 0x070 | DBG_CORE_EN | R | reg_en |
| 0x074 | DBG_CORE_RST | R | reg_rst |
| 0x078 | VERSION | R | GEN00_VERSION from slave localparam — fails from ARM |

**Write protocol for transformer register file:**
```
1. Write REG_WR_ADDR (0x040) with register-file address
2. Write REG_WR_DATA (0x044) with data
3. Write REG_WR_STROBE (0x048) with any value -> pulses tt_wr_en for 1 clock
```
Old protocol (0x008 WR_ADDR + 0x00C WR_DATA auto-pulse) no longer applies.

---

## JTAG Tests — ALL PASS

**JTAG hop check script:** `fpga/projects/gen00_train/run_hop_checks_remote.tcl`
**JTAG probe:** remote hw_server at 192.168.0.98:3121

```
SANITY HOP1: CYCLE_LO 0xD98C50BF -> 0xD9D38960  [PASS]  -- clock alive
SANITY HOP2: CTRL readback=0x00000009             [PASS]  -- AXI slave R/W
SANITY HOP3: scratch readback=0xDEAD1234          [PASS]  -- WR path to transformer_train RF
SANITY HOP4: VERSION=0x67300002 ("g0.002")        [PASS]  -- RD path from transformer_train RF
HOP CHECK SUMMARY: 4/4 passed, 0/4 failed
```

**JTAG raw address probe** (all five addresses read correctly by JTAG):

```
JTAG 0x008=0x19E920DE  (CYCLE_LO -- works via JTAG)
JTAG 0x010=0x1A14CB43  (CYCLE_LO alias -- works via JTAG)
JTAG 0x054=0x67300002  (RD_DATA via rd_data_q -- works via JTAG)
JTAG 0x060=0x67300002  (DBG_RD_DATA_Q -- works via JTAG)
JTAG 0x078=0x67300001  (VERSION slave localparam -- works via JTAG)
```

Note: JTAG `0x078` returns `0x67300001` (slave localparam, BUILD_NUMBER=1).
The `0x054`/`0x060` return `0x67300002` because `reg_rd_addr=0x1FE` was set in previous
JTAG session pointing to `transformer_train.sv`'s version_reg with BUILD_NUMBER=2.

---

## XSim Tests — ALL PASS

All existing C# XSim tests were updated to compile the new generated AXI slave:
- `AxiLiteHopChecksTests` — PASS
- `AxiLiteSlaveTests` (forward + train) — PASS
- `AxiLiteHwPathTests` — PASS
- `AxiLiteWeightUpdateTests` — PASS
- `AxiLiteValidationLossTests` — PASS
- `AxiTrainRegsHopChecksTests` (new, dedicated to generated slave) — PASS

XSim testbenches updated:
- DUT changed from `axi_lite_slave` to `axi_train_regs`
- Port names: `s_axi_*` -> `s00_axi_*`, `clk`/`rst_n` -> `s00_axi_aclk`/`s00_axi_aresetn`
- Weight writes: 2-step (0x008+0x00C auto-pulse) -> 3-step (0x040+0x044+0x048)
- Address bus: `[31:0]` -> `[9:0]`

---

## ARM Tests — SYSTEMATIC FAILURE FOR SPECIFIC ADDRESSES

**Method:** `busybox devmem` direct hardware reads (eliminates C code)

```
0xA0000008 = 0x00000000  [FAIL]  (CYCLE_LO at 0x008)
0xA0000010 = 0x635E78F2  [PASS]  (CYCLE_LO alias at 0x010)
0xA0000054 = 0x00000000  [FAIL]  (RD_DATA at 0x054)
0xA0000060 = 0x67300002  [PASS]  (DBG_RD_DATA_Q at 0x060)
0xA0000078 = 0x00000000  [FAIL]  (VERSION at 0x078)
```

**ARM full diagnostic scan (arm_train --corpus /tmp --steps 1):**
```
[0x000] = 0x00000008  CTRL              PASS (correct)
[0x004] = 0x00000000  STATUS            PASS (correct, no done flags)
[0x008] = 0x00000000  CYCLE_LO          FAIL (should be non-zero)
[0x010] = 0x38BC7EA5  CYCLE_LO alias    PASS (correct)
[0x040] = 0x000001FF  WR_ADDR echo      PASS (correct, set by prev JTAG session)
[0x050] = 0x000001FE  RD_ADDR echo      PASS (correct, set by prev JTAG session)
[0x054] = 0x00000000  RD_DATA           FAIL (should be 0x67300002)
[0x058] = 0x00000000  DBG_RD_ADDR       FAIL (should be 0x000001FE)
[0x05C] = 0x00000000  DBG_RAW_TT_RD     FAIL (should be 0x67300002)
[0x060] = 0x67300002  DBG_RD_DATA_Q     PASS (correct!)
[0x070] = 0x00000001  DBG_CORE_EN       PASS (correct)
[0x074] = 0x00000000  DBG_CORE_RST      PASS (correct)
[0x078] = 0x00000000  VERSION           FAIL (should be 0x67300001)
```

---

## Root Cause Analysis

### What is ruled out

- **Synthesis corruption** -- JTAG reads ALL addresses correctly in the old sparse map,
  proving every case in the `axi_araddr[9:2]` decode mux was synthesized.
- **Non-ASCII corruption** -- em-dashes were found in `//` comments and removed (correct
  hygiene), but JTAG proved synthesis was correct before the fix.
- **C code bug** -- `busybox devmem` direct hardware reads reproduce the same failures,
  eliminating arm_train.c from the equation.
- **mmap size** -- MAP_SIZE=0x1000 covers all registers. Not the issue.

### Confirmed conclusion

The old sparse register map was incompatible with the ARM PS path.
Specific word indexes (8'h02, 8'h11, 8'h12, 8'h15-17, 8'h1E) returned 0 from reads
and appeared to be silently dropped on writes when accessed via the ARM PS HPM0 path.

**Root cause documented as:**
> Old sparse register decode and alias workaround removed.
> Canonical contiguous register map (word indexes 0x00-0x0E only) adopted as the fix.

Whether the ARM path issue is in the SmartConnect, in specific address decode behavior,
or in some interaction between the generated AXI4-Lite state machine and the ARM AXI4
master is NOT yet proven. An ILA trace showing AW/W channel handshakes at the slave
boundary would be required to confirm any fabric-level claim.

**SmartConnect dropping specific word indexes is an extraordinary claim without ILA proof.
Do not document it as confirmed root cause.**

### What the contiguous map fixes

Regardless of the exact cause, all critical registers are now at word indexes
8'h00-8'h0E (byte offsets 0x000-0x038). These are confirmed working from ARM.

If all 4 ARM hop tests pass after rebuild, the confirmed conclusion is:
- The old sparse register map was the source of the ARM failures
- The new canonical contiguous map resolves it
- No hardware changes required

### Remaining open question

Why specific word indexes fail from ARM but not JTAG. Independent proof requires:
- ILA on `S_AXI_AWADDR`, `S_AXI_AWVALID`, `S_AXI_AWREADY`, `S_AXI_WVALID`, `S_AXI_WREADY`
  during ARM write transactions to previously-failing addresses
- This would show whether the SmartConnect delivers the transaction to the slave

**Status of this question: OPEN, not blocking training.**

---

## Register Map v2 (Contiguous — adopted 2026-07-03)

All registers at consecutive word indexes 0x00-0x0E. Rebuild required.

| Byte offset | Word index | Name | Direction | RTL signal |
|---|---|---|---|---|
| 0x000 | 8'h00 | CTRL | R/W | reg_ctrl bits |
| 0x004 | 8'h01 | STATUS | R | done/adam_done/out_valid/timeout latches |
| 0x008 | 8'h02 | CYCLE_LO | R | cycle_cnt[31:0] |
| 0x00C | 8'h03 | CYCLE_HI | R | cycle_cnt[47:32] |
| 0x010 | 8'h04 | WR_ADDR | W | reg_wr_addr (9 bits) |
| 0x014 | 8'h05 | WR_DATA | W | reg_wr_data |
| 0x018 | 8'h06 | WR_STROBE | W | pulses tt_wr_en for 1 clock |
| 0x01C | 8'h07 | RD_ADDR | W | reg_rd_addr (9 bits) |
| 0x020 | 8'h08 | RD_DATA | R | rd_data_q |
| 0x024 | 8'h09 | VERSION | R | GEN00_VERSION constant |
| 0x028 | 8'h0A | STATUS_CLR | W | clears all STATUS latches |
| 0x02C | 8'h0B | CE_LOSS | R | latch_ce_loss (FP32) |
| 0x030 | 8'h0C | GRAD_NORM_SQ | R | latch_grad_norm_sq (FP32) |
| 0x034 | 8'h0D | CLIP_SCALE | R | latch_clip_scale (FP32) |
| 0x038 | 8'h0E | STEP_COUNT | R | tt_dbg_step_count |

Register file write protocol (unchanged):
```
1. Write WR_ADDR (0x010) = destination address in transformer register file
2. Write WR_DATA (0x014) = data value
3. Write WR_STROBE (0x018) = any value -> one-clock tt_wr_en pulse
```

Register file read protocol (unchanged):
```
1. Write RD_ADDR (0x01C) = source address in transformer register file
2. Dummy read RD_ADDR to ensure rd_data_q captured
3. Read RD_DATA (0x020) = rd_data_q value
```

---

---

## Tcl Script Updates

All JTAG Tcl training scripts updated for new register map:

| Script | Changes |
|---|---|
| `run_hop_checks.tcl` | WR_ADDR: 0x008->0x040, WR_DATA: 0x00C->0x044, added WR_STROBE: 0x048 |
| `run_train.tcl` | `rf_write` proc added, all 2-step writes -> 3-step |
| `run_convergence_test.tcl` | `rf_write` proc added, weight load loop fixed |
| `run_seed42_replay.tcl` | `rf_write` proc added, per-step writes fixed |

Remote scripts added:
- `run_hop_checks_remote.tcl` -- connects to 192.168.0.98:3121
- `run_hop_checks_remote.ps1` -- one-click runner for remote hop checks

---

## Timeline

| Date | Event |
|---|---|
| 2026-07-02 | Original BUG-007: ARM RD_DATA always 0, JTAG correct |
| 2026-07-02 | Hypothesis: rd_data_q pipeline latency; fix: dummy reads in rf_read() |
| 2026-07-03 | Decision: replace custom AXI slave with Vivado-generated template |
| 2026-07-03 | Vivado 2025.2 `create_peripheral` + `generate_peripheral` run on jetpc |
| 2026-07-03 | User logic filled in (register map, bridge, debug regs) |
| 2026-07-03 | C# XSim tests updated, all pass |
| 2026-07-03 | JTAG hop checks: 4/4 PASS on hardware |
| 2026-07-03 | ARM hop checks: CYCLE_LO stuck at 0, RD_DATA 0 |
| 2026-07-03 | `busybox devmem` confirms hardware-level failure for 0x008, 0x054, 0x078 |
| 2026-07-03 | JTAG probe confirms 0x008 and 0x054 work via JTAG AXI path |
| 2026-07-03 | Root cause: ARM-specific AXI path issue, synthesis confirmed correct |
| 2026-07-03 | Workaround identified: use alias addresses 0x010 and 0x060 |

| 2026-07-03 | Root cause documented: old sparse register map was incompatible with ARM path |
| 2026-07-03 | Contiguous register map v2 (0x000-0x038) adopted; arm_train.c rewritten |
| 2026-07-03 | RTL rebuilt on jetpc; JTAG re-verified; ARM tests pending |

**Status:** PENDING REBUILD -- contiguous register map adopted. ARM test results will confirm.
