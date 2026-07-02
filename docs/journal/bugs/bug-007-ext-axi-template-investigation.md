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

- **Synthesis corruption** -- JTAG reads ALL addresses correctly, proving every case
  in the `axi_araddr[9:2]` decode mux is synthesized correctly.
- **Non-ASCII corruption** -- em-dashes were found in `//` comments in the .v file
  and removed. However they were inside `//` comment content (stripped before parsing)
  and JTAG proved synthesis was correct before the fix.
- **C code bug** -- `busybox devmem` direct hardware reads reproduce the same failures,
  eliminating arm_train.c from the equation.
- **mmap size** -- MAP_SIZE=0x1000, all failing addresses < 0x100. Not the issue.
- **Cache** -- 0x008 and 0x010 are in the SAME 64-byte cache line; one fails, one passes.
  Cache cannot explain this.

### What is confirmed

- The failing addresses from ARM path always return 0x00000000.
- The same addresses from JTAG path return correct values.
- Both paths go through the SAME SmartConnect and SAME `axi_train_regs` instance.
- The issue is in the ARM PS HPM0_FPD -> SmartConnect -> slave AXI path.

### Current hypothesis (unconfirmed)

The ARM PS uses AXI4 (with ARLEN, ARSIZE, ARBURST signals). The SmartConnect converts
AXI4 -> AXI4-Lite. For specific address values, the SmartConnect may be presenting
a different address to the slave than expected, causing the wrong case to fire in the
read mux. An ILA (Integrated Logic Analyzer) on the slave's S_AXI_ARADDR/S_AXI_ARVALID
signals would definitively show what address the slave actually receives.

---

## Workaround (no rebuild required)

Since the ARM path fails for `0x008` and `0x054` but the identical signals are
available at alias addresses `0x010` and `0x060`:

**In `arm_train.c`:**
```c
#define REG_CYCLE_LO    0x010   /* use alias -- 0x008 returns 0 from ARM */
#define REG_CYCLE_HI    0x014   /* use alias -- 0x00C returns 0 from ARM */
#define REG_RD_DATA     0x060   /* use DBG_RD_DATA_Q -- 0x054 returns 0 from ARM */
```

This workaround is safe because:
- `0x010` and `0x008` both return `cycle_cnt[31:0]` (case 8'h04 and 8'h02)
- `0x014` and `0x00C` both return `cycle_cnt[47:32]` (case 8'h05 and 8'h03)
- `0x060` and `0x054` both return `rd_data_q` (case 8'h18 and 8'h15)

All three alias registers are confirmed working from ARM via `busybox devmem`.

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

**Status:** OPEN — root cause requires ILA investigation. Workaround available.
