# BUG-007 — ARM RD_DATA always returns 0x00000000; JTAG returns correct value

**Date:** 2026-07-02  
**Status:** OPEN — root cause not yet identified  
**Severity:** BLOCKER — ARM HOP3 + HOP4 fail; ARM training cannot verify register-file path  
**Discovered by:** JTAG hop check passed (HOP3 = 0xDEAD1234 ✓), ARM hop check failed (HOP3 = 0x00000000 ✗)

---

## Symptom

```
SANITY HOP3: rf scratch write 0xDEAD1234 readback=0x00000000  [FAIL]
SANITY HOP4: transformer VERSION readback=0x00000000           [FAIL]
DEBUG: rd_path probe rf[0x200]=0x00000000  (expect 0xDEADBEEF if path alive)
```

JTAG reads the same registers in the same bitstream and gets the correct values.
ARM reads always return 0x00000000.

---

## What is confirmed working (ARM)

- HOP1: CYCLE_LO (0x010) increments — ARM→FPGA AXI bus alive ✓
- HOP2: CTRL (0x000) write+readback — axi_lite_slave registers writable/readable ✓
- RD_ADDR echo (0x050): writing 0x55 reads back 0x55 — `reg_rd_addr` IS being updated ✓

## What fails (ARM only)

- RD_DATA (0x054) always returns 0x00000000 for ALL rf_read calls
- This includes unmapped address 0x200 (should return 0xDEADBEEF), scratch 0x1FF, VERSION 0x1FE
- 100 µs `usleep` between write-RD_ADDR and read-RD_DATA makes NO difference

---

## Hypotheses tried and ruled out

### ❌ Timing/settling (DISPROVED)

**Theory:** the combinational `rd_data` path through `transformer_train` (with `keep_hierarchy`)
takes longer than 10 ns to settle after `reg_rd_addr` changes, causing ARM to read stale 0.

**Disproof:** 100 µs delay between write-RD_ADDR and read-RD_DATA still returns 0x00000000.
At 100 MHz, 100 µs = 10,000 clock cycles — any combinational path settles in <1 cycle.
Timing is definitively NOT the cause.

### ❌ rd_data_q latency (DISPROVED for current bitstream)

**Theory:** `rd_data_q` pipeline register in `axi_lite_slave` needs dummy reads to flush.

**Disproof:** The current bitstream (gen00-2026-07-02) was built BEFORE `rd_data_q` was added.
The current bitstream uses direct `s_axi_rdata <= tt_rd_data` (no pipeline). Flush reads
are irrelevant for this build.

### ❌ Stale bitstream

Not applicable — JTAG proves the bitstream has scratch_reg and VERSION working.

---

## What we know must be true

Since JTAG and ARM both route through the SAME SmartConnect → SAME `axi_lite_slave` →
SAME `transformer_train`, and since:
- JTAG reads `s_axi_rdata` case `10'h054: s_axi_rdata <= tt_rd_data` and gets correct values
- ARM reads the same address and case and gets 0

The ARM and JTAG transactions must differ in a way that causes the case statement to
NOT fire for ARM, OR `tt_rd_data` evaluates to 0 at the exact moment ARM's read lands.

---

## Suspected remaining hypotheses

1. **SmartConnect address presentation:** SmartConnect presents different `s_axi_araddr` to
   the slave for ARM vs JTAG transactions, causing a different case branch to fire.
   (Though `araddr[9:0]` analysis shows both should give 0x054.)

2. **RTL synthesis artifact:** `tt_rd_data` in the synthesized netlist is driven by logic
   that evaluates differently based on some state that JTAG sets but ARM does not.

3. **AXI protocol difference:** ARM issues read transactions in a way that hits a protocol
   edge case in `axi_lite_slave` (e.g. `s_axi_rvalid` handling, read-before-write ordering).

---

## Fix path

The `rd_data_q` change (new bitstream) breaks the `rd_data` combinational path into a
registered stage. This changes the synthesis path fundamentally and may resolve the issue
regardless of the exact root cause. Root cause remains unconfirmed until the new bitstream
is tested.

**ILA probe (if new bitstream also fails):** add Vivado ILA on `s_axi_araddr`, `tt_rd_addr`,
`tt_rd_data`, `s_axi_rdata`, `s_axi_arvalid` to capture the actual values on board.

---

## Timeline

| Date | Event |
|------|-------|
| 2026-07-01 | HOP3/HOP4 added to `transformer_train.sv` (scratch_reg, GEN00_VERSION) |
| 2026-07-02 | gen00-2026-07-02 bitstream built; JTAG HOP3/HOP4 PASS |
| 2026-07-02 | ARM HOP3/HOP4 fail with 0x00000000 |
| 2026-07-02 | 100 µs delay tested — no change, timing theory disproved |
| 2026-07-02 | Root cause unknown; `rd_data_q` new bitstream is next test |

**Date:** 2026-07-02  
**Status:** OPEN  
**Severity:** BLOCKER — ARM HOP3 + HOP4 fail; ARM training cannot verify register-file path  
**Discovered by:** JTAG hop check passed (HOP3 = 0xDEAD1234 ✓), ARM hop check failed (HOP3 = 0x00000000 ✗)

---

## Symptom

```
SANITY HOP3: rf scratch write 0xDEAD1234 readback=0x00000000  [FAIL]
SANITY HOP4: transformer VERSION readback=0x00000000           [FAIL]
DEBUG: rd_path probe rf[0x200]=0x00000000  (expect 0xDEADBEEF if path alive)
```

JTAG reads the same register in the same bitstream and gets the correct value.  
ARM reads get 0x00000000.

---

## Root cause

The fix for BUG-007's predecessor introduced a pipeline register in `axi_lite_slave.sv`:

```systemverilog
// rd_data_q captures tt_rd_data every cycle
always_ff @(posedge clk) rd_data_q <= tt_rd_data;

// RD_DATA read returns rd_data_q, not raw tt_rd_data
10'h054: s_axi_rdata <= rd_data_q;
```

This adds **1 cycle latency**: after writing `RD_ADDR`, the new `tt_rd_data` value appears
combinationally in the same cycle, but `rd_data_q` doesn't capture it until the NEXT clock edge.

The ARM `rf_read()` function in `arm_train.c`:
```c
reg_write(REG_RD_ADDR, addr);   // cycle T: reg_rd_addr updated
return reg_read(REG_RD_DATA);   // cycle T+1: reads rd_data_q[T] = STALE (pre-update value)
```

AXI-Lite minimum timing:
- Write completes at cycle T (BVALID → BREADY)
- ARM immediately issues read: ARVALID arrives at T+1
- `s_axi_rdata <= rd_data_q` captures `rd_data_q[T]` = value BEFORE the new `reg_rd_addr`

**JTAG is not affected** because JTAG transactions take ~microseconds. By the time JTAG
issues the read after the write, `rd_data_q` has been updated many thousands of cycles earlier.

---

## Fix

In `arm_train.c`, insert dummy read(s) between write-RD_ADDR and read-RD_DATA to guarantee
`rd_data_q` has captured the new value:

```c
static uint32_t rf_read(uint32_t addr) {
    reg_write(REG_RD_ADDR, addr);
    // Flush: read RD_ADDR echo twice to insert ≥2 AXI cycles.
    // rd_data_q needs 1 clock edge to capture new tt_rd_data after reg_rd_addr changes.
    // Two dummy reads guarantee ≥2 cycles regardless of AXI arbitration timing.
    (void)reg_read(REG_RD_ADDR);
    (void)reg_read(REG_RD_ADDR);
    return reg_read(REG_RD_DATA);
}
```

Alternatively, the RTL fix would be to use a two-stage pipeline register or increase latency,
but the C-side flush is simpler and keeps the RTL unchanged.

---

## Verification

After fix, expected ARM hop output:
```
SANITY HOP3: rf scratch write 0xDEAD1234 readback=0xDEAD1234  [PASS]
SANITY HOP4: VERSION=0x67300001 ("g0.001")                    [PASS]
```

---

## Timeline

| Time | Event |
|------|-------|
| 2026-07-01 | HOP3/HOP4 added to `transformer_train.sv` (scratch_reg, GEN00_VERSION) |
| 2026-07-02 | gen00-2026-07-02 bitstream built; ARM HOP3/HOP4 return 0x00000000 |
| 2026-07-02 | JTAG hop check written and run — JTAG HOP3/HOP4 PASS with same bitstream |
| 2026-07-02 | Root cause isolated: `rd_data_q` pipeline latency not accounted for in `rf_read()` |
| 2026-07-02 | Fix identified: two dummy reads of RD_ADDR before reading RD_DATA |
