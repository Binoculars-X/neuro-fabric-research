# Week 7 Record — 30/06/26
<!-- concise weekly record; daily blocks stay short -->

## Week 7 Start — FPGA Bring-Up To Real Training Harness

Focus shifted from pure RTL/XSim validation to real ZCU102 execution and PC-driven training tests.

The main direction for the week is clear:

```text
FEAT-002 gen00 baseline -> PC-controlled FPGA test wrapper -> real train-step readback -> convergence logging
```

---

## 24/06/26 — ZCU102 Transformer Train Bitstream + Gen00 Test Plan

- ZCU102 hardware path validated end-to-end with Vivado 2025.2 at `C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat`.
- Current FEAT-002 `transformer_train` wrapped as `transformer_train_zcu102` and routed successfully.
- Final bitstream: `fpga/scripts/transformer_train_zcu102_out/transformer_train_zcu102.bit`.
- Timing closed at 125 MHz: WNS `+0.640 ns`, TNS `0.000 ns`, 0 failing setup/hold endpoints.
- Routed utilization: LUT `78.64%`, CLB sites `98.77%`, DSP `25.24%`, RAMB18 `3.18%`, I/O `3.05%`.
- Programmed board with FEAT-002 wrapper bitstream; observed `LED0` heartbeat blinking and `LED2` start-issued high.
- Hardware result proves PL bitstream, SI570 clock, XDC pins, reset/start wrapper, and LED mapping work.
- Forward did not complete because current wrapper ties write bus off; no samples/weights/tokens are loaded yet.
- Created FEAT-003 v2 training-lane architecture plan for 105K model: 4 shared `48x12` matmul lanes, softmax row lanes, tiled MLP, PL-local memory-backed state.
- Added zero-padded RTL generation plan: `rtl/common`, `rtl/gen00`, `rtl/gen01`, `rtl/gen02`.
- Created concise `gen00` FPGA testing plan: PC sends vectors/samples, FPGA runs forward/train, PC reads real results/signatures/status.
- Decided LED-only and ROM-only tests are insufficient; real validation requires host write/read path and convergence logs.
- Confirmed Vivado SysMon telemetry is script-readable: PL/PS temperature plus rails including `VCCINT`, `VCCAUX`, `VCCBRAM`, `VCC_PSAUX`, `VCC_PSINTLP`.

---

## Immediate Next Work

1. Refactor RTL folders into `common/` and `gen00/` without changing behavior.
2. Update XSim and Vivado scripts so all existing gen00 tests still pass after the move.
3. Define the PC host interface for gen00: control/status/write transactions/result readback.
4. Build `tb_` tests using the real `transformer_train`, not a fake DUT.
5. Export XSim seed vectors as PC-sendable `wr_addr/wr_data` transactions.
6. Build one host-test FPGA bitstream only after wrapper simulation is green.
7. Run long PC-driven training cycles while logging results and SysMon telemetry.
