# FEAT-001 — FPGA XSim Development Pipeline (Xilinx)

## Status
💡 Open — not yet started

## Discovered
11/06/26 — Day 18. Natural next step after LUT exp approximation is validated on CPU.

## Description

Standard bottom-up pipeline for bringing NeuronFabric training onto Xilinx FPGA with XSim simulation at each layer before moving to hardware.

## Development order

### 1. BF16 arithmetic module
- BF16 multiply-accumulate (MAC) unit
- Verify: simulate a handful of known multiply-add pairs, compare to IEEE 754 BF16 reference
- Parameterise: latency (pipeline depth), number of parallel MACs

### 2. Matrix multiplication
- Tiled MatMul built from BF16 MACs
- Verify: small matrix products (4×4, 8×8) match C# reference output within BF16 rounding
- Include transpose variants (B^T for attention scores, A^T for weight gradients)

### 3. Attention
- QKV projection → scores → softmax (exp LUT-256, `2^n·2^f`) → weighted sum
- Verify: forward pass logits match `AttentionCore` C# reference
- Key: exp LUT table fits in BRAM; test LUT read latency vs pipelined MAC

### 4. MLP (feed-forward block)
- Two linear layers + GeLU activation
- Verify: forward output matches `AttentionLayer` FF block
- GeLU approximation may need its own LUT (or polynomial); decide at this stage

### 5. LayerNorm
- Mean and variance reduction over embed dim
- Reciprocal sqrt via Newton-Raphson or LUT
- Verify: normalised output matches C# `LayerNorm` within tolerance

### 6. Adam optimiser
- Moment accumulators (m=FP32, v=FP32) updated each step
- BF16 weight store with FP32 master (w=BF16, master=FP32)
- Verify: weight update trajectory matches C# `AdamBF16WeightsAttentionCore` for 100 steps

### 7. Full Transformer (integration)
- Stack N layers; wire residual connections; add embedding + unembedding
- **Do NOT attempt Shakespeare 334K in XSim** — cycle-accurate simulation of 334K params × 1,000 steps would take days
- Verify strategy: generate golden test vectors from C# using a **tiny synthetic model** (e.g. layers=2, embed=8, heads=2, ff=24, vocab=16, seqLen=4) — small enough that XSim finishes in seconds
  - Forward pass: compare RTL logits to C# `CpuAdamBF16WeightsTransformerBus.Forward()` output
  - Backward pass: compare RTL weight deltas to C# after 1 Adam step
  - Loss: compare scalar cross-entropy to C# `TrainStep()` return value
- Once RTL passes on the synthetic model, full Shakespeare training runs on **actual FPGA hardware** (not XSim)
- Test vector generation: add a `Neuro.Attention.Tests` fixture that serialises tiny-model I/O to binary files for consumption by the XSim testbench

## Cross-repo test flow

The two repos stay fully independent — no code dependency, just file conventions.

A new project **`Neuro.Attention.XSim.LocalTests`** (separate from `Neuro.Attention.Tests`) owns the FPGA test loop. It is never run in CI — local developer machine only.

```
neuro-fabric (C#)
  Neuro.Attention.XSim.LocalTests
    FpgaVecGen
      ↓ writes  run/fpga-testvecs/<module>/input.hex
                run/fpga-testvecs/<module>/expected.hex
    FpgaXSimRunner
      ↓ Process.Start("xvlog", "bf16_mac.sv tb_bf16_mac.sv")
      ↓ Process.Start("xelab", "tb_bf16_mac")
      ↓ Process.Start("xsim",  "tb_bf16_mac -runall")
      ↓ writes  run/fpga-testvecs/<module>/pass_fail.txt
    FpgaVecCheck
      ↓ reads   pass_fail.txt
      ↓ fails / passes as a normal xUnit test
```

- **Developer workflow is pure C#** — just `dotnet test Neuro.Attention.XSim.LocalTests`; the `xvlog`/`xelab`/`xsim` toolchain is invoked automatically behind the scenes, no manual shell commands needed
- **CI exclusion:** `Neuro.Attention.XSim.LocalTests` is not listed in `.github/workflows/build.yml` — the yaml runs only `Neuro.Attention.Tests` and `Neuro.Core.Tests`; XSim tests never execute in GitHub Actions
- Verilog source paths passed to `xvlog` resolve via env var `NEURO_FPGA_SRC` pointing to `neuro-fabric-fpga/rtl/` — the only coupling between the two repos
- `NEURO_TESTVECS` env var points to `neuro-fabric/run/fpga-testvecs/` for the hex file exchange folder

## Notes

- XSim simulates RTL exactly — treat it as the ground-truth hardware reference before any synthesis
- Each stage should have a self-contained testbench with pass/fail assertion on numeric output
- C# project already provides golden reference values at every level; generate test vectors from `Neuro.Attention.Tests`
- BF16 and exp LUT are already validated on CPU — reuse the same LUT table content for FPGA BRAM init
