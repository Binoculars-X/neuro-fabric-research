# FPGA Neural Network — Developer Context

Supplementary context not covered in ARCHITECTURE.md or DEMO.md.  
Contains: C# implementation code, FPGA?ASIC roadmap, and GPU comparison details.

---

## Project Status

- [x] Architecture designed (ARCHITECTURE.md)
- [x] Demo plan designed (DEMO.md)
- [ ] C# project scaffolded (no .csproj or .cs files exist yet)
- [ ] FPGA HDL conversion
- [ ] ASIC tape-out (future)

---

## C# Implementation

### Project Structure

```
\neuro-fabric\
  ARCHITECTURE.md
  DEMO.md
  DEV-CONTEXT.md
  src/
    Neuro.Core/
      NeuronCore.cs
      NeuronLayer.cs
      NeuralBus.cs
      Activations.cs
      LossFunctions.cs
    Neuro.Benchmarks/
      FpgaVsGpuBenchmark.cs
      BenchmarkResult.cs
    Neuro.Datasets/
      IrisLoader.cs
      MnistLoader.cs
      HarLoader.cs
  models/
    trained_iris.bin
    trained_mnist.bin
```

---

### `NeuronSignal` Enum

```csharp
public enum NeuronSignal
{
    Forward,      // 00 — forward inference pass
    Backward,     // 01 — backpropagation pass
    WeightRead,   // 10 — export weights from all cores
    WeightWrite   // 11 — import weights into all cores
}
```

FPGA: 2-bit control wire broadcast to all cores simultaneously.

---

### `NeuronCore`

```csharp
public class NeuronCore
{
    private readonly float[] _weights;
    private float _bias;
    private float _lastZ;      // pre-activation, cached for backprop
    private float _lastOutput; // post-activation, cached for backprop

    public int InputCount => _weights.Length;

    public NeuronCore(int inputCount)
    {
        var rng = new Random();
        _weights = Enumerable.Range(0, inputCount)
            .Select(_ => (float)(rng.NextDouble() * 2 - 1))
            .ToArray();
        _bias = 0f;
    }

    public float Forward(float[] inputs)
    {
        _lastZ = inputs.Zip(_weights, (i, w) => i * w).Sum() + _bias;
        _lastOutput = Activations.ReLU(_lastZ);
        return _lastOutput;
    }

    public float[] Backward(float gradientIn, float learningRate)
    {
        float delta = gradientIn * Activations.ReLUDerivative(_lastZ);
        float[] gradientOut = _weights.Select(w => w * delta).ToArray();
        for (int i = 0; i < _weights.Length; i++)
            _weights[i] -= learningRate * delta;   // BRAM write on FPGA
        _bias -= learningRate * delta;
        return gradientOut;
    }

    // WeightRead — streams weights onto bus
    public float[] ReadWeights() => (float[])_weights.Clone();

    // WeightWrite — loads weight slice from bus
    public void WriteWeights(float[] weights)
    {
        if (weights.Length != _weights.Length)
            throw new ArgumentException("Weight count mismatch");
        weights.CopyTo(_weights, 0);
    }
}
```

---

### `NeuronLayer`

```csharp
public class NeuronLayer
{
    private readonly NeuronCore[] _cores;

    public int WeightCount => _cores.Sum(c => c.InputCount);

    public NeuronLayer(int neuronCount, int inputsPerNeuron)
    {
        _cores = Enumerable.Range(0, neuronCount)
            .Select(_ => new NeuronCore(inputsPerNeuron))
            .ToArray();
    }

    public float[] Dispatch(float[] inputs, NeuronSignal signal,
                            float[]? gradients = null, float learningRate = 0.01f)
    {
        return signal switch
        {
            NeuronSignal.Forward  => Forward(inputs),
            NeuronSignal.Backward => Backward(gradients!, learningRate),
            _ => throw new InvalidOperationException($"Signal {signal} handled at bus level")
        };
    }

    // All cores fire simultaneously — models FPGA parallel execution
    private float[] Forward(float[] inputs)
    {
        var outputs = new float[_cores.Length];
        Parallel.For(0, _cores.Length, i =>
            outputs[i] = _cores[i].Forward(inputs));
        return outputs;
    }

    private float[] Backward(float[] gradients, float learningRate)
    {
        var partials = new float[_cores.Length][];
        Parallel.For(0, _cores.Length, i =>
            partials[i] = _cores[i].Backward(gradients[i], learningRate));

        // Accumulate gradients — adder tree on FPGA
        var gradientAccum = new float[_cores[0].InputCount];
        for (int i = 0; i < _cores.Length; i++)
            for (int j = 0; j < gradientAccum.Length; j++)
                gradientAccum[j] += partials[i][j];

        return gradientAccum;
    }

    // WeightRead: flat dump [core0_w0, core0_w1, ..., coreN_wM]
    public float[] DumpWeights() =>
        _cores.SelectMany(c => c.ReadWeights()).ToArray();

    // WeightWrite: distribute flat array back to cores
    public void LoadWeights(float[] flat)
    {
        int stride = flat.Length / _cores.Length;
        Parallel.For(0, _cores.Length, i =>
            _cores[i].WriteWeights(flat.Skip(i * stride).Take(stride).ToArray()));
    }
}
```

---

### `NeuralBus`

```csharp
public class NeuralBus
{
    private readonly NeuronLayer[] _layers;

    public NeuralBus(NeuronLayer[] layers) => _layers = layers;

    public float[] RunForward(float[] input)
    {
        var signal = input;
        foreach (var layer in _layers)
            signal = layer.Dispatch(signal, NeuronSignal.Forward);
        return signal;
    }

    public void RunBackward(float[] lossGradient, float learningRate = 0.01f)
    {
        var grad = lossGradient;
        foreach (var layer in _layers.Reverse())
            grad = layer.Dispatch(null!, NeuronSignal.Backward, grad, learningRate);
    }

    // WeightRead — collect all weights from all layers
    public byte[] ExportModel()
    {
        var all = _layers.SelectMany(l => l.DumpWeights()).ToArray();
        var bytes = new byte[all.Length * sizeof(float)];
        Buffer.BlockCopy(all, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    // WeightWrite — distribute weights back to all layers
    public void ImportModel(byte[] data)
    {
        var all = new float[data.Length / sizeof(float)];
        Buffer.BlockCopy(data, 0, all, 0, data.Length);
        int offset = 0;
        foreach (var layer in _layers)
        {
            int count = layer.WeightCount;
            layer.LoadWeights(all.Skip(offset).Take(count).ToArray());
            offset += count;
        }
    }

    public void SaveModel(string path) => File.WriteAllBytes(path, ExportModel());
    public void LoadModel(string path) => ImportModel(File.ReadAllBytes(path));
}
```

---

### `Activations`

```csharp
public static class Activations
{
    public static float ReLU(float x) => MathF.Max(0, x);
    public static float ReLUDerivative(float x) => x > 0 ? 1f : 0f;

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    public static float SigmoidDerivative(float x) { var s = Sigmoid(x); return s * (1 - s); }

    public static float[] Softmax(float[] x)
    {
        float max = x.Max();
        var exp = x.Select(v => MathF.Exp(v - max)).ToArray();
        float sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }
}
```

---

### `FpgaVsGpuBenchmark`

```csharp
public class FpgaVsGpuBenchmark
{
    private const int Runs = 100_000;

    // Simulates FPGA: Parallel.For models all cores firing per clock cycle
    public static BenchmarkResult RunFpgaSimulation(NeuralBus bus, float[] input)
    {
        // Warmup
        for (int i = 0; i < 1000; i++) bus.RunForward(input);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Runs; i++) bus.RunForward(input);
        sw.Stop();

        return new BenchmarkResult
        {
            Platform = "FPGA (simulated, parallel cores)",
            TotalMs = sw.ElapsedMilliseconds,
            LatencyUs = sw.Elapsed.TotalMicroseconds / Runs,
            InferencesPerSec = (long)(Runs / sw.Elapsed.TotalSeconds),
            EstimatedWatts = 5.0
        };
    }

    // GPU baseline via ONNX Runtime CUDA execution provider
    // Requires: Microsoft.ML.OnnxRuntime.Gpu NuGet package
    // Requires: equivalent model exported to model_7layer.onnx
    public static BenchmarkResult RunGpuBaseline(string onnxModelPath, float[] input)
    {
        using var session = new InferenceSession(
            onnxModelPath,
            SessionOptions.MakeSessionOptionWithCudaProvider());

        var tensor = new DenseTensor<float>(input, new[] { 1, input.Length });
        var inputs = new List<NamedOnnxValue>
            { NamedOnnxValue.CreateFromTensor("input", tensor) };

        // Warmup
        for (int i = 0; i < 1000; i++) session.Run(inputs);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Runs; i++) session.Run(inputs);
        sw.Stop();

        return new BenchmarkResult
        {
            Platform = "GPU RTX 3080 (ONNX/CUDA, single sample)",
            TotalMs = sw.ElapsedMilliseconds,
            LatencyUs = sw.Elapsed.TotalMicroseconds / Runs,
            InferencesPerSec = (long)(Runs / sw.Elapsed.TotalSeconds),
            EstimatedWatts = 155.0
        };
    }

    public static void PrintComparison(BenchmarkResult fpga, BenchmarkResult gpu)
    {
        Console.WriteLine($"\n{"Metric",-25} {"FPGA",-20} {"GPU RTX 3080",-20} {"Winner",-10}");
        Console.WriteLine(new string('-', 75));
        Print("Latency (µs)",    $"{fpga.LatencyUs:F2}",          $"{gpu.LatencyUs:F2}",          fpga.LatencyUs < gpu.LatencyUs);
        Print("Inf/sec",         $"{fpga.InferencesPerSec:N0}",   $"{gpu.InferencesPerSec:N0}",   fpga.InferencesPerSec > gpu.InferencesPerSec);
        Print("Power (W)",       $"{fpga.EstimatedWatts:F0}",     $"{gpu.EstimatedWatts:F0}",     fpga.EstimatedWatts < gpu.EstimatedWatts);
        Print("Inf/Watt",        $"{fpga.InfPerWatt:N0}",         $"{gpu.InfPerWatt:N0}",         fpga.InfPerWatt > gpu.InfPerWatt);
    }

    private static void Print(string metric, string fpgaVal, string gpuVal, bool fpgaWins) =>
        Console.WriteLine($"{metric,-25} {fpgaVal,-20} {gpuVal,-20} {(fpgaWins ? "FPGA ?" : "GPU ?"),-10}");
}

public record BenchmarkResult
{
    public string Platform { get; init; } = "";
    public long TotalMs { get; init; }
    public double LatencyUs { get; init; }
    public long InferencesPerSec { get; init; }
    public double EstimatedWatts { get; init; }
    public double InfPerWatt => InferencesPerSec / EstimatedWatts;
}
```

---

## FPGA ? ASIC Roadmap

```
Stage 1: C# Prototype
  Purpose : validate architecture, backprop, dataset accuracy
  Cost    : dev time only
  Target  : Neuro.Core + benchmarks passing all 4 demo stages

Stage 2: FPGA Implementation
  Device  : Xilinx Artix-7 (small demo) or Kintex-7 325T (full 10K cores)
  Purpose : real power + latency measurements, investor demo
  Cost    : $150–500 dev board
  HDL     : VHDL or SystemVerilog, generated from C# architecture

Stage 3: FPGA Small Production
  Volume  : up to ~10K units/year
  Device  : Kintex-7 or Ultrascale+
  Unit cost: $100–500/unit
  Viable while: model fits in BRAM, throughput requirement <50M inf/sec

Stage 4: ASIC Tape-Out
  Trigger : any of —
    - >10K units/year (cost crossover vs FPGA)
    - >50M weights (BRAM exhaustion)
    - >10K neurons/layer (LUT exhaustion)
    - need >1 GHz clock (FPGA routing limit)
  Node    : 28nm (low cost) or 12nm (high perf)
  NRE cost: $1M–$30M depending on node
  Unit cost: $2–10/unit at volume
  Power gain: 10–100× better than FPGA at same scale
```

---

## GPU Comparison Notes

### RTX 3080 — Why It Cannot Win at Single-Sample Inference

| Overhead | Time | Notes |
|---|---|---|
| PCIe transfer | ~5–10µs | Data must cross CPU?GPU bus |
| CUDA kernel launch | ~5–20µs | Fixed cost regardless of workload size |
| Warp scheduling | ~5–15µs | Waits for full warp (32 threads) to fill |
| Weight fetch (GDDR6) | ~10µs | Weights not on-core, live in GDDR6X |
| **Total minimum** | **~50–200µs** | Even for a single float[] input |

FPGA total: **~35–100ns** (7 clock cycles × 5ns at 200MHz, weights on BRAM).

### RTX 4090 — Even Worse at Low Utilization

The 4090 is designed for 1B+ inf/sec. At 5–50M inf/sec it operates at ~5% utilization but cannot throttle below:
- ~80W minimum (PCIe controller, HBM power gates, cooling fans stay active)
- ~150W at even moderate CUDA activity

Running a 5–50M inf/sec workload on a 4090 wastes 75–95% of its power budget.  
FPGA at the same workload runs near full utilization at 5–15W.

### When GPU Wins

| Scenario | GPU advantage |
|---|---|
| Batch inference >10,000 samples | Massively parallel CUDA cores fully utilized |
| Training large models (>100M params) | HBM bandwidth + tensor cores + mature tooling |
| Layer width >10K neurons | BRAM exhausted on FPGA |
| Rapid model iteration | PyTorch/CUDA ecosystem vs HDL development time |

---

## Solution Structure

```
\neuro-fabric\
  ARCHITECTURE.md
  DEMO.md
  DEV-CONTEXT.md
  src/
    NeuroCore.slnx
    ¦
    +-- Neuro.Core/                        ? class library — all neuron/bus implementation
    ¦     Neuro.Core.csproj
    ¦     NeuronCore.cs
    ¦     NeuronLayer.cs
    ¦     NeuralBus.cs
    ¦     Activations.cs
    ¦     LossFunctions.cs
    ¦
    +-- Neuro.Infrastructure/              ? class library — external integrations
    ¦     Neuro.Infrastructure.csproj
    ¦     Persistence/
    ¦       MongoModelRepository.cs        ? save/load trained models to MongoDB
    ¦       IModelRepository.cs
    ¦     Datasets/
    ¦       IrisLoader.cs
    ¦       MnistLoader.cs
    ¦       HarLoader.cs
    ¦
    +-- Neuro.Demo/                        ? console app — runnable demo + benchmarks
    ¦     Neuro.Demo.csproj
    ¦     Program.cs
    ¦     Benchmarks/
    ¦       FpgaVsGpuBenchmark.cs
    ¦       BenchmarkResult.cs
    ¦     Stages/
    ¦       Stage1_Iris.cs
    ¦       Stage2_Mnist.cs
    ¦       Stage3_Latency.cs
    ¦       Stage4_SensorStream.cs
    ¦
    +-- tests/
          Neuro.Core.Tests/                ? xUnit unit tests — core logic
                Neuro.Core.Tests.csproj
          Neuro.Integration.Tests/         ? xUnit integration tests — dataset + model persistence
                Neuro.Integration.Tests.csproj
```

### Project Names Summary

| Project | Type | Purpose |
|---|---|---|
| `Neuro.Core` | Class library | Neuron, layer, bus, activations, loss |
| `Neuro.Infrastructure` | Class library | MongoDB, dataset loaders, file persistence |
| `Neuro.Demo` | Console app | Runnable demo, benchmarks, 4 stages |
| `Neuro.Core.Tests` | xUnit | Unit tests for core logic |
| `Neuro.Integration.Tests` | xUnit | End-to-end dataset + persistence tests |

### NuGet Dependencies by Project

```
Neuro.Core              — none (pure C#, no dependencies)
Neuro.Infrastructure    — MongoDB.Driver, CsvHelper (dataset parsing)
Neuro.Demo              — BenchmarkDotNet, Microsoft.ML.OnnxRuntime.Gpu
Neuro.Core.Tests        — xUnit, FluentAssertions
Neuro.Integration.Tests — xUnit, Testcontainers.MongoDB
```

---

## Design Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Layer count | 7 | Matches human neocortex depth (6 cortical + subcortical relay) |
| Activation (hidden) | ReLU | Simple derivative, no vanishing gradient in shallow nets |
| Activation (output) | Softmax / Sigmoid | Multi-class / binary respectively |
| Weight format | `float` (C#) ? Q8.8 FPGA | FP32 for prototype accuracy, fixed-point for FPGA efficiency |
| Interconnect | Axon?synapse only | No lateral connections needed for feedforward; saves FPGA routing |
| Bus control | 2-bit signal | Forward / Backward / WeightRead / WeightWrite — maps to 2 wires |
| Parallelism model | `Parallel.For` | Models concurrent FPGA `always` blocks; scales to core count |
| Weight storage format | Raw `float[]` as `byte[]` | Simple, portable, maps directly to BRAM DMA burst |
