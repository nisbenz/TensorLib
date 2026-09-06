# Performance notes

For commands, suite definitions, timing methodology, CSV fields, comparison
boundaries, and contribution guidance, see the dedicated
[Benchmark guide](benchmarks.md).

TensorLib's benchmark suite measures public tensor operations, autograd, neural
network modules, end-to-end training work, and CPU thread scaling. Matmul is
still covered, but it is one part of the report rather than a proxy for the
whole library.

## Build and run

Use a separate native-tuned release build:

```sh
make benchmark-quick BUILD_DIR=build-bench
```

This writes `benchmark-results.csv`. Other useful targets are:

```sh
make benchmark-full BUILD_DIR=build-bench
make benchmark-kernels BUILD_DIR=build-bench
make benchmark-scaling BUILD_DIR=build-bench
make benchmark-compare BUILD_DIR=build-bench
```

Set an explicit scaling ladder or output path when needed:

```sh
make benchmark-scaling BUILD_DIR=build-bench \
  BENCHMARK_THREADS=1,2,4,8 BENCHMARK_CSV=results-i5.csv
```

The underlying executable also supports direct use:

```sh
build-bench/bench_tensorlib --profile quick --suite all \
  --threads 1,2,4,8 --csv raw-results.csv
```

The quick profile uses five adaptively batched samples of at least 20 ms. The
full profile uses fifteen samples of at least 100 ms after a longer warm-up.
`--smoke` exists for correctness and portability checks; its timings are not
performance results. See the [profile table](benchmarks.md#profiles) for exact
warm-up durations and intended uses.

## What is measured

- **Kernels:** allocation, views and copies, contiguous and broadcast
  elementwise work, GELU, reductions, gather, and realistic matmul shapes.
- **Autograd:** graph construction and a composed forward/backward lifecycle.
- **Neural networks:** Linear, LayerNorm, causal attention, a decoder block,
  MNIST MLP forward/training, and TinyLM forward/training.
- **Scaling:** large add, GELU, matmul, and TinyLM workloads across the requested
  OpenMP thread ladder, with speedup and parallel efficiency.

Forward-only neural measurements include dynamic graph construction. TensorLib
does not currently expose a no-grad inference mode, so calling these results
"inference-only" would be misleading. Output allocation is likewise included
because most public tensor APIs allocate their result.

Reductions and gather are currently serial. Their kernel rows explicitly say
`serial`; they are not repeated at different thread counts. Streaming kernels
report effective GB/s, matmuls report GFLOP/s, and model workloads report
samples/s or tokens/s.

## Optional comparisons

If CMake finds OpenBLAS, it builds `bench_openblas` with the same square, QKV,
and batched-attention matmul shapes. The comparison uses wall-clock time and
the same allocation boundary as the TensorLib public API.

If PyTorch is installed, `make benchmark-compare` also runs the eager float32
CPU reference in `benchmarks/bench_pytorch.py`. It matches the MLP and
TinyLM-style forward/training workloads without making PyTorch a TensorLib
build dependency. Reference results use `BENCHMARK_REFERENCE_THREADS`, which
defaults to one. Set `BENCHMARK_THREADS=1` as well when collecting a deliberately
matched single-thread comparison.

## Reading results

Use the median as the primary number and p95 as a noise indicator. Scaling
speedup is `one_thread_time / N_thread_time`; parallel efficiency is that
speedup divided by `N`. Memory-bound operations often stop scaling before
matmul does, while hybrid CPUs can show non-uniform steps as efficiency and
performance cores enter the run.

Do not compare results unless the CPU, compiler, build options, shapes, profile,
and thread count match. The standard-library runner records the OS, CPU model,
logical processor count, and compiler in every CSV row.

For publishable numbers:

- Close other CPU-heavy programs and connect laptops to power.
- Use a fixed performance governor where the operating system permits it.
- Pin processes or cores consistently when comparing hybrid CPUs.
- Run the full profile in several fresh processes and compare medians.
- Set both TensorLib and reference libraries to the same thread count.

Benchmark CI only executes `--smoke`; fixed performance thresholds are avoided
until stable results from comparable dedicated hosts are available.
