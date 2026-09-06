# Benchmark guide

TensorLib includes a CPU benchmark harness for tensor kernels, autograd,
neural-network modules, end-to-end training steps, and OpenMP scaling. This
guide explains how to produce reproducible results and how to interpret the
CSV output. It does not publish a permanent leaderboard: results depend on the
CPU, compiler, build flags, thread placement, and system load.

## Quick start

Run the short benchmark profile in a native-tuned release build:

```sh
make benchmark-quick BUILD_DIR=build-bench
```

The command builds `bench_tensorlib`, prints a report, and writes
`benchmark-results.csv`. To choose the output file explicitly:

```sh
make benchmark-quick BUILD_DIR=build-bench \
  BENCHMARK_CSV=results.csv
```

Use a dedicated build directory. Benchmark builds enable
`TENSORLIB_NATIVE_OPTIMIZATIONS`, which adds host-specific compiler flags and
may produce binaries that do not run on another CPU.

## Build and run without Make

The equivalent CMake workflow is:

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTENSORLIB_BUILD_BENCHMARKS=ON \
  -DTENSORLIB_NATIVE_OPTIMIZATIONS=ON
cmake --build build-bench --parallel
python3 scripts/run_benchmarks.py \
  --executable build-bench/bench_tensorlib \
  --profile quick --suite all --csv benchmark-results.csv
```

The Python runner adds the operating system, CPU model, logical processor
count, and compiler to every CSV row. The C executable can also be invoked
directly when host metadata is not needed:

```sh
build-bench/bench_tensorlib --profile quick --suite kernels \
  --csv raw-results.csv
```

Run `build-bench/bench_tensorlib --help` for the accepted command-line options.

## Profiles

| Profile | Samples | Minimum time per sample | Warm-up | Intended use |
|---------|--------:|------------------------:|--------:|--------------|
| `smoke` | 1 | none | none | Correctness and portability checks only |
| `quick` | 5 | 20 ms | 20 ms | Local development and regression triage |
| `full` | 15 | 100 ms | 250 ms | Results intended for comparison or publication |

The harness first times one operation, then batches up to 1,000,000 iterations
per sample to reach the profile's minimum duration. It reports time per
operation. The median is the primary result; p95 is useful for spotting noisy
runs. Do not treat `smoke` timings as performance measurements.

Run the full profile with:

```sh
make benchmark-full BUILD_DIR=build-bench \
  BENCHMARK_CSV=results-full.csv
```

## Suites and measurement boundaries

Select one suite with `--suite kernels|autograd|nn|scaling`, or use `all`.

| Suite | Representative cases | What the timed region includes |
|-------|----------------------|--------------------------------|
| `kernels` | allocation, views, copies, broadcasting, GELU, reductions, gather, matmul, RHS packing | Public API call and output allocation/free |
| `autograd` | composed elementwise forward; forward plus backward | Dynamic graph construction; backward case also includes gradient computation, zeroing, and graph cleanup |
| `nn` | Linear, LayerNorm, attention, decoder block, MNIST MLP, TinyLM | Eager forward graph construction; training rows include loss, backward, optimizer step, and zero-grad |
| `scaling` | large add, GELU, square matmul, TinyLM forward/train | The same workload across the requested OpenMP thread ladder, plus derived speedup data for tensor kernels |

Inputs and model construction happen outside the timed region. Most public
tensor operations allocate their outputs, so output allocation and destruction
remain inside it. Forward-only NN rows are labeled `forward;graph-build`:
TensorLib has no no-grad execution mode, and these rows should not be described
as inference-only latency.

The `kernels`, `autograd`, and `nn` suites currently run their cases with one
requested thread. Use the `scaling` suite to study thread count. Reductions and
gather are serial today and identify that fact in the `layout` column.

## Thread scaling

By default, the runner tests powers of two up to the number of logical
processors, followed by the exact logical-processor count. Set a stable ladder
explicitly when comparing runs:

```sh
make benchmark-scaling BUILD_DIR=build-bench \
  BENCHMARK_THREADS=1,2,4,8 \
  BENCHMARK_CSV=scaling.csv
```

Or run the executable directly:

```sh
build-bench/bench_tensorlib --suite scaling --profile full \
  --threads 1,2,4,8 --csv scaling-raw.csv
```

Speedup is `baseline_time / current_time`, where the baseline is the first
thread count in the supplied ladder. For conventional one-thread-relative
speedup, put `1` first. Parallel efficiency is `speedup / thread_count`.

When TensorLib is built without OpenMP, requests above one thread are written
as `skipped` rows. Configure with `-DTENSORLIB_ENABLE_OPENMP=OFF` only when a
serial portability result is intentional.

## Optional OpenBLAS and PyTorch comparisons

Configure the benchmark build after OpenBLAS is installed. If CMake finds
`cblas.h` and the OpenBLAS library, it creates `bench_openblas`. PyTorch is an
optional runtime dependency of `benchmarks/bench_pytorch.py`; neither reference
is required to build TensorLib.

Run the available references with a matched single-thread setting:

```sh
make benchmark-compare BUILD_DIR=build-bench \
  BENCHMARK_THREADS=1 BENCHMARK_REFERENCE_THREADS=1
```

This produces `benchmark-results.csv`,
`benchmark-results-openblas.csv`, and/or
`benchmark-results-pytorch.csv`, depending on what is installed.

Compare only rows with the same case, shape, profile, and actual thread count.
The OpenBLAS baseline matches the three matmul workloads and includes output
allocation/free. The PyTorch script matches those matmul shapes and the
MNIST/TinyLM-style workloads in eager float32. These are workload references,
not claims that framework internals, graph construction, memory allocators, or
optimizer implementations are identical.

## CSV schema

The native and reference runners share these columns:

| Column | Meaning |
|--------|---------|
| `suite`, `case` | Workload group and stable case name |
| `shape`, `layout` | Symbolic dimensions and relevant memory/execution layout |
| `profile` | `smoke`, `quick`, or `full` |
| `requested_threads`, `actual_threads` | Requested OpenMP/library setting and observed team size |
| `median_seconds`, `p95_seconds` | Per-operation timing statistics |
| `metric`, `value` | Unit and derived throughput, or milliseconds per call |
| `iterations` | Operations batched into each timing sample |
| `checksum` | Consumed result used to keep measured work observable |
| `status` | `ok`, `skipped`, or `derived` |

`scripts/run_benchmarks.py` appends `os`, `cpu`, `logical_processors`, and
`compiler`. Scaling rows with `status=derived` contain speedup in `value` and
parallel efficiency in `checksum`; their raw timing columns are zero. TinyLM
scaling speedup is currently printed to the console but is not emitted as a
separate derived CSV row, so calculate it from its `status=ok` timing rows when
processing CSV data.

Metric formulas are conventional effective rates:

- Matmul uses `2 * M * N * K / median_seconds` FLOP/s, multiplied by batch
  dimensions when present.
- Elementwise bandwidth counts the bytes named by the case (for example, two
  reads plus one write for contiguous binary operations).
- Model throughput divides samples or tokens processed per call by the median.

Use these metrics to compare the same case. Effective GB/s is not a measurement
of hardware memory-controller traffic, and GFLOP/s does not account for every
instruction surrounding the nominal matmul work.

## Reproducible reporting checklist

Before benchmarking, run the correctness suite:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For results that others can reproduce:

- Record the commit, CPU model, operating system, compiler, build type, CMake
  options, profile, suite, and thread ladder.
- Close CPU-heavy applications, connect laptops to power, and use a stable
  performance governor where the operating system permits it.
- Keep affinity and thread placement consistent, especially on hybrid CPUs.
- Run the `full` profile in several fresh processes and report the distribution
  of medians rather than selecting the fastest run.
- Compare `actual_threads`, not only the requested value, and give reference
  libraries the same thread count.
- Retain the raw CSV files with any reported summary.

CI runs only the smoke profile. The project intentionally has no fixed
performance threshold because shared CI hosts do not provide stable timing.

## Adding a benchmark case

Place the case in the suite that owns its measurement boundary:

- `benchmarks/bench_kernels.c` for public tensor primitives.
- `benchmarks/bench_autograd.c` for graph lifecycle workloads.
- `benchmarks/bench_nn.c` for layers or model-level workloads.
- `benchmarks/bench_scaling.c` for explicit thread scaling.

Allocate reusable inputs outside the operation callback, consume at least one
output value through the checksum, release per-call outputs inside the callback,
and use the shared harness rather than adding a new timer. If a reference case
is added, keep its shape and timed allocation boundary aligned and document any
semantic difference. Validate harness changes with `test_bench_harness` and run
the smoke benchmark before collecting timings.

