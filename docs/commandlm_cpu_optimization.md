# CommandLM CPU training optimization

This report records the unpinned measurements used to optimize TensorLib's
approximately 30-million-parameter CPU training path. Raw CSV and `perf` files
were kept under `/tmp` and are intentionally not part of the repository.

## Workload and host

- Model: 1,024-token vocabulary, context 256, width 512, 8 attention heads,
  9 decoder layers, 29,553,152 trainable parameters, FP32 eager autograd and
  FP32 AdamW.
- Workloads: batch 1 (CLI comparison) and batch 16 (intended training batch).
- Host: Intel Core i5-13400F, 6 hyper-threaded P-cores plus 4 E-cores, 16
  logical CPUs, AVX2, 15 GiB RAM.
- Software: Linux 7.2.6, GCC 16.2.1, OpenMP, `-O3`, native CPU tuning.
- Baseline: `2260447`; optimized result: `a79edd4`.
- Protocol: three fresh processes per configuration, full profile (15 timed
  samples after warm-up), requested threads 1/2/4/8/16, no CPU affinity.
  Tables report the median of the three process medians. CV is the coefficient
  of variation across those three medians.

## End-to-end results

Batch 1:

| Threads | Baseline ms | Optimized ms | Change | Optimized tokens/s | CV |
|---:|---:|---:|---:|---:|---:|
| 1 | 1036.842 | 978.808 | -5.6% | 261.5 | 2.38% |
| 2 | 628.703 | 580.076 | -7.7% | 441.3 | 0.40% |
| 4 | 466.644 | 349.552 | -25.1% | 732.4 | 0.39% |
| 8 | 426.622 | 315.178 | -26.1% | 812.2 | 0.25% |
| 16 | 389.382 | 275.895 | **-29.1%** | **927.9** | 0.78% |

Batch 16:

| Threads | Baseline ms | Optimized ms | Change | Optimized tokens/s | CV |
|---:|---:|---:|---:|---:|---:|
| 1 | 11775.290 | 11157.023 | -5.3% | 367.1 | 1.76% |
| 2 | 6807.583 | 6052.903 | -11.1% | 676.7 | 0.28% |
| 4 | 4324.345 | 3489.083 | -19.3% | 1173.9 | 0.27% |
| 8 | 3785.364 | 3085.628 | -18.5% | 1327.4 | 0.04% |
| 16 | 3306.740 | 2631.164 | **-20.4%** | **1556.7** | 0.71% |

At 16 threads the batch-1 p95 fell from 403.453 to 283.918 ms and the
batch-16 p95 fell from 3383.292 to 2733.564 ms. Batch-16 scaling improved from
3.56x (22.3% efficiency) to 4.24x (26.5% efficiency). Every result row was
`status=ok`, and requested and actual thread counts matched.

The three-run TinyLM regression medians were 140.504, 92.832, 66.992, 64.688,
and 60.907 ms at 1/2/4/8/16 threads. Those are 9.8-19.5% faster than the
supplied baseline, so no thread count regressed. Cross-run CV was 12.25% at
one thread and 13.64% at four threads because the first process was an outlier;
the host used the `powersave` governor and had a background load average above
2.7. The other CVs were 1.26%, 2.41%, and 6.17%. All samples were retained;
the lower-variance 30M results above remain the primary acceptance measurement.

## Attribution at 16 threads

| Batch | Version | Forward ms | Backward ms | AdamW ms |
|---:|:---|---:|---:|---:|
| 1 | baseline | 147.008 | 166.124 | 74.512 |
| 1 | optimized | 118.881 | 121.088 | 36.048 |
| 16 | baseline | 1195.865 | 2002.969 | 74.474 |
| 16 | optimized | 1069.775 | 1488.466 | 36.154 |

For the profiled batch-16 backward call, matmul fell from 786.347 to
608.434 ms, reshape from 51.603 to 23.905 ms, slice from 42.803 to 23.749 ms,
gradient accumulation from 35.372 to 8.478 ms, and persistent merge from
25.375 to 13.168 ms. These instrumented operation calls are attribution data;
the phase and end-to-end timers remain the performance result.

Allocations per step fell from 681 to 645 and allocated bytes from 7.615 GB to
7.133 GB. Peak live tensor storage was essentially flat at 6.34 GB for batch
16 (590 MB for batch 1), so the changes do not trade speed for an unbounded
cache. Context teardown returned tracked live bytes to zero.

## Hardware-counter interpretation

`perf stat` was collected at every thread count. At 16 threads it measured
219.34 task-clock seconds over 25.12 elapsed seconds. P-core IPC was 1.624 and
E-core IPC was 1.579, with zero context switches and migrations. The combined
cache-miss ratios were high enough (32.6% P-core, 38.8% E-core) to confirm
material cache/memory pressure alongside compute work.

Frame-pointer `perf record` captures at 1, 4, and 16 threads lost zero samples.
At 16 threads the packed matmul kernel accounted for 60.2% of sampled P-core
cycles and 64.6% of sampled E-core cycles; RHS packing accounted for 7.5% and
6.0%, respectively. These are sampled cycle shares, not wall-clock shares.
They reconcile with matmul remaining the largest timed operation after the
backward, clone, accumulation, and optimizer improvements.

## Changes and tradeoffs

The optimized path parallelizes large autograd fills, RHS packing, million-
element tensor work, transposed clones, AdamW norm/state passes, and fuses
sibling slice gradients. It also avoids avoidable storage-sized leaf-gradient
copies and accelerates contiguous row clones. Small-work thresholds retain the
serial path and existing odd, sliced, transposed, broadcast, and zero-sized
tensor tests cover the important layout boundaries.

OpenBLAS was evaluated but not adopted globally: it was only 4-7% faster on
two large dense shapes while TensorLib's custom path was about 9x faster on the
attention shape. FP16/BF16 and reduced-precision optimizer states were not
enabled: this AVX2 CPU lacks native BF16/AMX acceleration, and changing training
or checkpoint numerics would require separate convergence validation.

The remaining limit is packed matmul plus its bandwidth-intensive packing and
clone traffic. A future backend should select per shape rather than replace the
portable kernel wholesale. Batch 16 gives the best throughput when about
6.34 GB of transient tensor storage is acceptable; batch 1 is the practical
low-memory CLI choice.

The Release suite passed all 39 tests. A current-source ASan+UBSan build passed
all 36 applicable tests (`detect_leaks=0` because LeakSanitizer cannot operate
under the tracing environment). A real one-step batch-1 CLI run loaded the
existing checkpoint, trained, validated, saved to `/tmp`, and resumed that new
checkpoint for another step; the two steps took 0.361 and 0.355 seconds and
produced finite losses. No public API, CLI option, or checkpoint schema changed.

## Reproduction

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTENSORLIB_BUILD_BENCHMARKS=ON \
  -DTENSORLIB_NATIVE_OPTIMIZATIONS=ON \
  -DTENSORLIB_ENABLE_OPENMP=ON
cmake --build build-bench --parallel
ctest --test-dir build-bench --output-on-failure

python3 scripts/run_benchmarks.py \
  --executable build-bench/bench_tensorlib --profile full \
  --suite command --threads 1,2,4,8,16 \
  --csv /tmp/tensorlib-command-run1.csv
```

Repeat the last command in two fresh processes with unique CSV names. Use
`--suite nn` for the TinyLM regression suite. For a profiling build, add
`-DCMAKE_C_FLAGS_RELEASE="-O3 -g -fno-omit-frame-pointer -DNDEBUG"`; run
`perf stat` with separate `cpu_core` and `cpu_atom` events and use
`perf record -g` at 1, 4, and 16 threads. Keep all raw captures outside the
source tree.
