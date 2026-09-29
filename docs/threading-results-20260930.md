# Threading profile validation, 2026-09-30

The ENV implementation was validated on an Intel Core i5-13400F (6 P cores
with SMT, 4 E cores, 16 logical CPUs), Linux, GCC 16.2.1, native Release
optimization, and OpenMP. Benchmarked code: `6842b1b`. Untuned defaults remain
unchanged; this comparison evaluates optional process settings.

Three fresh processes were measured for each profile and suite. The table
uses the median of the three process medians. `scaling` used the full profile
(15 samples, 100 ms minimum/sample, 250 ms warm-up); `policy` used quick
(5 samples, 20 ms minimum/sample, 20 ms warm-up). Medium-operation results are
exploratory; validate a selected profile with full runs on your workload.

## Profiles

All three used `OMP_DYNAMIC=FALSE`. Only the settings listed below were set;
other `TENSORLIB_*` variables were unset.

| Profile | Global ceiling | Operation settings | Placement |
|---------|---------------:|--------------------|-----------|
| default16 | 16 | existing defaults | runtime default, unbound |
| p12 | 12 | existing defaults | first 12 explicit places, P cores and SMT |
| mixed | 16 | settings below | all 16 explicit places |

Mixed settings:

```sh
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE
export TENSORLIB_MEMORY_THREADS=6 TENSORLIB_MEMORY_MIN_ELEMENTS=262144
export TENSORLIB_COMPUTE_THREADS=12 TENSORLIB_MATMUL_THREADS=12
export TENSORLIB_PACKED_MATMUL_THREADS=16
export TENSORLIB_GELU_THREADS=6 TENSORLIB_GELU_MIN_ELEMENTS=4096
```

The p12 and mixed profiles also used this machine's verified numbering:

```sh
export OMP_PLACES='{0},{2},{4},{6},{8},{10},{1},{3},{5},{7},{9},{11},{12},{13},{14},{15}'
export OMP_PROC_BIND=close
```

Thread budgets affect existing paths only. The MEMORY threshold also raises
clone's eligibility threshold, so clone at 65,536 elements stays serial in
mixed. Packed multiplication retains its separate budget and threshold.

## Selected results

All times below are milliseconds per call, including output allocation/free.
TinyLM training includes graph construction, backward, AdamW, and zero-grad;
the model uses batch 2, context 128, channels 192, and 4 layers.

| Workload | default16 | p12 | mixed | default16 / mixed |
|----------|----------:|----:|------:|------------------:|
| Add, 262,144 elements (quick) | 0.05375 | 0.05341 | 0.00934 | 5.75× |
| GELU, 65,536 elements (quick) | 0.27979 | 0.27858 | 0.05296 | 5.28× |
| Clone, 65,536 elements (quick) | 0.02163 | 0.00924 | 0.00495 | 4.37× |
| Add, 8,388,608 elements | 4.85420 | 4.51523 | 3.59833 | 1.35× |
| GELU, 8,388,608 elements | 7.08036 | 6.91298 | 7.22612 | 0.98× |
| Square matmul, 1536 | 15.98744 | 14.60418 | 14.51372 | 1.10× |
| Batched attention matmul, [128×256×64] × [128×64×256] | 4.66169 | 3.62936 | 3.78772 | 1.23× |
| TinyLM forward | 13.42951 | 13.21010 | 10.46259 | 1.28× |
| TinyLM training step | 34.34856 | 34.57177 | 32.98111 | 1.04× |

The mixed profile improves several workloads, but large GELU is slightly
slower and the training-step gain is modest. Placement changes as well as
budgets contribute to these comparisons. These results do not isolate each
setting's effect or establish a best profile for every workload. Earlier
measurements can differ with load, temperature, placement, and run order.

All 37 workload summaries are in [the CSV](threading-results-20260930.csv).
Raw CSV/logs and the exact process runner are retained on the inspected host
under `/tmp/tensorlib-thread-config/results/` and
`/tmp/tensorlib-thread-config/run_profiles.py`.

To reproduce, use the same native Release build, profile settings, and CPU
placement, then invoke each command in three fresh processes:

```sh
./build-bench/bench_tensorlib --suite policy --profile quick --threads 16 --csv policy.csv
./build-bench/bench_tensorlib --suite scaling --profile full --threads 16 --csv scaling.csv
```

Use `--threads 12` for p12. The benchmark ladder sets the global ceiling;
ENV operation caps remain bounded by that ceiling.

## Correctness validation

The final implementation passes all 50 tests with default, mixed, and forced
serial group settings, and all 49 tests in a build without OpenMP. Both builds
enable compiler warnings as errors for the library. ENV tests cover inheritance,
threshold boundaries, runtime/task caps, malformed/overflow values, cached
settings, concurrent initialization, and nested work. Matrix checks compare
every output against a scalar reference under caps 1, 3, and 7. Serial-policy
testing also exposed and fixed inconsistent AdamW state validation before
updates; the non-finite-state regression now runs without OpenMP too.
