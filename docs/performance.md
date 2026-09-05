# Performance and benchmarking

TensorLib includes focused matrix-multiplication benchmarks for measuring its
blocked scalar path, runtime-dispatched AVX2/FMA kernel, packed right-hand-side
reuse, and OpenMP scaling. OpenBLAS is an optional comparison dependency.

## Reproducible build

Use a separate native-tuned build so normal binaries remain portable:

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DTENSORLIB_BUILD_TESTS=OFF \
  -DTENSORLIB_BUILD_EXAMPLES=OFF \
  -DTENSORLIB_BUILD_BENCHMARKS=ON \
  -DTENSORLIB_NATIVE_OPTIMIZATIONS=ON
cmake --build build-bench --parallel
```

The Makefile provides the equivalent convenience commands:

```sh
make benchmark-matmul
make benchmark-matmul-reuse
make benchmark-packed-views
make benchmark-compare
```

OpenBLAS comparison targets are created only when CMake finds both its headers
and library. TensorLib's own benchmarks remain available without it.

## Reporting results

Always report enough context to reproduce a result:

- CPU model and operating system
- compiler name and version
- CMake options and build type
- `OMP_NUM_THREADS` and OpenBLAS thread count
- tensor shapes, warm-up count, and measured repetitions
- median throughput from multiple process runs

Avoid comparing a multithreaded run with a single-threaded baseline. For a fair
single-thread comparison, set both `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`.

## Optimization roadmap

Potential future work includes ARM NEON and AVX-512 kernels, fused element-wise
and normalization operations, attention-specific kernels, and quantized data
types. Each optimization should preserve the scalar fallback, add numerical
tests, and include before/after results produced with the process above.
