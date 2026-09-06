# Contributing to TensorLib

Thanks for improving TensorLib. Keep changes focused, portable, and easy to
review.

## Build and test

The default workflow builds the library, examples, and test suite:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python3 scripts/check_markdown_links.py
```

Before opening a pull request, also run a strict build:

```sh
cmake -S . -B build-strict \
  -DTENSORLIB_WARNINGS_AS_ERRORS=ON \
  -DTENSORLIB_ENABLE_OPENMP=OFF
cmake --build build-strict --parallel
ctest --test-dir build-strict --output-on-failure
```

For memory and undefined-behavior checks, enable the sanitizer option with a
GCC or Clang debug build:

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORLIB_ENABLE_SANITIZERS=ON \
  -DTENSORLIB_ENABLE_OPENMP=OFF
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

## Change guidelines

- Follow the existing C99 style and keep compiler warnings at zero.
- Add tests for public behavior, failure paths, and non-contiguous views.
- Keep ownership rules explicit whenever a function accepts or returns memory.
- Avoid machine-specific instructions unless a portable fallback and runtime
  dispatch are present.
- Keep commits small and give each commit one clear purpose.
- Update the relevant guide when public behavior or build controls change.

## Performance changes

Correctness tests must pass before benchmarking. Record the compiler, flags,
CPU, thread count, matrix shapes, and repeated-run variance with any performance
claim. See the [benchmark guide](docs/benchmarks.md) for the standard commands
and reporting format.
