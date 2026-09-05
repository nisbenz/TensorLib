CMAKE ?= cmake
BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= 2
CMAKE_FLAGS ?=
BENCHMARK_CSV ?= benchmark-results.csv
BENCHMARK_THREADS ?=
BENCHMARK_REFERENCE_THREADS ?= 1

all: test

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_FLAGS)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel $(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) -C $(BUILD_TYPE) --output-on-failure

example: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) --target autograd_example
	$(BUILD_DIR)/autograd_example

mnist: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) --target mnist_mlp
	$(BUILD_DIR)/mnist_mlp

tiny-lm: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) --target tiny_lm
	$(BUILD_DIR)/tiny_lm --help

benchmarks:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DTENSORLIB_BUILD_BENCHMARKS=ON -DTENSORLIB_NATIVE_OPTIMIZATIONS=ON $(CMAKE_FLAGS)
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel $(JOBS)

benchmark-quick: benchmarks
	python3 scripts/run_benchmarks.py --executable $(BUILD_DIR)/bench_tensorlib \
		--profile quick --csv $(BENCHMARK_CSV) \
		$(if $(BENCHMARK_THREADS),--threads $(BENCHMARK_THREADS),)

benchmark-full: benchmarks
	python3 scripts/run_benchmarks.py --executable $(BUILD_DIR)/bench_tensorlib \
		--profile full --csv $(BENCHMARK_CSV) \
		$(if $(BENCHMARK_THREADS),--threads $(BENCHMARK_THREADS),)

benchmark-kernels: benchmarks
	python3 scripts/run_benchmarks.py --executable $(BUILD_DIR)/bench_tensorlib \
		--suite kernels --profile quick --csv $(BENCHMARK_CSV)

benchmark-scaling: benchmarks
	python3 scripts/run_benchmarks.py --executable $(BUILD_DIR)/bench_tensorlib \
		--suite scaling --profile quick --csv $(BENCHMARK_CSV)

benchmark-compare: benchmark-quick
	@if test -x "$(BUILD_DIR)/bench_openblas"; then \
		$(BUILD_DIR)/bench_openblas --profile quick \
			--threads $(BENCHMARK_REFERENCE_THREADS) \
			--csv benchmark-results-openblas.csv; \
	else echo "OpenBLAS not found; comparison skipped."; fi
	@if python3 -c "import torch" >/dev/null 2>&1; then \
		python3 benchmarks/bench_pytorch.py --profile quick \
			--threads $(BENCHMARK_REFERENCE_THREADS) \
			--csv benchmark-results-pytorch.csv; \
	else echo "PyTorch not found; comparison skipped."; fi

clean:
	@test "$(BUILD_DIR)" != "." && test "$(BUILD_DIR)" != "/"
	$(CMAKE) -E remove_directory "$(BUILD_DIR)"

.PHONY: all configure build test example mnist tiny-lm benchmarks \
	benchmark-quick benchmark-full benchmark-kernels benchmark-scaling \
	benchmark-compare clean
