CMAKE ?= cmake
BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= 2
CMAKE_FLAGS ?=

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

benchmark-matmul: benchmarks
	$(BUILD_DIR)/bench_tensor_matmul

benchmark-matmul-reuse: benchmarks
	$(BUILD_DIR)/bench_tensor_matmul_reuse

benchmark-packed-views: benchmarks
	$(BUILD_DIR)/bench_tensor_matmul_packed_views

benchmark-compare: benchmarks
	@echo "===== TensorLib ====="
	$(BUILD_DIR)/bench_tensor_matmul
	@if test -x "$(BUILD_DIR)/bench_openblas_matmul"; then \
		echo "===== OpenBLAS ====="; $(BUILD_DIR)/bench_openblas_matmul; \
	else echo "OpenBLAS not found; comparison skipped."; fi

clean:
	@test "$(BUILD_DIR)" != "." && test "$(BUILD_DIR)" != "/"
	$(CMAKE) -E remove_directory "$(BUILD_DIR)"

.PHONY: all configure build test example mnist tiny-lm benchmarks \
	benchmark-matmul benchmark-matmul-reuse benchmark-packed-views \
	benchmark-compare clean
