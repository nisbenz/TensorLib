CC ?= gcc

# Core optimization flags
# -flto: link-time optimization (cross-module inlining, dead code elimination)
# -fno-math-errno: math functions don't set errno (avoids stores to errno)
# -fno-signed-zeros: treats -0 as +0 for more algebraic transforms
# -fno-trapping-math: assume no FP exceptions are generated
# -freciprocal-math: allows x/y -> x*(1/y) and similar
# -funroll-loops: unroll loops where profitable
OPT_FLAGS = -flto -fno-math-errno -fno-signed-zeros -fno-trapping-math -freciprocal-math -funroll-loops

# Release build flags (with debug symbols)
CFLAGS = -O3 -march=native -mtune=native $(OPT_FLAGS) -Wall -Wextra -g -std=c99 -fopenmp

# PGO generation flags (no -g to avoid profile pollution)
PGO_CFLAGS = -O3 -march=native -mtune=native $(OPT_FLAGS) -Wall -Wextra -std=c99 -fopenmp -fprofile-generate

# PGO use flags (no -g for maximum performance)
PGO_USE_CFLAGS = -O3 -march=native -mtune=native $(OPT_FLAGS) -Wall -Wextra -std=c99 -fopenmp -fprofile-use
SRC = src/tensor/tensor_core.c src/tensor/tensor_alloc.c src/tensor/tensor_view.c src/tensor/tensor_ops.c src/tensor/tensor_gather.c src/tensor/tensor_reduc.c src/tensor/tensor_matmul.c src/autograd/autograd_core.c src/autograd/autograd_ops.c src/autograd/autograd_view.c src/autograd/autograd_gather.c src/autograd/autograd_reduc.c src/autograd/autograd_matmul.c src/autograd/autograd_backward.c src/init/rng.c src/nn/parameter.c src/nn/module.c src/nn/linear.c src/nn/embedding.c src/nn/positional_embedding.c src/nn/layer_norm.c src/nn/dropout.c src/nn/multihead_attention.c src/nn/decoder_block.c src/nn/decoder.c src/nn/mlp.c src/losses/classification.c src/nn/causal_mask.c src/optim/sgd.c src/optim/optim_common.c src/optim/adamw.c src/serialization/checkpoint.c
HEADERS = include/tensorlib/tensor.h include/tensorlib/tensor_matmul.h include/tensorlib/autograd.h include/tensorlib/nn.h tests/fixtures/test_common.h
INCLUDES = -Iinclude/tensorlib -Itests/fixtures

# OpenBLAS is optional. The OpenBLAS comparison benchmarks are only built when
# OpenBLAS (header cblas.h + linkable libopenblas) is present, so targets like
# benchmark-all still work on systems without it. To point at a custom install,
# set OPENBLAS_INCLUDE and OPENBLAS_LIB on the make command line.
HAVE_OPENBLAS := $(shell printf '#include <cblas.h>\nextern void openblas_set_num_threads(int);\nint main(void){openblas_set_num_threads(1); return 0;}\n' | $(CC) $(OPENBLAS_INCLUDE) -x c - $(OPENBLAS_LIB) -lopenblas -o /tmp/tensorlib_openblas_probe 2>/dev/null && echo yes; rm -f /tmp/tensorlib_openblas_probe)
ifeq ($(strip $(HAVE_OPENBLAS)),yes)
OPENBLAS_BINS = $(BIN)/bench_openblas_matmul $(BIN)/bench_openblas_matmul_reuse
else
OPENBLAS_BINS =
endif
BIN = bin
TESTS = $(BIN)/test_tensor_core $(BIN)/test_tensor_alloc $(BIN)/test_tensor_view $(BIN)/test_tensor_ops $(BIN)/test_tensor_reduc $(BIN)/test_tensor_matmul $(BIN)/test_autograd_core $(BIN)/test_autograd_ops $(BIN)/test_autograd_view $(BIN)/test_autograd_gather $(BIN)/test_autograd_reduc $(BIN)/test_autograd_matmul $(BIN)/test_autograd_backward $(BIN)/test_autograd_integration $(BIN)/test_autograd_public_contract $(BIN)/test_nn_rng $(BIN)/test_nn_parameter $(BIN)/test_nn_module $(BIN)/test_nn_init $(BIN)/test_nn_linear $(BIN)/test_nn_embedding $(BIN)/test_nn_positional_embedding $(BIN)/test_nn_layer_norm $(BIN)/test_nn_dropout $(BIN)/test_nn_multihead_attention $(BIN)/test_nn_decoder_block $(BIN)/test_nn_decoder $(BIN)/test_nn_loss $(BIN)/test_nn_causal_mask $(BIN)/test_nn_sgd $(BIN)/test_nn_adamw $(BIN)/test_nn_checkpoint $(BIN)/test_nn_mlp

all: test

example: $(BIN)/autograd_example
	./$(BIN)/autograd_example

mnist: $(BIN)/mnist_mlp
	./$(BIN)/mnist_mlp

tiny-lm: $(BIN)/tiny_lm
	./$(BIN)/tiny_lm --help

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_tensor_%: tests/unit/tensor/test_tensor_%.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ $< $(SRC) -lm

$(BIN)/test_autograd_%: tests/unit/autograd/test_autograd_%.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ $< $(SRC) -lm

$(BIN)/test_nn_%: tests/unit/nn/test_nn_%.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ $< $(SRC) -lm

$(BIN)/test_nn_%: tests/unit/optim/test_nn_%.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ $< $(SRC) -lm

$(BIN)/autograd_example: examples/autograd_example.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ examples/autograd_example.c $(SRC) -lm

$(BIN)/mnist_mlp: examples/mnist/mnist_mlp.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ examples/mnist/mnist_mlp.c $(SRC) -lm

$(BIN)/tiny_lm: examples/tiny_lm/tiny_lm.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ examples/tiny_lm/tiny_lm.c $(SRC) -lm


$(BIN)/bench_tensor_matmul: benchmarks/matmul/bench_tensor_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_tensor_matmul.c $(SRC) -lm

$(BIN)/bench_tensor_matmul_reuse: benchmarks/matmul/bench_tensor_matmul_reuse.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_tensor_matmul_reuse.c $(SRC) -lm

$(BIN)/bench_tensor_matmul_packed_views: benchmarks/matmul/bench_tensor_matmul_packed_views.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_tensor_matmul_packed_views.c $(SRC) -lm


ifeq ($(strip $(HAVE_OPENBLAS)),yes)
$(BIN)/bench_openblas_matmul: benchmarks/matmul/bench_openblas_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(OPENBLAS_INCLUDE) -o $@ benchmarks/matmul/bench_openblas_matmul.c $(SRC) $(OPENBLAS_LIB) -lopenblas -lm

$(BIN)/bench_openblas_matmul_reuse: benchmarks/matmul/bench_openblas_matmul_reuse.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(OPENBLAS_INCLUDE) -o $@ benchmarks/matmul/bench_openblas_matmul_reuse.c $(SRC) $(OPENBLAS_LIB) -lopenblas -lm
endif

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; echo; done

benchmark-matmul: $(BIN)/bench_tensor_matmul
	./$(BIN)/bench_tensor_matmul

benchmark-matmul-reuse: $(BIN)/bench_tensor_matmul_reuse
	./$(BIN)/bench_tensor_matmul_reuse

benchmark-packed-views: $(BIN)/bench_tensor_matmul_packed_views
	./$(BIN)/bench_tensor_matmul_packed_views

ifeq ($(strip $(HAVE_OPENBLAS)),yes)
benchmark-openblas: $(BIN)/bench_openblas_matmul
	./$(BIN)/bench_openblas_matmul

benchmark-openblas-reuse: $(BIN)/bench_openblas_matmul_reuse
	./$(BIN)/bench_openblas_matmul_reuse

benchmark-compare:
	@echo "===== TensorLib (OpenMP) ====="
	OMP_NUM_THREADS=1 ./$(BIN)/bench_tensor_matmul | head -4
	@echo ""
	OMP_NUM_THREADS= ./$(BIN)/bench_tensor_matmul | head -4
	@echo ""
	@echo "===== OpenBLAS ====="
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./$(BIN)/bench_openblas_matmul | head -4
	@echo ""
	OMP_NUM_THREADS= OPENBLAS_NUM_THREADS=0 ./$(BIN)/bench_openblas_matmul | head -4
	@echo ""

benchmark-all: benchmark-matmul benchmark-matmul-reuse benchmark-packed-views benchmark-openblas benchmark-openblas-reuse
else
benchmark-openblas:
	@echo "OpenBLAS not found: skipping OpenBLAS comparison benchmarks."

benchmark-openblas-reuse:
	@echo "OpenBLAS not found: skipping OpenBLAS comparison benchmarks."

benchmark-compare:
	@echo "OpenBLAS not found: cannot run comparison. Run 'make benchmark-matmul' instead."

benchmark-all: benchmark-matmul benchmark-matmul-reuse benchmark-packed-views
endif

clean:
	rm -rf $(BIN)

.PHONY: all test example mnist tiny-lm benchmark-all benchmark-matmul benchmark-matmul-reuse benchmark-packed-views benchmark-openblas benchmark-openblas-reuse benchmark-compare clean
