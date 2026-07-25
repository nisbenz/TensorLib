CC = gcc
CFLAGS = -O3 -march=native -mtune=native -Wall -Wextra -g -std=c11
SRC = src/tensor/tensor_core.c src/tensor/tensor_alloc.c src/tensor/tensor_view.c src/tensor/tensor_ops.c src/tensor/tensor_reduc.c src/tensor/tensor_matmul.c src/autograd/autograd_core.c src/autograd/autograd_ops.c src/autograd/autograd_view.c src/autograd/autograd_reduc.c src/autograd/autograd_matmul.c src/autograd/autograd_backward.c src/init/rng.c src/nn/parameter.c src/nn/module.c src/nn/linear.c src/nn/mlp.c src/losses/classification.c src/optim/sgd.c
HEADERS = include/tensorlib/tensor.h include/tensorlib/tensor_matmul.h include/tensorlib/autograd.h include/tensorlib/nn.h tests/fixtures/test_common.h
INCLUDES = -Iinclude/tensorlib -Itests/fixtures

BIN = bin
TESTS = $(BIN)/test_tensor_core $(BIN)/test_tensor_alloc $(BIN)/test_tensor_view $(BIN)/test_tensor_ops $(BIN)/test_tensor_reduc $(BIN)/test_tensor_matmul $(BIN)/test_autograd_core $(BIN)/test_autograd_ops $(BIN)/test_autograd_view $(BIN)/test_autograd_reduc $(BIN)/test_autograd_matmul $(BIN)/test_autograd_backward $(BIN)/test_autograd_integration $(BIN)/test_autograd_public_contract $(BIN)/test_nn_rng $(BIN)/test_nn_parameter $(BIN)/test_nn_module $(BIN)/test_nn_init $(BIN)/test_nn_linear $(BIN)/test_nn_loss $(BIN)/test_nn_sgd $(BIN)/test_nn_mlp

all: test

example: $(BIN)/autograd_example
	./$(BIN)/autograd_example

mnist: $(BIN)/mnist_mlp
	./$(BIN)/mnist_mlp

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_tensor_core: tests/unit/tensor/test_tensor_core.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/tensor/test_tensor_core.c $(SRC) -lm

$(BIN)/test_tensor_alloc: tests/unit/tensor/test_tensor_alloc.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/tensor/test_tensor_alloc.c $(SRC) -lm

$(BIN)/test_tensor_view: tests/unit/tensor/test_tensor_view.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/tensor/test_tensor_view.c $(SRC) -lm

$(BIN)/test_tensor_ops: tests/unit/tensor/test_tensor_ops.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/tensor/test_tensor_ops.c $(SRC) -lm

$(BIN)/test_tensor_reduc: tests/unit/tensor/test_tensor_reduc.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/tensor/test_tensor_reduc.c $(SRC) -lm

$(BIN)/test_tensor_matmul: tests/unit/tensor/test_tensor_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/tensor/test_tensor_matmul.c $(SRC) -lm

$(BIN)/test_autograd_core: tests/unit/autograd/test_autograd_core.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_core.c $(SRC) -lm

$(BIN)/test_autograd_ops: tests/unit/autograd/test_autograd_ops.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_ops.c $(SRC) -lm

$(BIN)/test_autograd_view: tests/unit/autograd/test_autograd_view.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_view.c $(SRC) -lm

$(BIN)/test_autograd_reduc: tests/unit/autograd/test_autograd_reduc.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_reduc.c $(SRC) -lm

$(BIN)/test_autograd_matmul: tests/unit/autograd/test_autograd_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_matmul.c $(SRC) -lm

$(BIN)/test_autograd_backward: tests/unit/autograd/test_autograd_backward.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_backward.c $(SRC) -lm

$(BIN)/test_autograd_integration: tests/unit/autograd/test_autograd_integration.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_integration.c $(SRC) -lm

$(BIN)/test_autograd_public_contract: tests/unit/autograd/test_autograd_public_contract.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/autograd/test_autograd_public_contract.c $(SRC) -lm

$(BIN)/test_nn_rng: tests/unit/nn/test_nn_rng.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/nn/test_nn_rng.c $(SRC) -lm

$(BIN)/test_nn_parameter: tests/unit/nn/test_nn_parameter.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/nn/test_nn_parameter.c $(SRC) -lm

$(BIN)/test_nn_module: tests/unit/nn/test_nn_module.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/nn/test_nn_module.c $(SRC) -lm

$(BIN)/test_nn_init: tests/unit/nn/test_nn_init.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/nn/test_nn_init.c $(SRC) -lm

$(BIN)/test_nn_linear: tests/unit/nn/test_nn_linear.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/nn/test_nn_linear.c $(SRC) -lm

$(BIN)/test_nn_loss: tests/unit/nn/test_nn_loss.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/nn/test_nn_loss.c $(SRC) -lm

$(BIN)/test_nn_sgd: tests/unit/optim/test_nn_sgd.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/optim/test_nn_sgd.c $(SRC) -lm

$(BIN)/test_nn_mlp: tests/unit/nn/test_nn_mlp.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ tests/unit/nn/test_nn_mlp.c $(SRC) -lm

$(BIN)/autograd_example: examples/autograd_example.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ examples/autograd_example.c $(SRC) -lm

$(BIN)/mnist_mlp: examples/mnsit/mnist_mlp.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -Iinclude $(INCLUDES) -o $@ examples/mnsit/mnist_mlp.c $(SRC) -lm

$(BIN)/bench_tensor_matmul: benchmarks/matmul/bench_tensor_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_tensor_matmul.c $(SRC) -lm

$(BIN)/bench_tensor_matmul_reuse: benchmarks/matmul/bench_tensor_matmul_reuse.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_tensor_matmul_reuse.c $(SRC) -lm

$(BIN)/bench_tensor_matmul_packed_views: benchmarks/matmul/bench_tensor_matmul_packed_views.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_tensor_matmul_packed_views.c $(SRC) -lm

$(BIN)/bench_openblas_matmul: benchmarks/matmul/bench_openblas_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_openblas_matmul.c $(SRC) -lopenblas -lm

$(BIN)/bench_openblas_matmul_reuse: benchmarks/matmul/bench_openblas_matmul_reuse.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ benchmarks/matmul/bench_openblas_matmul_reuse.c $(SRC) -lopenblas -lm

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; echo; done

benchmark-matmul: $(BIN)/bench_tensor_matmul
	./$(BIN)/bench_tensor_matmul

benchmark-matmul-reuse: $(BIN)/bench_tensor_matmul_reuse
	./$(BIN)/bench_tensor_matmul_reuse

benchmark-packed-views: $(BIN)/bench_tensor_matmul_packed_views
	./$(BIN)/bench_tensor_matmul_packed_views

benchmark-all: benchmark-matmul benchmark-matmul-reuse benchmark-packed-views benchmark-openblas benchmark-openblas-reuse

benchmark-openblas: $(BIN)/bench_openblas_matmul
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./$(BIN)/bench_openblas_matmul

benchmark-openblas-reuse: $(BIN)/bench_openblas_matmul_reuse
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./$(BIN)/bench_openblas_matmul_reuse

clean:
	rm -rf $(BIN)

.PHONY: all test example mnist benchmark-all benchmark-matmul benchmark-matmul-reuse benchmark-packed-views benchmark-openblas benchmark-openblas-reuse clean
