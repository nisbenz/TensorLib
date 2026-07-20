CC = gcc
CFLAGS = -O3 -march=native -mtune=native -Wall -Wextra -g -std=c11
SRC = src/tensor_core.c src/tensor_alloc.c src/tensor_view.c src/tensor_ops.c src/tensor_reduc.c src/tensor_matmul.c
HEADERS = include/tensor.h include/tensor_matmul.h

BIN = bin
TESTS = $(BIN)/test_tensor_core $(BIN)/test_tensor_alloc $(BIN)/test_tensor_view $(BIN)/test_tensor_ops $(BIN)/test_tensor_reduc $(BIN)/test_tensor_matmul

all: test

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_tensor_core: tests/test_tensor_core.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_core.c $(SRC) -lm

$(BIN)/test_tensor_alloc: tests/test_tensor_alloc.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_alloc.c $(SRC) -lm

$(BIN)/test_tensor_view: tests/test_tensor_view.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_view.c $(SRC) -lm

$(BIN)/test_tensor_ops: tests/test_tensor_ops.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_ops.c $(SRC) -lm

$(BIN)/test_tensor_reduc: tests/test_tensor_reduc.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_reduc.c $(SRC) -lm

$(BIN)/test_tensor_matmul: tests/test_tensor_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_matmul.c $(SRC) -lm

$(BIN)/bench_tensor_matmul: benchmarks/bench_tensor_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ benchmarks/bench_tensor_matmul.c $(SRC) -lm

$(BIN)/bench_tensor_matmul_reuse: benchmarks/bench_tensor_matmul_reuse.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ benchmarks/bench_tensor_matmul_reuse.c $(SRC) -lm

$(BIN)/bench_tensor_matmul_packed_views: benchmarks/bench_tensor_matmul_packed_views.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ benchmarks/bench_tensor_matmul_packed_views.c $(SRC) -lm

$(BIN)/bench_openblas_matmul: benchmarks/bench_openblas_matmul.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ benchmarks/bench_openblas_matmul.c $(SRC) -lopenblas -lm

$(BIN)/bench_openblas_matmul_reuse: benchmarks/bench_openblas_matmul_reuse.c $(SRC) $(HEADERS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ benchmarks/bench_openblas_matmul_reuse.c $(SRC) -lopenblas -lm

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

.PHONY: all test benchmark-all benchmark-matmul benchmark-matmul-reuse benchmark-packed-views benchmark-openblas benchmark-openblas-reuse clean
