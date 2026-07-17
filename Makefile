CC = gcc
CFLAGS = -Wall -Wextra -g -std=c11
SRC = src/tensor_core.c src/tensor_alloc.c src/tensor_view.c src/tensor_ops.c src/tensor_reduc.c

BIN = bin
TESTS = $(BIN)/test_tensor_core $(BIN)/test_tensor_alloc $(BIN)/test_tensor_view $(BIN)/test_tensor_ops $(BIN)/test_tensor_reduc

all: test

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_tensor_core: tests/test_tensor_core.c $(SRC) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_core.c $(SRC) -lm

$(BIN)/test_tensor_alloc: tests/test_tensor_alloc.c $(SRC) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_alloc.c $(SRC) -lm

$(BIN)/test_tensor_view: tests/test_tensor_view.c $(SRC) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_view.c $(SRC) -lm

$(BIN)/test_tensor_ops: tests/test_tensor_ops.c $(SRC) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_ops.c $(SRC) -lm

$(BIN)/test_tensor_reduc: tests/test_tensor_reduc.c $(SRC) | $(BIN)
	$(CC) $(CFLAGS) -o $@ tests/test_tensor_reduc.c $(SRC) -lm

test: $(TESTS)
	@for t in $(TESTS); do ./$$t || exit 1; echo; done

clean:
	rm -rf $(BIN)

.PHONY: all test clean
