# TensorLib

A from-scratch deep learning tensor library in C with autograd, Transformer modules, and CPU optimizations.

## Features

- **Tensor library** — N-dimensional float32 arrays with NumPy-style broadcasting, strided views (transpose, reshape, slice, expand, squeeze), and reference-counted storage
- **Autograd engine** — Dynamic reverse-mode automatic differentiation over 23 differentiable operations, with stale-graph detection and gradient accumulation
- **Neural network modules** — Linear, Embedding, LayerNorm, Dropout, Multi-head Causal Self-Attention, Decoder Block (pre-norm), full Decoder stack, MLP
- **Loss functions** — Cross-entropy (with built-in log-softmax), Softmax, LogSoftmax
- **Optimizers** — SGD, AdamW (decoupled weight decay, bias correction, gradient clipping)
- **Checkpointing** — Versioned, atomic (transaction-safe) save/load with optimizer and RNG state
- **SIMD acceleration** — AVX2+FMA matmul micro-kernel with tile-based blocked algorithm
- **OpenMP parallelism** — Multi-threaded matmul kernels
- **Deterministic RNG** — Splitmix64 PRNG with uniform and normal (Box-Muller) distributions
- **No external ML dependencies** — Pure C, no Python, CUDA, or third-party ML libraries required

## Build

### CMake (recommended)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Requires a compiler with OpenMP support (GCC, Clang, MSVC).

### Makefile (GNU Make, GCC)

```sh
make          # build and run all tests
make tiny-lm  # build the TinyLM example
make mnist    # build the MNIST example
make clean
```

Profile-guided optimization (PGO) for TinyLM:

```sh
make pgo-tiny-lm CORPUS=corpus.txt
```

## Project structure

```
include/tensorlib/     # Public API headers
  tensor.h             #   Core tensor structs and operations
  tensor_matmul.h      #   Matmul internals
  autograd.h           #   Autograd engine API
  nn.h                 #   Neural network module API

src/                   # Source implementation
  tensor/              #   Tensor ops, views, reductions, matmul
  autograd/            #   Autograd forward/backward ops, graph traversal
  nn/                  #   Module system, layers (Linear, Embedding, LayerNorm,
                       #   Dropout, MultiheadAttention, DecoderBlock, Decoder, MLP)
  losses/              #   Cross-entropy, Softmax, LogSoftmax
  optim/               #   SGD, AdamW, gradient clipping
  init/                #   PRNG and weight initialization
  serialization/       #   Checkpoint save/load

tests/                 # Test suite
  unit/
    tensor/            #   6 test files
    autograd/          #   9 test files
    nn/                #   16 test files
    optim/             #   2 test files

examples/
  autograd_example.c   #   Computation graph demo
  tiny_lm/             #   Byte-level decoder language model (~1.9M params)
  mnsit/               #   MNIST MLP classifier

benchmarks/
  matmul/              #   Matmul performance benchmarks
```

## Examples

### Autograd demo

```sh
./build/autograd_example
```

Builds a computation graph `input @ weights + bias -> exp -> mean`, backpropagates, and prints all gradients.

### MNIST MLP

```sh
./build/mnist_mlp
```

784→128 ReLU→10 Softmax MLP trained with SGD and cross-entropy loss.

### TinyLM — byte-level language model

```sh
./build/tiny_lm corpus.txt --steps 1000 --generate 200 --prompt "Hello"
```

4-layer, 192-width, 6-head decoder transformer (~1.9M params) trained with AdamW on raw byte corpora. See [examples/tiny_lm/README.md](examples/tiny_lm/README.md) for details.

## Testing

33 unit test executables covering tensors, autograd, all NN modules, optimizers, loss functions, and checkpointing.

```sh
cmake --build build --config Release --target test
# or with Makefile:
make test
```

## Benchmarks

Matmul benchmarks compare TensorLib's blocked AVX2 kernel against OpenBLAS:

```sh
make benchmark-compare
```

## Design notes

- **Eager execution** — Graph is built dynamically during forward pass; no JIT compilation
- **Manual memory management** — Reference counting for `Storage`, `ag_tensor`, and `ag_node`; no garbage collector
- **Single-threaded API** — Not thread-safe; assumes single-threaded usage
- **CPU only** — No GPU support
- **SIMD scope** — AVX2+FMA only used in matmul; element-wise ops, activations, norm, softmax are scalar (optimization opportunity tracked in `optimizations.md`)
