# Changelog

All notable changes to TensorLib are documented in this file.

## [0.3.0] - 2026-09-21

TensorLib 0.3.0 improves CPU training performance and introduces CommandLM as
an experimental end-to-end language-model example. CommandLM demonstrates the
pipeline and safety boundaries; its generated-command quality is not yet
production-ready.

### Training and autograd performance

- Added leaf-only gradient retention so training can discard intermediate
  gradients while preserving the existing retain-all behavior by default.
- Fused sibling slice-gradient accumulation and added fast paths for leading
  and middle-axis reductions, same-shape contributions, and persistent
  gradient merges.
- Parallelized large autograd fills, contiguous gradient reductions,
  million-element tensor operations, row-contiguous and transposed clones,
  large right-hand-side packing, and AdamW validation and gradient norms.
- Improved packed-matrix scheduling by splitting columns adaptively and reduced
  avoidable storage-sized leaf-gradient copies.

On the documented i5-13400F benchmark host at 16 threads, these changes reduced
the approximately 30-million-parameter CommandLM training step by 29.1% at
batch 1 and 20.4% at batch 16. AdamW time fell from about 74.5 ms to 36.1 ms.
These results are workload- and machine-specific; the full methodology and
phase breakdown are in `docs/commandlm_cpu_optimization.md`.

### Experimental CommandLM example

- Added a configurable 29.6-million-parameter structured-command model with
  training, validation, inference, explicit resume support, and atomic
  checkpoints containing model, AdamW, and RNG state.
- Added a deterministic 1,024-token BPE-style tokenizer with byte fallback,
  versioned vocabulary persistence, and round-trip tests.
- Added deterministic corpus generation plus import and audit tooling for the
  NL2Bash and CLI-1M datasets.
- Added conservative output validation: the example emits allowlisted command
  records, rejects shell metacharacters and dangerous commands, and never
  executes generated text.
- Added CommandLM-specific benchmark shapes and labeled training-phase metrics.

CommandLM is released as an experimental example. Its current model outputs may
be poor without a carefully curated corpus, sufficient training, and separate
quality evaluation, so generated commands should be inspected rather than
trusted or executed.

### Quality

- Expanded tests for leaf-only gradient retention, fused slice gradients,
  tensor layouts, parallel AdamW validation and rollback, and tokenizer
  persistence.
- Hardened corpus parsing, batch validation, checkpoint compatibility checks,
  and explicit checkpoint resume behavior.

## [0.2.0] - 2026-09-06

TensorLib 0.2.0 is a performance-focused release. It substantially reduces
the cost of TinyLM forward and training workloads while preserving deterministic
gradients, checkpoint compatibility, and the public tensor API.

### Tensor and matrix kernels

- Parallelized contiguous, packed, and batched matrix multiplication with an
  adaptive OpenMP thread policy that avoids parallel overhead on small jobs.
- Added retained, transpose-aware matrix packs and reusable packed workspaces.
  Linear layers now cache packed weights across forward and backward passes and
  invalidate the cache when weights change.
- Added direct batched right-hand-side gradient multiplication and direct
  projected-weight gradient contraction. Linear weight gradients use a
  vectorized, deterministic GEMM path.
- Added contiguous fast paths for tensor cloning, reductions, and broadcast
  binary operations. Strided cloning now uses a tiled copy path, and broadcast
  iteration uses stride cursors with parallel processing for large rows.
- Removed redundant per-element metadata validation from coordinate helpers.

### Fused neural-network operations

- Replaced composed LayerNorm with a fused autograd kernel and parallelized its
  deterministic backward pass.
- Added fused row-wise softmax and log-softmax kernels, an indexed
  cross-entropy kernel, and parallel softmax and cross-entropy backward paths.
- Fused and parallelized GELU backward.
- Specialized contiguous slice backward to avoid generic scatter overhead.

### Autograd and training

- Transfers first persistent gradients without copying and accumulates
  compatible contributions in place, avoiding temporary tensors.
- Fuses broadcast-gradient reduction and reduces broadcast gradients at their
  producers when possible.
- Reuses graph traversal storage and routes Linear backward through direct
  gradient kernels.
- Restores serial fast paths for small gradients so parallel scheduling does
  not penalize small models and tensors.
- AdamW now has a contiguous single-pass update path, specialized gradient
  scanning, and deterministic parallel updates.

### Measured impact

On the repository's i5-13400F reference host, using the full benchmark profile
and comparing matched rows with the pre-optimization baseline:

- LayerNorm throughput increased from 251,803 to 3,239,353 tokens/s (+1,186%).
- Transposed contiguous-copy throughput increased from 1.31 to 15.73 GB/s
  (+1,099%).
- Causal-attention throughput increased from 31,688 to 114,511 tokens/s
  (+261%).
- TinyLM forward throughput increased from 4,340 to 11,447 tokens/s (+164%).
- Decoder-block throughput increased from 19,449 to 46,829 tokens/s (+141%).
- Across the thread-scaling ladder, TinyLM forward rose from approximately
  4.6-5.2k to 11.5-19.5k tokens/s, and training rose from approximately 0.8k
  to 1.5-1.8k tokens/s.

These figures are machine- and workload-specific. The benchmark suite records
the compiler, host, thread count, timing distribution, and correctness status
needed to reproduce and interpret them.

### Checkpoint and examples

- The release includes a new `tiny_lm.chk` for the fixed TinyLM preset: four
  decoder blocks, width 192, six heads, context length 128, byte vocabulary 256,
  and 1,902,976 trainable parameters.
- The checkpoint uses the versioned transactional format and contains model
  parameters, AdamW moments and step counters, and RNG state so training can be
  resumed.
- The release checkpoint's SHA-256 digest is
  `bc11a667129224f64f9ede0788bb6367206dc912ac1e42a5bca5c5115edd21fc`.

### Build, benchmarks, and quality

- Added an installable CMake package and portable switches for OpenMP, native
  CPU tuning, tests, examples, benchmarks, warnings-as-errors, and sanitizers.
- Expanded CI coverage across Linux, macOS, Windows, GCC, Clang, and package
  consumption.
- Added a unified benchmark harness for tensor kernels, autograd, neural-network
  workloads, thread scaling, training phases, allocation pressure, and matched
  OpenBLAS and PyTorch comparisons.
- Added correctness coverage for fused kernels, packed-matrix cache invalidation,
  strided and broadcast layouts, deterministic parallel gradients, checkpoint
  behavior, and benchmark timing.

### Fixes

- Preserved training gradients through the fused LayerNorm path.
- Sized the backward-contribution workspace by graph arity.
- Configured benchmark thread counts before measurement and added lifecycle
  resets so timing samples are comparable.
- Kept warning-clean serial and strict Clang builds.

## [0.1.0] - 2026-07-27

- Initial release of the tensor core, reverse-mode autograd engine, neural-network
  modules, optimizers, checkpointing, and TinyLM example.

[0.3.0]: https://github.com/nisbenz/tensorlib/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nisbenz/tensorlib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nisbenz/tensorlib/releases/tag/v0.1.0
