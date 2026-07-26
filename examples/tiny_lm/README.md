# TensorLib TinyLM

TinyLM is a byte-level decoder language model intended to test end-to-end
corpus training before larger experiments.

## Fixed 2M preset

- 4 decoder blocks
- width 192
- 6 attention heads
- context length 128 bytes
- vocabulary 256 bytes
- approximately 1.9 million trainable parameters
- learned positional embeddings, dropout 0.1, AdamW, gradient clipping

Because the tokenizer is raw bytes, any file can be used as a corpus and no
tokenizer model is required. UTF-8 characters may occupy multiple tokens.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target tiny_lm
```

## Train

Start with a short timing run:

```sh
./build/tiny_lm corpus.txt --steps 10 --generate 0
```

Then train and generate:

```sh
./build/tiny_lm corpus.txt \
  --steps 1000 \
  --checkpoint tiny_lm.chk \
  --prompt "Once upon a time"
```

Resume the same architecture and optimizer:

```sh
./build/tiny_lm corpus.txt \
  --steps 1000 \
  --checkpoint tiny_lm.chk \
  --resume
```

On multi-config Windows generators, the executable is normally under
`build/Release/tiny_lm.exe`.

Run `tiny_lm --help` for all controls. Checkpoints include model parameters,
AdamW moments and steps, and RNG state. `--steps` counts updates in the current
invocation; it does not represent an absolute global step.

## Corpus guidance

The 90/10 split is positional: the final 10% is validation data. For meaningful
validation, concatenate independent documents and shuffle document order before
creating the final corpus file. Avoid placing only one author, topic, or source
in the validation tail.

Begin with 1-10 MB of clean, repetitive, single-domain text. A 2M byte model is
best used for style imitation, constrained command completion, source-code or
configuration autocomplete, log/event sequence modeling, and simple short-form
stories. It is not a general question-answering model.

Good first datasets:

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is the
  strongest test of whether a very small model can learn simple English. Start
  with a shuffled 5-20 MB subset rather than the complete multi-gigabyte file.
- [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
  is a convenient ~1 MB smoke-test corpus. It demonstrates style and character
  structure, but its archaic language is not the easiest route to useful text.
- A synthetic command corpus is the best practical choice. Store repeated
  examples such as `USER: turn on the kitchen light\nCOMMAND: LIGHT KITCHEN
  ON\n\n`, with many paraphrases and a small, exact output grammar.
- Clean C source, configuration files, or device logs work well for local
  completion and sequence modeling. Normalize timestamps, IDs, and large
  numeric values so the small model spends capacity on recurring structure.

On the repository's i5-13400F test machine, a Release build completed one
batch-1 update in about 0.66 seconds, or roughly 193 byte tokens/second. This
makes 1,000 updates about an 11-minute timing run and 50,000 updates roughly a
9-hour run before validation and checkpoint overhead.
