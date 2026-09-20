# CommandLM

CommandLM is a standalone structured-command example built on TensorLib's
configurable GPT-style decoder. It translates natural-language developer
requests into an allowlisted command record; it never executes generated text.

The default model has 9 decoder blocks, width 512, 8 attention heads, context
256, and a 1,024-token deterministic BPE-style tokenizer. It contains about
29.6M trainable parameters. The tokenizer includes byte fallback tokens and a
versioned binary vocabulary file.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target command_lm
```

## Prepare and train

Generate a reproducible developer-command corpus and run the practical 16-
thread training configuration:

```sh
OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE ./build/command_lm command.txt \
  --generate-corpus 100000 \
  --tokenizer command.tok \
  --checkpoint command.chk \
  --steps 50000 \
  --batch-size 16 \
  --threads 16
```

To import the MIT-licensed NL2Bash corpus into the same record format, run:

```sh
python3 scripts/import_nl2bash.py /path/to/nl2bash/data/bash nl2bash.safe.txt
```

The importer keeps only non-destructive commands and simple pipelines, removes
ambiguous requests, shuffles deterministically, and writes provenance metadata
next to the corpus. Each record has a `REQUEST:` section and a raw Bash
`COMMAND:` section. Inference rejects shell metacharacters and a conservative
dangerous-command list; generated commands are never executed.

Before training, audit an existing corpus with:

```sh
python3 scripts/audit_command_corpus.py command.txt
```

For a quick pipeline check, use `--steps 0 --generate 0`, or use a few updates
with `--steps 1 --eval-interval 1`. Checkpoints include model, AdamW, and RNG
state. Resuming is explicit with `--resume`; a compatible checkpoint is never
loaded silently.
