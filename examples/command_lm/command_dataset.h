#ifndef COMMAND_DATASET_H
#define COMMAND_DATASET_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    unsigned char* bytes;
    size_t size;
    size_t train_size;
} command_corpus;

int command_corpus_generate(command_corpus* corpus, size_t examples,
                            uint64_t seed);
int command_corpus_load(command_corpus* corpus, const char* path);
int command_corpus_save(const command_corpus* corpus, const char* path);
void command_corpus_destroy(command_corpus* corpus);

#endif
