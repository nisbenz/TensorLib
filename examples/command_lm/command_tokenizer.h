#ifndef COMMAND_TOKENIZER_H
#define COMMAND_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

enum {
    COMMAND_TOKENIZER_BASE = 256,
    COMMAND_TOKENIZER_VOCAB = 1024,
    COMMAND_TOKENIZER_MAX_TOKEN_BYTES = 32
};

typedef struct {
    uint16_t left;
    uint16_t right;
    uint32_t count;
} command_merge;

typedef struct {
    char* tokens[COMMAND_TOKENIZER_VOCAB];
    size_t lengths[COMMAND_TOKENIZER_VOCAB];
    command_merge merges[COMMAND_TOKENIZER_VOCAB - COMMAND_TOKENIZER_BASE];
    size_t merge_count;
} command_tokenizer;

void command_tokenizer_init(command_tokenizer* tokenizer);
void command_tokenizer_destroy(command_tokenizer* tokenizer);

int command_tokenizer_train(command_tokenizer* tokenizer,
                            const unsigned char* text,
                            size_t length,
                            size_t target_vocab);

int command_tokenizer_encode(const command_tokenizer* tokenizer,
                             const unsigned char* text,
                             size_t length,
                             uint16_t* output,
                             size_t capacity,
                             size_t* output_length);

int command_tokenizer_decode(const command_tokenizer* tokenizer,
                             const uint16_t* ids,
                             size_t count,
                             unsigned char* output,
                             size_t capacity,
                             size_t* output_length);

int command_tokenizer_save(const command_tokenizer* tokenizer, const char* path);
int command_tokenizer_load(command_tokenizer* tokenizer, const char* path);

#endif
