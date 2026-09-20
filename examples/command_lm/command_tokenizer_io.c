#include "command_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t longest_token(const command_tokenizer* tokenizer,
                            const unsigned char* text, size_t length,
                            size_t position, uint16_t* token)
{
    size_t best = 1;
    uint16_t best_token = text[position];
    for (size_t index = COMMAND_TOKENIZER_BASE;
         index < COMMAND_TOKENIZER_BASE + tokenizer->merge_count; ++index) {
        size_t width = tokenizer->lengths[index];
        if (width <= best || width > length - position) continue;
        if (memcmp(tokenizer->tokens[index], text + position, width) == 0) {
            best = width;
            best_token = (uint16_t)index;
        }
    }
    *token = best_token;
    return best;
}

int command_tokenizer_encode(const command_tokenizer* tokenizer,
                             const unsigned char* text, size_t length,
                             uint16_t* output, size_t capacity,
                             size_t* output_length)
{
    size_t count = 0;
    if (tokenizer == NULL || text == NULL || output == NULL ||
        output_length == NULL) return -1;
    for (size_t position = 0; position < length;) {
        uint16_t token;
        size_t width = longest_token(tokenizer, text, length, position, &token);
        if (count == capacity) return -1;
        output[count++] = token;
        position += width;
    }
    *output_length = count;
    return 0;
}

int command_tokenizer_decode(const command_tokenizer* tokenizer,
                             const uint16_t* ids, size_t count,
                             unsigned char* output, size_t capacity,
                             size_t* output_length)
{
    size_t length = 0;
    if (tokenizer == NULL || ids == NULL || output == NULL ||
        output_length == NULL) return -1;
    for (size_t index = 0; index < count; ++index) {
        uint16_t token = ids[index];
        size_t width;
        if (token >= COMMAND_TOKENIZER_VOCAB ||
            tokenizer->tokens[token] == NULL) return -1;
        width = tokenizer->lengths[token];
        if (width > capacity - length) return -1;
        memcpy(output + length, tokenizer->tokens[token], width);
        length += width;
    }
    if (length < capacity) output[length] = '\0';
    *output_length = length;
    return 0;
}

int command_tokenizer_save(const command_tokenizer* tokenizer, const char* path)
{
    FILE* file;
    uint32_t count;
    if (tokenizer == NULL || path == NULL) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    count = (uint32_t)(COMMAND_TOKENIZER_BASE + tokenizer->merge_count);
    if (fwrite("CTK1", 1, 4, file) != 4 ||
        fwrite(&count, sizeof(count), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t length = (uint32_t)tokenizer->lengths[index];
        if (fwrite(&length, sizeof(length), 1, file) != 1 ||
            fwrite(tokenizer->tokens[index], 1, length, file) != length) {
            fclose(file);
            return -1;
        }
    }
    fclose(file);
    return 0;
}

int command_tokenizer_load(command_tokenizer* tokenizer, const char* path)
{
    FILE* file;
    char magic[4];
    uint32_t count;
    if (tokenizer == NULL || path == NULL) return -1;
    file = fopen(path, "rb");
    if (file == NULL || fread(magic, 1, 4, file) != 4 ||
        memcmp(magic, "CTK1", 4) != 0 ||
        fread(&count, sizeof(count), 1, file) != 1 ||
        count < COMMAND_TOKENIZER_BASE || count > COMMAND_TOKENIZER_VOCAB) {
        if (file != NULL) fclose(file);
        return -1;
    }
    command_tokenizer_destroy(tokenizer);
    command_tokenizer_init(tokenizer);
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t length;
        char* bytes;
        if (fread(&length, sizeof(length), 1, file) != 1 ||
            length == 0 || length > COMMAND_TOKENIZER_MAX_TOKEN_BYTES) {
            fclose(file);
            command_tokenizer_destroy(tokenizer);
            return -1;
        }
        bytes = (char*)malloc((size_t)length + 1);
        if (bytes == NULL || fread(bytes, 1, length, file) != length) {
            free(bytes);
            fclose(file);
            command_tokenizer_destroy(tokenizer);
            return -1;
        }
        bytes[length] = '\0';
        if (index < COMMAND_TOKENIZER_BASE) {
            free(bytes);
        } else {
            tokenizer->tokens[index] = bytes;
            tokenizer->lengths[index] = length;
            ++tokenizer->merge_count;
        }
    }
    fclose(file);
    return 0;
}
