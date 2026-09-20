#include "command_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* text;
    size_t length;
    uint32_t count;
} candidate;

static char* copy_bytes(const unsigned char* data, size_t length)
{
    char* result = (char*)malloc(length + 1);
    if (result == NULL) return NULL;
    memcpy(result, data, length);
    result[length] = '\0';
    return result;
}

void command_tokenizer_init(command_tokenizer* tokenizer)
{
    if (tokenizer == NULL) return;
    memset(tokenizer, 0, sizeof(*tokenizer));
    for (size_t index = 0; index < COMMAND_TOKENIZER_BASE; ++index) {
        unsigned char value = (unsigned char)index;
        tokenizer->tokens[index] = copy_bytes(&value, 1);
        tokenizer->lengths[index] = 1;
    }
}

void command_tokenizer_destroy(command_tokenizer* tokenizer)
{
    if (tokenizer == NULL) return;
    for (size_t index = 0; index < COMMAND_TOKENIZER_VOCAB; ++index) {
        free(tokenizer->tokens[index]);
        tokenizer->tokens[index] = NULL;
        tokenizer->lengths[index] = 0;
    }
    tokenizer->merge_count = 0;
}

static unsigned long hash_bytes(const unsigned char* data, size_t length)
{
    unsigned long hash = 2166136261u;
    for (size_t index = 0; index < length; ++index) {
        hash ^= data[index];
        hash *= 16777619u;
    }
    return hash;
}

static int candidate_add(candidate* items, size_t* count, size_t capacity,
                         int* slots, size_t slot_count,
                         const unsigned char* data, size_t length)
{
    size_t slot = (size_t)(hash_bytes(data, length) % slot_count);
    for (size_t probe = 0; probe < slot_count; ++probe) {
        int index = slots[slot];
        if (index < 0) break;
        if (items[index].length == length &&
            memcmp(items[index].text, data, length) == 0) {
            if (items[index].count != UINT32_MAX) ++items[index].count;
            return 0;
        }
        slot = (slot + 1) % slot_count;
    }
    if (*count == capacity) return 0;
    items[*count].text = copy_bytes(data, length);
    if (items[*count].text == NULL) return -1;
    items[*count].length = length;
    items[*count].count = 1;
    slots[slot] = (int)*count;
    ++*count;
    return 0;
}

static int candidate_compare(const void* left, const void* right)
{
    const candidate* a = (const candidate*)left;
    const candidate* b = (const candidate*)right;
    if (a->count != b->count) return a->count < b->count ? 1 : -1;
    if (a->length != b->length) return a->length < b->length ? 1 : -1;
    return memcmp(a->text, b->text, a->length);
}

int command_tokenizer_train(command_tokenizer* tokenizer,
                            const unsigned char* text, size_t length,
                            size_t target_vocab)
{
    const size_t candidate_capacity = 4096;
    const size_t slot_count = 8192;
    candidate* candidates;
    int* slots;
    size_t candidate_count = 0;
    if (tokenizer == NULL || text == NULL || length == 0 ||
        target_vocab < COMMAND_TOKENIZER_BASE ||
        target_vocab > COMMAND_TOKENIZER_VOCAB) return -1;

    command_tokenizer_destroy(tokenizer);
    command_tokenizer_init(tokenizer);
    candidates = (candidate*)calloc(candidate_capacity, sizeof(*candidates));
    slots = (int*)malloc(slot_count * sizeof(*slots));
    if (candidates == NULL || slots == NULL) {
        free(candidates);
        free(slots);
        return -1;
    }
    for (size_t index = 0; index < slot_count; ++index) slots[index] = -1;

    for (size_t start = 0; start < length; ++start) {
        size_t maximum = length - start;
        if (maximum > COMMAND_TOKENIZER_MAX_TOKEN_BYTES) {
            maximum = COMMAND_TOKENIZER_MAX_TOKEN_BYTES;
        }
        for (size_t width = 2; width <= maximum; ++width) {
            int printable = 1;
            for (size_t offset = 0; offset < width; ++offset) {
                if (text[start + offset] < 9 || text[start + offset] > 126) {
                    printable = 0;
                    break;
                }
            }
            if (printable && candidate_add(candidates, &candidate_count,
                                           candidate_capacity, slots, slot_count,
                                           text + start, width) != 0) {
                goto fail;
            }
        }
    }
    free(slots);

    qsort(candidates, candidate_count, sizeof(*candidates), candidate_compare);
    for (size_t index = 0; index < candidate_count &&
             COMMAND_TOKENIZER_BASE + tokenizer->merge_count < target_vocab;
         ++index) {
        size_t token = COMMAND_TOKENIZER_BASE + tokenizer->merge_count;
        tokenizer->tokens[token] = candidates[index].text;
        tokenizer->lengths[token] = candidates[index].length;
        candidates[index].text = NULL;
        tokenizer->merges[token - COMMAND_TOKENIZER_BASE].count =
            candidates[index].count;
        ++tokenizer->merge_count;
    }
    for (size_t index = 0; index < candidate_count; ++index) {
        free(candidates[index].text);
    }
    free(candidates);
    return 0;

fail:
    for (size_t index = 0; index < candidate_count; ++index) {
        free(candidates[index].text);
    }
    free(candidates);
    free(slots);
    return -1;
}
