#ifndef TENSORLIB_TENSOR_ALLOC_INTERNAL_H
#define TENSORLIB_TENSOR_ALLOC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "../../include/tensorlib/tensor.h"

enum { TENSOR_ALLOC_METADATA, TENSOR_ALLOC_GRAPH, TENSOR_ALLOC_MATMUL,
       TENSOR_ALLOC_AUX_KINDS };

typedef struct {
    uint64_t allocations;
    uint64_t frees;
    size_t allocated_bytes;
    size_t live_bytes;
    size_t peak_live_bytes;
    uint64_t auxiliary_allocations[TENSOR_ALLOC_AUX_KINDS];
    size_t auxiliary_bytes[TENSOR_ALLOC_AUX_KINDS];
    size_t copied_bytes;
} tensor_alloc_stats;

void tensor_alloc_stats_enable(int enabled);
void tensor_alloc_stats_reset(void);
void tensor_alloc_stats_reset_counters(void);
void tensor_alloc_stats_read(tensor_alloc_stats* result);
void tensor_alloc_record_auxiliary(int kind, size_t bytes);
void tensor_alloc_record_copy(size_t bytes);
void* tensor_profile_malloc(size_t bytes, int kind);
void* tensor_profile_calloc(size_t count, size_t bytes, int kind);
void* tensor_profile_realloc(void* pointer, size_t bytes, int kind);

/* Shared private indexing helpers used by tensor, autograd, and NN code. */
int tensor_flat_index(const tensor* value, int flat);
int tensor_row_base(const tensor* value, int row, int width);

#endif
