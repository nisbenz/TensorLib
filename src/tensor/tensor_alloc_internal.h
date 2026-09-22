#ifndef TENSORLIB_TENSOR_ALLOC_INTERNAL_H
#define TENSORLIB_TENSOR_ALLOC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "../../include/tensorlib/tensor.h"

typedef struct {
    uint64_t allocations;
    uint64_t frees;
    size_t allocated_bytes;
    size_t live_bytes;
    size_t peak_live_bytes;
} tensor_alloc_stats;

void tensor_alloc_stats_enable(int enabled);
void tensor_alloc_stats_reset(void);
void tensor_alloc_stats_reset_counters(void);
void tensor_alloc_stats_read(tensor_alloc_stats* result);

/* Shared private indexing helpers used by tensor, autograd, and NN code. */
int tensor_flat_index(const tensor* value, int flat);
int tensor_row_base(const tensor* value, int row, int width);

#endif
