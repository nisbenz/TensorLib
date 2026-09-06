#ifndef TENSORLIB_TENSOR_ALLOC_INTERNAL_H
#define TENSORLIB_TENSOR_ALLOC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

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

#endif
