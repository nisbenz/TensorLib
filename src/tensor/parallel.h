#ifndef TENSORLIB_TENSOR_PARALLEL_H
#define TENSORLIB_TENSOR_PARALLEL_H

#include <stdio.h>

typedef enum {
    TENSORLIB_PARALLEL_MEMORY,
    TENSORLIB_PARALLEL_COMPUTE,
    TENSORLIB_PARALLEL_MATMUL,
    TENSORLIB_PARALLEL_CLONE,
    TENSORLIB_PARALLEL_PACK_RHS,
    TENSORLIB_PARALLEL_GELU,
    TENSORLIB_PARALLEL_GELU_BACKWARD,
    TENSORLIB_PARALLEL_PACKED_MATMUL,
    TENSORLIB_PARALLEL_BATCHED_MATMUL,
    TENSORLIB_PARALLEL_ADAMW,
    TENSORLIB_PARALLEL_COUNT
} tensorlib_parallel_kind;

/* Resolve a cached ENV policy against the caller's work threshold and task cap.
 * A thread count of one, a small workload, or nested parallel work runs serially.
 * Unset fields retain the caller threshold and current OpenMP maximum.
 */
int tensorlib_parallel_threads_for(tensorlib_parallel_kind kind,
                                  long long work, long long default_limit,
                                  int tasks);
void tensorlib_parallel_report(FILE* stream);

#ifdef _OPENMP
#include <omp.h>
#endif


#endif /* TENSORLIB_TENSOR_PARALLEL_H */
