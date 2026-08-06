#ifndef TENSORLIB_TENSOR_PARALLEL_H
#define TENSORLIB_TENSOR_PARALLEL_H

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Adaptive use of OpenMP: threads are only engaged when the problem is large
 * enough to amortize the fork/join overhead, and only as many threads are used
 * as there is independent work to do. On memory-bandwidth-bound workloads
 * (e.g. large matmuls) a single thread already saturates DRAM, so callers pass
 * an appropriate hard-upper budget via `max_work` (see tensor_matmul.c).
 *
 *   work     : total amount of work (elements or FLOPs).
 *   limit    : only parallelize when work >= limit.
 *   tasks    : number of independent, disjoint chunks (row-blocks, batches...).
 *              Pass a hard cap for tasks to bound how many threads can ever be
 *              used along this axis. 0 means "no cap".
 *
 * Returns the number of threads for an OpenMP `num_threads(...)`, or 1 to run
 * serially (no parallel region is entered, so there is zero overhead).
 *
 * When OpenMP is disabled the function is a constant-folding `1`.
 */
static inline int tensorlib_parallel_threads(long long work,
                                             long long limit_min,
                                             int tasks) {
#ifdef _OPENMP
    if (omp_in_parallel()) return 1;            /* no nested parallelism */
    if (work < limit_min) return 1;             /* not worth forking    */
    int max_threads = omp_get_max_threads();
    if (max_threads <= 1) return 1;
    if (tasks <= 0 || max_threads < tasks) tasks = max_threads;
    return (tasks > 1) ? tasks : 1;
#else
    (void)work; (void)limit_min; (void)tasks;
    return 1;
#endif
}

#endif /* TENSORLIB_TENSOR_PARALLEL_H */