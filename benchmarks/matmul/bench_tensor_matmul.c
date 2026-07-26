#include <stdio.h>
#include <time.h>
#include <stdint.h>

#include "../../include/tensorlib/tensor.h"

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
    SMALL_BATCHES = 8,
    SMALL_M = 128,
    SMALL_K = 128,
    SMALL_N = 128,
    LARGE_M = 4096,
    LARGE_K = 4096,
    LARGE_N = 4096,
    WARMUP = 3,
    REPEATS = 20
};

static void fill_input(tensor* t, float scale) {
    int elements = tensor_numel(t);
    for (int i = 0; i < elements; ++i) {
        t->storage->data[i] = scale * (float)((i % 23) - 11);
    }
}

static int run_benchmark(const char* label, int batch, int m, int k, int n,
                         int warmup, int repeats) {
    int a_ndim = (batch > 1) ? 3 : 2;
    int a_dims[3] = {batch > 1 ? batch : 0, m, k};
    int b_ndim = (batch > 1) ? 3 : 2;
    int b_dims[3] = {batch > 1 ? batch : 0, k, n};

    tensor* a = t_alloc(a_ndim, a_ndim == 3 ? a_dims : a_dims + 1);
    tensor* b = t_alloc(b_ndim, b_ndim == 3 ? b_dims : b_dims + 1);
    if (a == NULL || b == NULL) {
        t_free(b); t_free(a);
        fprintf(stderr, "failed to allocate\n");
        return 0;
    }

    fill_input(a, 0.01f);
    fill_input(b, 0.02f);

    for (int r = 0; r < warmup; ++r) {
        tensor* output = t_matmul(a, b);
        t_free(output);
    }

    clock_t start = clock();
    volatile float checksum = 0.0f;
    for (int r = 0; r < repeats; ++r) {
        tensor* output = t_matmul(a, b);
        if (output == NULL) {
            t_free(b); t_free(a);
            fprintf(stderr, "matmul failed\n");
            return 0;
        }
        checksum += output->storage->data[0];
        checksum += output->storage->data[tensor_numel(output) - 1];
        t_free(output);
    }
    clock_t end = clock();

    double seconds = (double)(end - start) / (double)CLOCKS_PER_SEC;
    double ops = 2.0 * (double)batch * (double)m * (double)k * (double)n * (double)repeats;
    double gflops = (seconds > 0.0) ? ops / seconds / 1.0e9 : 0.0;

#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif

    printf("%s: batch=%d M=%d K=%d N=%d threads=%d warmup=%d repeats=%d\n",
           label, batch, m, k, n, threads, warmup, repeats);
    printf("  elapsed_seconds=%.6f gflops=%.6f checksum=%.6f\n",
           seconds, gflops, (double)checksum);

    t_free(b);
    t_free(a);
    return 1;
}

int main(void) {
    printf("=== TensorLib Matmul Benchmarks ===\n\n");

    /* Small batched matmul (existing baseline) */
    run_benchmark("small_batched", SMALL_BATCHES, SMALL_M, SMALL_K, SMALL_N,
                  WARMUP, 50);

    /* Single large matmul to show OpenMP scaling */
    run_benchmark("large_single", 1, LARGE_M, LARGE_K, LARGE_N,
                  WARMUP, REPEATS);

    printf("\n");
    return 0;
}
