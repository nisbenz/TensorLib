#include <stdio.h>
#include <time.h>

#include "../include/tensor.h"
#include "../include/tensor_matmul.h"

enum {
    BATCHES = 8,
    M = 128,
    K = 128,
    N = 128,
    WARMUP = 3,
    REPEATS = 50
};

static void fill_input(tensor* t, float scale) {
    int elements = tensor_numel(t);
    for (int i = 0; i < elements; ++i) {
        t->storage->data[i] = scale * (float)((i % 23) - 11);
    }
}

int main(void) {
    int a_dims[3] = {BATCHES, M, K};
    int b_dims[3] = {BATCHES, K, N};
    tensor* a = t_alloc(3, a_dims);
    tensor* b = t_alloc(3, b_dims);
    if (a == NULL || b == NULL) {
        t_free(b);
        t_free(a);
        fprintf(stderr, "failed to allocate benchmark inputs\n");
        return 1;
    }

    fill_input(a, 0.01f);
    fill_input(b, 0.02f);

    int output_dims[3] = {BATCHES, M, N};
    tensor* output = t_alloc(3, output_dims);
    if (output == NULL) {
        t_free(b);
        t_free(a);
        fprintf(stderr, "failed to allocate benchmark output\n");
        return 1;
    }

    int use_avx2 = matmul_avx2_available();
    void (*kernel)(const float*, const float*, float*, int, int, int) =
        use_avx2 ? matmul_2d_avx2_contiguous : matmul_2d_blocked_contiguous;

    for (int repeat = 0; repeat < WARMUP; ++repeat) {
        for (int batch = 0; batch < BATCHES; ++batch) {
            kernel(a->storage->data + batch * M * K,
                   b->storage->data + batch * K * N,
                   output->storage->data + batch * M * N,
                   M, K, N);
        }
    }

    clock_t start = clock();
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        for (int batch = 0; batch < BATCHES; ++batch) {
            kernel(a->storage->data + batch * M * K,
                   b->storage->data + batch * K * N,
                   output->storage->data + batch * M * N,
                   M, K, N);
        }
    }
    clock_t end = clock();

    volatile float checksum = output->storage->data[0];
    checksum += output->storage->data[tensor_numel(output) - 1];

    double seconds = (double)(end - start) / (double)CLOCKS_PER_SEC;
    double operations = 2.0 * (double)BATCHES * (double)M * (double)K * (double)N * (double)REPEATS;
    double gflops = (seconds > 0.0) ? operations / seconds / 1.0e9 : 0.0;

    printf("matmul kernel: batch=%d M=%d K=%d N=%d warmup=%d repeats=%d path=%s\n",
           BATCHES, M, K, N, WARMUP, REPEATS,
           use_avx2 ? "avx2-fma" : "blocked-scalar");
    printf("elapsed_seconds=%.6f gflops=%.6f checksum=%.6f\n",
           seconds, gflops, (double)checksum);

    t_free(output);
    t_free(b);
    t_free(a);
    return 0;
}
