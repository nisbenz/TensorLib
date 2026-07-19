#include <stdio.h>
#include <time.h>

#include <cblas.h>

#include "../include/tensor.h"

/* Keep this comparison single-threaded, like bench_tensor_matmul. */
extern void openblas_set_num_threads(int num_threads);

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

static void openblas_batch_matmul(const tensor* a, const tensor* b, tensor* output) {
    const int matrix_size_a = M * K;
    const int matrix_size_b = K * N;
    const int matrix_size_output = M * N;

    for (int batch = 0; batch < BATCHES; ++batch) {
        const float* a_data = a->storage->data + batch * matrix_size_a;
        const float* b_data = b->storage->data + batch * matrix_size_b;
        float* output_data = output->storage->data + batch * matrix_size_output;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f, a_data, K,
                    b_data, N,
                    0.0f, output_data, N);
    }
}

int main(void) {
    openblas_set_num_threads(1);

    int a_dims[3] = {BATCHES, M, K};
    int b_dims[3] = {BATCHES, K, N};
    int output_dims[3] = {BATCHES, M, N};
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

    for (int repeat = 0; repeat < WARMUP; ++repeat) {
        tensor* output = t_alloc(3, output_dims);
        if (output == NULL) {
            t_free(b);
            t_free(a);
            fprintf(stderr, "failed to allocate benchmark output\n");
            return 1;
        }
        openblas_batch_matmul(a, b, output);
        t_free(output);
    }

    clock_t start = clock();
    volatile float checksum = 0.0f;
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        tensor* output = t_alloc(3, output_dims);
        if (output == NULL) {
            t_free(b);
            t_free(a);
            fprintf(stderr, "failed to allocate benchmark output\n");
            return 1;
        }
        openblas_batch_matmul(a, b, output);
        checksum += output->storage->data[0];
        checksum += output->storage->data[tensor_numel(output) - 1];
        t_free(output);
    }
    clock_t end = clock();

    double seconds = (double)(end - start) / (double)CLOCKS_PER_SEC;
    double operations = 2.0 * (double)BATCHES * (double)M * (double)K *
                        (double)N * (double)REPEATS;
    double gflops = (seconds > 0.0) ? operations / seconds / 1.0e9 : 0.0;

    printf("OpenBLAS: batch=%d M=%d K=%d N=%d warmup=%d repeats=%d\n",
           BATCHES, M, K, N, WARMUP, REPEATS);
    printf("elapsed_seconds=%.6f gflops=%.6f checksum=%.6f\n",
           seconds, gflops, (double)checksum);

    t_free(b);
    t_free(a);
    return 0;
}
