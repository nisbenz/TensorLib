#include <stdio.h>
#include <time.h>

#include "../../include/tensorlib/tensor.h"

enum {
    M = 256,
    K = 256,
    N = 256,
    WARMUP = 3,
    REPEATS = 20
};

static void fill_input(tensor* tensor_value, float scale) {
    int elements = tensor_numel(tensor_value);
    for (int index = 0; index < elements; ++index) {
        tensor_value->storage->data[index] =
            scale * (float)((index % 23) - 11);
    }
}

int main(void) {
    int a_base_dims[2] = {K, M};
    int b_base_dims[2] = {N, K};
    tensor* a_base = t_alloc(2, a_base_dims);
    tensor* b_base = t_alloc(2, b_base_dims);
    if (a_base == NULL || b_base == NULL) {
        t_free(b_base);
        t_free(a_base);
        fprintf(stderr, "failed to allocate benchmark inputs\n");
        return 1;
    }

    fill_input(a_base, 0.01f);
    fill_input(b_base, 0.02f);
    tensor* a = t_transpose(a_base, 0, 1);
    tensor* b = t_transpose(b_base, 0, 1);
    if (a == NULL || b == NULL) {
        t_free(b);
        t_free(a);
        t_free(b_base);
        t_free(a_base);
        fprintf(stderr, "failed to create benchmark views\n");
        return 1;
    }

    clock_t pack_start = clock();
    tensor_matmul_packed_rhs* packed_b = t_pack_matmul_rhs(b);
    clock_t pack_end = clock();
    if (packed_b == NULL) {
        t_free(b);
        t_free(a);
        t_free(b_base);
        t_free(a_base);
        fprintf(stderr, "failed to pack benchmark rhs\n");
        return 1;
    }

    for (int repeat = 0; repeat < WARMUP; ++repeat) {
        tensor* output = t_matmul_packed_rhs(a, packed_b);
        if (output == NULL) {
            t_free_matmul_packed_rhs(packed_b);
            t_free(b);
            t_free(a);
            t_free(b_base);
            t_free(a_base);
            fprintf(stderr, "packed matmul failed during warmup\n");
            return 1;
        }
        t_free(output);
    }

    clock_t start = clock();
    volatile float checksum = 0.0f;
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        tensor* output = t_matmul_packed_rhs(a, packed_b);
        if (output == NULL) {
            t_free_matmul_packed_rhs(packed_b);
            t_free(b);
            t_free(a);
            t_free(b_base);
            t_free(a_base);
            fprintf(stderr, "packed matmul failed during benchmark\n");
            return 1;
        }
        checksum += output->storage->data[0];
        checksum += output->storage->data[tensor_numel(output) - 1];
        t_free(output);
    }
    clock_t end = clock();

    double pack_seconds = (double)(pack_end - pack_start) / CLOCKS_PER_SEC;
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    double operations = 2.0 * (double)M * (double)K * (double)N * REPEATS;
    double gflops = (seconds > 0.0) ? operations / seconds / 1.0e9 : 0.0;
    printf("packed transposed views: M=%d K=%d N=%d warmup=%d repeats=%d\n",
           M, K, N, WARMUP, REPEATS);
    printf("pack_seconds=%.6f elapsed_seconds=%.6f gflops=%.6f checksum=%.6f\n",
           pack_seconds, seconds, gflops, (double)checksum);

    t_free_matmul_packed_rhs(packed_b);
    t_free(b);
    t_free(a);
    t_free(b_base);
    t_free(a_base);
    return 0;
}
