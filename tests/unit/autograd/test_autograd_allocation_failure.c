#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../../include/tensorlib/autograd.h"
#include "../../../src/tensor/tensor_alloc_internal.h"

void* __real_malloc(size_t bytes);
void* __real_calloc(size_t count, size_t bytes);
void* __real_realloc(void* pointer, size_t bytes);
static int remaining = -1;
static int failures;

static int fail_now(void) {
    if (remaining < 0) return 0;
    if (remaining-- != 0) return 0;
    remaining = -1;
    ++failures;
    return 1;
}

void* __wrap_malloc(size_t bytes) {
    return fail_now() ? NULL : __real_malloc(bytes);
}
void* __wrap_calloc(size_t count, size_t bytes) {
    return fail_now() ? NULL : __real_calloc(count, bytes);
}
void* __wrap_realloc(void* pointer, size_t bytes) {
    return fail_now() ? NULL : __real_realloc(pointer, bytes);
}

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "allocation rollback check failed at line %d\n", __LINE__); \
    return 1; } } while (0)

int main(void) {
    int dims[2] = {3, 3};
    tensor* raw = t_alloc(2, dims);
    CHECK(raw != NULL);
    for (int i = 0; i < 9; ++i) raw->storage->data[i] = (float)(i + 1) * 0.1f;
    ag_tensor* input = ag_from_owned_tensor(raw, 1);
    ag_tensor* transpose = ag_transpose(input, 0, 1);
    ag_tensor* product = ag_matmul(input, transpose);
    ag_tensor* rows = ag_sum(product, 1, 0);
    ag_tensor* loss = ag_sum(rows, 0, 0);
    ag_tensor* values[] = {input, transpose, product, rows, loss};
    CHECK(loss != NULL && ag_backward(loss) == 0);
    tensor* single_step[5];
    for (int i = 0; i < 5; ++i) {
        single_step[i] = t_clone(values[i]->grad);
        CHECK(single_step[i] != NULL);
    }
    int failed_passes = 0, completed = 0;
    for (int budget = 0; budget < 512; ++budget) {
        tensor* snapshots[5];
        for (int i = 0; i < 5; ++i) {
            snapshots[i] = t_clone(values[i]->grad);
            CHECK(snapshots[i] != NULL);
        }
        failures = 0;
        remaining = budget;
        int status = ag_backward(loss);
        remaining = -1;
        if (status != 0) {
            ++failed_passes;
            for (int i = 0; i < 5; ++i) {
                CHECK(values[i]->grad != NULL && values[i]->graph_index == -1);
                for (int j = 0; j < tensor_numel(snapshots[i]); ++j)
                    CHECK(values[i]->grad->storage->data[tensor_flat_index(values[i]->grad, j)] ==
                          snapshots[i]->storage->data[j]);
            }
        } else {
            for (int i = 0; i < 5; ++i) {
                CHECK(values[i]->grad != NULL && values[i]->graph_index == -1);
                for (int j = 0; j < tensor_numel(snapshots[i]); ++j) {
                    float expected = snapshots[i]->storage->data[j] + single_step[i]->storage->data[j];
                    CHECK(fabsf(values[i]->grad->storage->data[tensor_flat_index(values[i]->grad, j)] -
                                expected) < 1e-5f);
                }
            }
        }
        for (int i = 0; i < 5; ++i) t_free(snapshots[i]);
        if (failures == 0) { completed = status == 0; break; }
    }
    CHECK(failed_passes > 0 && completed);
    for (int i = 4; i >= 0; --i) {
        t_free(single_step[i]);
        ag_tensor_release(values[i]);
    }
    printf("allocation rollback: %d failing passes validated\n", failed_passes);
    return 0;
}
