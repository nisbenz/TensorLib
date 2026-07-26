#include <math.h>
#include <stdio.h>

#include <autograd.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    } \
} while (0)

static ag_tensor* make_table(void)
{
    int dims[] = {3, 2};
    tensor* raw = t_alloc(2, dims);

    if (raw == NULL) return NULL;
    for (int i = 0; i < 6; ++i) raw->storage->data[i] = (float)(i + 1);
    return ag_from_owned_tensor(raw, 1);
}

static tensor* make_indices(void)
{
    int dims[] = {3};
    tensor* result = t_alloc(1, dims);

    if (result == NULL) return NULL;
    result->storage->data[0] = 2.0f;
    result->storage->data[1] = 0.0f;
    result->storage->data[2] = 2.0f;
    return result;
}

static void test_forward_and_repeated_backward(void)
{
    ag_tensor* table = make_table();
    tensor* indices = make_indices();
    ag_tensor* gathered = ag_gather_rows(table, indices);
    ag_tensor* loss;

    CHECK(gathered != NULL);
    if (gathered != NULL) {
        float expected[] = {5, 6, 1, 2, 5, 6};
        CHECK(gathered->value->ndim == 2);
        CHECK(gathered->value->dims[0] == 3);
        CHECK(gathered->value->dims[1] == 2);
        for (int i = 0; i < 6; ++i) {
            CHECK(gathered->value->storage->data[i] == expected[i]);
        }
    }
    loss = gathered == NULL ? NULL : ag_sum(gathered, 0, 0);
    if (loss != NULL) {
        ag_tensor* scalar = ag_sum(loss, 0, 0);
        ag_tensor_release(loss);
        loss = scalar;
    }
    CHECK(loss != NULL && ag_backward(loss) == 0);
    if (table != NULL && table->grad != NULL) {
        float expected_grad[] = {1, 1, 0, 0, 2, 2};
        for (int i = 0; i < 6; ++i) {
            CHECK(table->grad->storage->data[i] == expected_grad[i]);
        }
    } else {
        CHECK(0);
    }

    ag_tensor_release(loss);
    ag_tensor_release(gathered);
    t_free(indices);
    ag_tensor_release(table);
}

static void test_invalid_indices(void)
{
    ag_tensor* table = make_table();
    int dims[] = {1};
    tensor* indices = t_alloc(1, dims);

    indices->storage->data[0] = 1.5f;
    CHECK(ag_gather_rows(table, indices) == NULL);
    indices->storage->data[0] = -1.0f;
    CHECK(ag_gather_rows(table, indices) == NULL);
    indices->storage->data[0] = 3.0f;
    CHECK(ag_gather_rows(table, indices) == NULL);
    indices->storage->data[0] = NAN;
    CHECK(ag_gather_rows(table, indices) == NULL);
    CHECK(ag_gather_rows(NULL, indices) == NULL);

    t_free(indices);
    ag_tensor_release(table);
}

int main(void)
{
    test_forward_and_repeated_backward();
    test_invalid_indices();
    if (failures != 0) {
        fprintf(stderr, "%d gather checks failed\n", failures);
        return 1;
    }
    printf("All autograd gather checks passed.\n");
    return 0;
}
