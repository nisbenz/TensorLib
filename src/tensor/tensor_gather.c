#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include "../../include/tensorlib/tensor.h"

static int tensor_flat_index(const tensor* value, int flat)
{
    int index = value->offset;
    int remaining = flat;

    for (int dim = value->ndim - 1; dim >= 0; --dim) {
        int coordinate = remaining % value->dims[dim];
        remaining /= value->dims[dim];
        index += coordinate * value->strides[dim];
    }
    return index;
}

tensor* t_gather_rows(tensor* table, tensor* indices)
{
    tensor* result;
    int* result_dims;
    int index_count;
    int width;

    if (!tensor_has_valid_metadata(table) ||
        !tensor_has_valid_metadata(indices) ||
        table->ndim != 2 || indices->ndim == INT_MAX) {
        return NULL;
    }
    result_dims = (int*)malloc(
        (size_t)(indices->ndim + 1) * sizeof(*result_dims));
    if (result_dims == NULL) return NULL;
    for (int dim = 0; dim < indices->ndim; ++dim) {
        result_dims[dim] = indices->dims[dim];
    }
    width = table->dims[1];
    result_dims[indices->ndim] = width;
    result = t_alloc(indices->ndim + 1, result_dims);
    free(result_dims);
    if (result == NULL) return NULL;

    index_count = tensor_numel(indices);
    for (int item = 0; item < index_count; ++item) {
        float raw_index =
            indices->storage->data[tensor_flat_index(indices, item)];
        int row;

        if (!isfinite(raw_index) || floorf(raw_index) != raw_index ||
            raw_index < 0.0f || raw_index >= (float)table->dims[0]) {
            t_free(result);
            return NULL;
        }
        row = (int)raw_index;
        for (int column = 0; column < width; ++column) {
            int table_index =
                table->offset + row * table->strides[0] +
                column * table->strides[1];
            result->storage->data[item * width + column] =
                table->storage->data[table_index];
        }
    }
    return result;
}
