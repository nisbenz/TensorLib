#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/tensor.h"

typedef struct {
    int is_vector;
    int batch_rank;
    int rows;
    int inner;
    int columns;
} matmul_operand_info;

static int operand_batch_dim(const tensor* operand,
                             const matmul_operand_info* info,
                             int output_axis,
                             int output_batch_ndim) {
    int rank_offset = output_batch_ndim - info->batch_rank;
    if (output_axis < rank_offset) return 1;
    return operand->dims[output_axis - rank_offset];
}

static int operand_batch_offset(const tensor* operand,
                                const matmul_operand_info* info,
                                const int* batch_coords,
                                int batch_ndim) {
    int offset = operand->offset;

    for (int operand_axis = 0; operand_axis < info->batch_rank; ++operand_axis) {
        int output_axis = batch_ndim - info->batch_rank + operand_axis;
        int coordinate = batch_coords[output_axis];

        /* A singleton batch axis is broadcast by reusing coordinate zero. */
        if (operand->dims[operand_axis] == 1) coordinate = 0;
        offset += coordinate * operand->strides[operand_axis];
    }

    return offset;
}

static int output_batch_offset(const tensor* output,
                               const int* batch_coords,
                               int batch_ndim) {
    int offset = output->offset;
    for (int axis = 0; axis < batch_ndim; ++axis) {
        offset += batch_coords[axis] * output->strides[axis];
    }
    return offset;
}

static int broadcast_batch_shape(const tensor* a,
                                 const matmul_operand_info* a_info,
                                 const tensor* b,
                                 const matmul_operand_info* b_info,
                                 int batch_ndim,
                                 int* batch_dims) {
    for (int axis = 0; axis < batch_ndim; ++axis) {
        int a_dim = operand_batch_dim(a, a_info, axis, batch_ndim);
        int b_dim = operand_batch_dim(b, b_info, axis, batch_ndim);

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) return 0;
        batch_dims[axis] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    return 1;
}

static void matmul_2d_strided(const tensor* a,
                              const matmul_operand_info* a_info,
                              int a_base,
                              const tensor* b,
                              const matmul_operand_info* b_info,
                              int b_base,
                              tensor* output,
                              int output_base,
                              int output_batch_ndim) {
    const int a_row_stride = a_info->is_vector ? 0 : a->strides[a->ndim - 2];
    const int a_inner_stride = a->strides[a->ndim - 1];
    const int b_inner_stride = b->strides[b_info->is_vector ? 0 : b->ndim - 2];
    const int b_column_stride = b_info->is_vector ? 0 : b->strides[b->ndim - 1];

    int output_row_stride = 0;
    int output_column_stride = 0;
    if (!a_info->is_vector && !b_info->is_vector) {
        output_row_stride = output->strides[output_batch_ndim];
        output_column_stride = output->strides[output_batch_ndim + 1];
    } else if (a_info->is_vector) {
        /* The promoted leading row dimension is removed from the result. */
        if (!b_info->is_vector) {
            output_column_stride = output->strides[output_batch_ndim];
        }
    } else {
        /* The promoted trailing column dimension is removed from the result. */
        output_row_stride = output->strides[output_batch_ndim];
    }

    for (int row = 0; row < a_info->rows; ++row) {
        for (int column = 0; column < b_info->columns; ++column) {
            float sum = 0.0f;

            for (int inner = 0; inner < a_info->inner; ++inner) {
                int a_index = a_base + row * a_row_stride + inner * a_inner_stride;
                int b_index = b_base + inner * b_inner_stride + column * b_column_stride;
                sum += a->storage->data[a_index] * b->storage->data[b_index];
            }

            int output_index = output_base + row * output_row_stride + column * output_column_stride;
            output->storage->data[output_index] = sum;
        }
    }
}

tensor* t_matmul(tensor* a, tensor* b) {
    if (!tensor_has_valid_metadata(a) || !tensor_has_valid_metadata(b)) {
        return NULL;
    }

    /* Scalars are not matrix operands. Vectors are supported through promotion. */
    if (a->ndim == 0 || b->ndim == 0) {
        fprintf(stderr, "ERROR: Matmul operands must have at least one dimension.\n");
        return NULL;
    }

    matmul_operand_info a_info = {
        .is_vector = (a->ndim == 1),
        .batch_rank = (a->ndim > 1) ? a->ndim - 2 : 0,
        .rows = (a->ndim == 1) ? 1 : a->dims[a->ndim - 2],
        .inner = a->dims[a->ndim - 1],
        .columns = 0
    };
    matmul_operand_info b_info = {
        .is_vector = (b->ndim == 1),
        .batch_rank = (b->ndim > 1) ? b->ndim - 2 : 0,
        .rows = 0,
        .inner = (b->ndim == 1) ? b->dims[0] : b->dims[b->ndim - 2],
        .columns = (b->ndim == 1) ? 1 : b->dims[b->ndim - 1]
    };

    if (a_info.inner != b_info.inner) {
        fprintf(stderr, "ERROR: Matmul inner dimensions do not match.\n");
        return NULL;
    }

    if (a_info.batch_rank > INT_MAX - 2 || b_info.batch_rank > INT_MAX - 2) {
        return NULL;
    }

    int batch_ndim = (a_info.batch_rank > b_info.batch_rank)
                   ? a_info.batch_rank
                   : b_info.batch_rank;
    int* batch_dims = NULL;
    if (batch_ndim > 0) {
        batch_dims = (int*)malloc((size_t)batch_ndim * sizeof(int));
        if (batch_dims == NULL) return NULL;
    }

    if (!broadcast_batch_shape(a, &a_info, b, &b_info, batch_ndim, batch_dims)) {
        fprintf(stderr, "ERROR: Matmul batch dimensions are not broadcast-compatible.\n");
        free(batch_dims);
        return NULL;
    }

    int output_ndim = batch_ndim + 2 - a_info.is_vector - b_info.is_vector;
    int* output_dims = NULL;
    if (output_ndim > 0) {
        output_dims = (int*)malloc((size_t)output_ndim * sizeof(int));
        if (output_dims == NULL) {
            free(batch_dims);
            return NULL;
        }

        for (int axis = 0; axis < batch_ndim; ++axis) {
            output_dims[axis] = batch_dims[axis];
        }

        int output_axis = batch_ndim;
        if (!a_info.is_vector) output_dims[output_axis++] = a_info.rows;
        if (!b_info.is_vector) output_dims[output_axis] = b_info.columns;
    }

    tensor* output = t_alloc(output_ndim, output_dims);
    free(output_dims);
    if (output == NULL) {
        free(batch_dims);
        return NULL;
    }

    size_t batch_count;
    if (!tensor_checked_numel(batch_ndim, batch_dims, &batch_count)) {
        free(batch_dims);
        t_free(output);
        return NULL;
    }

    int* batch_coords = NULL;
    if (batch_ndim > 0) {
        batch_coords = (int*)calloc((size_t)batch_ndim, sizeof(int));
        if (batch_coords == NULL) {
            free(batch_dims);
            t_free(output);
            return NULL;
        }
    }

    for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        int a_base = operand_batch_offset(a, &a_info, batch_coords, batch_ndim);
        int b_base = operand_batch_offset(b, &b_info, batch_coords, batch_ndim);
        int output_base = output_batch_offset(output, batch_coords, batch_ndim);

        matmul_2d_strided(a, &a_info, a_base,
                          b, &b_info, b_base,
                          output, output_base, batch_ndim);

        if (batch_ndim > 0) {
            advance_coords(batch_coords, batch_dims, batch_ndim);
        }
    }

    free(batch_coords);
    free(batch_dims);
    return output;
}
