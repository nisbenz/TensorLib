#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
tensor* t_sum(tensor* a, int dim) {
    if (!tensor_has_valid_metadata(a)) {
        return NULL;
    }

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for sum.\n");
        return NULL;
    }


    int output_ndim = a->ndim - 1;
    int* output_dims = NULL;

    if (output_ndim > 0) {
        output_dims = malloc((size_t)output_ndim * sizeof(int));
        if (output_dims == NULL) {
            return NULL;
        }

        for (int input_axis = 0, output_axis = 0;
             input_axis < a->ndim;
             ++input_axis) {
            if (input_axis != dim) {
                output_dims[output_axis++] = a->dims[input_axis];
            }
        }
    }


    tensor* out = t_alloc(output_ndim, output_dims);
    free(output_dims);

    if (out == NULL) {
        return NULL;
    }

    int* output_coords = NULL;

    if (output_ndim > 0) {
        output_coords = calloc((size_t)output_ndim, sizeof(int));
        if (output_coords == NULL) {
            t_free(out);
            return NULL;
        }
    }
    int* input_coords = calloc((size_t)a->ndim, sizeof(int));
    if (input_coords == NULL) {
        free(output_coords);
        t_free(out);
        return NULL;
    }

    int output_elements = tensor_numel(out);

    for (int output_index = 0;
         output_index < output_elements;
         ++output_index) {


        for (int input_axis = 0, output_axis = 0;
             input_axis < a->ndim;
             ++input_axis) {

            if (input_axis == dim) {
                input_coords[input_axis] = 0;
            } else {
                input_coords[input_axis] =
                    output_coords[output_axis++];
            }
        }

        float sum = 0.0f;


        for (int reduced_index = 0;
             reduced_index < a->dims[dim];
             ++reduced_index) {

            input_coords[dim] = reduced_index;

            int input_index =
                get_flat_index_nd(a, input_coords);

            sum += a->storage->data[input_index];
        }

        out->storage->data[output_index] = sum;
        if (output_ndim > 0) {
            advance_coords(output_coords, out->dims, output_ndim);
        }
    }

    free(output_coords);
    free(input_coords);
    return out;
}

tensor* t_mean(tensor* a, int dim) {
    if (!tensor_has_valid_metadata(a)) {
        return NULL;
    }

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for mean.\n");
        return NULL;
    }

    tensor* out = t_sum(a, dim);
    if (out == NULL) {
        return NULL;
    }

    const float reduction_size = (float)a->dims[dim];
    const int output_elements = tensor_numel(out);
    for (int i = 0; i < output_elements; ++i) {
        out->storage->data[i] /= reduction_size;
    }

    return out;
}
