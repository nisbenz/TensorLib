#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tensor.h"

static int make_reduction_dims(const tensor* a,
                               int dim,
                               int keepdim,
                               int* output_ndim,
                               int** output_dims) {
    if (a == NULL || output_ndim == NULL || output_dims == NULL) return 0;

    *output_ndim = keepdim ? a->ndim : a->ndim - 1;
    *output_dims = NULL;
    if (*output_ndim == 0) return 1;

    int* dims = (int*)malloc((size_t)*output_ndim * sizeof(int));
    if (dims == NULL) return 0;

    int output_axis = 0;
    for (int input_axis = 0; input_axis < a->ndim; ++input_axis) {
        if (input_axis == dim) {
            if (keepdim) dims[output_axis++] = 1;
        } else {
            dims[output_axis++] = a->dims[input_axis];
        }
    }

    *output_dims = dims;
    return 1;
}

static tensor* reduce_sum(tensor* a, int dim, int keepdim) {
    if (!tensor_has_valid_metadata(a)) {
        return NULL;
    }

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for sum.\n");
        return NULL;
    }


    int output_ndim = 0;
    int* output_dims = NULL;

    if (!make_reduction_dims(a, dim, keepdim, &output_ndim, &output_dims)) {
        return NULL;
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
                if (keepdim) ++output_axis;
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

tensor* t_sum(tensor* a, int dim) {
    return reduce_sum(a, dim, 0);
}

tensor* t_sum_keepdim(tensor* a, int dim) {
    return reduce_sum(a, dim, 1);
}

static tensor* reduce_mean(tensor* a, int dim, int keepdim) {
    if (!tensor_has_valid_metadata(a)) {
        return NULL;
    }

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for mean.\n");
        return NULL;
    }

    tensor* out = reduce_sum(a, dim, keepdim);
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

tensor* t_mean(tensor* a, int dim) {
    return reduce_mean(a, dim, 0);
}

tensor* t_mean_keepdim(tensor* a, int dim) {
    return reduce_mean(a, dim, 1);
}

static tensor* reduce_max(tensor* a, int dim, int keepdim) {
    if (!tensor_has_valid_metadata(a)) {
        return NULL;
    }

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for max.\n");
        return NULL;
    }

    if (a->dims[dim] <= 0) {
        fprintf(stderr, "ERROR: Cannot compute max over an empty dimension.\n");
        return NULL;
    }

    int output_ndim = 0;
    int* output_dims = NULL;

    if (!make_reduction_dims(a, dim, keepdim, &output_ndim, &output_dims)) {
        return NULL;
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
                if (keepdim) ++output_axis;
            } else {
                input_coords[input_axis] = output_coords[output_axis++];
            }
        }

        float maximum = 0.0f;
        int found_value = 0;

        for (int reduced_index = 0;
             reduced_index < a->dims[dim];
             ++reduced_index) {

            input_coords[dim] = reduced_index;


            int input_index = get_flat_index_nd(a, input_coords);
            float value = a->storage->data[input_index];

            if (isnan(value)) {
                maximum = value;
                break;
            }

            if (!found_value || value > maximum) {
                maximum = value;
                found_value = 1;
            }
        }

        out->storage->data[output_index] = maximum;

        if (output_ndim > 0) {
            advance_coords(output_coords, out->dims, output_ndim);
        }
    }

    free(output_coords);
    free(input_coords);

    return out;
}

tensor* t_max(tensor* a, int dim) {
    return reduce_max(a, dim, 0);
}

tensor* t_max_keepdim(tensor* a, int dim) {
    return reduce_max(a, dim, 1);
}
