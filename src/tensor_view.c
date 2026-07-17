#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"

static tensor* make_view(tensor* base, int ndim, const int* dims, const int* strides, int offset) {
    size_t unused;
    if (!tensor_has_valid_metadata(base)) return NULL;
    if (!tensor_checked_numel(ndim, dims, &unused)) return NULL;
    if (ndim > 0 && strides == NULL) return NULL;

    tensor* view = (tensor*)calloc(1, sizeof(tensor));
    if (view == NULL) return NULL;

    view->ndim = ndim;
    view->offset = offset;

    if (tensor_copy_metadata(ndim, dims, strides, &view->dims, &view->strides) != 0) {
        t_free(view);
        return NULL;
    }

    add_ref_count(base->storage, view);
    if (!tensor_has_valid_metadata(view)) {
        t_free(view);
        return NULL;
    }
    return view;
}

tensor* t_transpose(tensor* a, int dim0, int dim1) {
    if (!tensor_has_valid_metadata(a)) return NULL;
    if (dim0 < 0 || dim0 >= a->ndim || dim1 < 0 || dim1 >= a->ndim) {
        fprintf(stderr, "ERROR: Transpose dimensions out of bounds.\n");
        return NULL;
    }

    int* dims = NULL;
    int* strides = NULL;
    if (tensor_copy_metadata(a->ndim, a->dims, a->strides, &dims, &strides) != 0) return NULL;

    if (a->ndim > 0) {
        dims[dim0] = a->dims[dim1];
        dims[dim1] = a->dims[dim0];
        strides[dim0] = a->strides[dim1];
        strides[dim1] = a->strides[dim0];
    }

    tensor* view = make_view(a, a->ndim, dims, strides, a->offset);
    free(dims);
    free(strides);
    return view;
}

tensor* t_contiguous(tensor* t) {
    if (!tensor_has_valid_metadata(t)) return NULL;
    if (is_contiguous(t)) {
        return make_view(t, t->ndim, t->dims, t->strides, t->offset);
    }

    return t_clone(t);
}

tensor* t_reshape(tensor* a, int new_ndim, int* new_dims) {
    if (!tensor_has_valid_metadata(a)) return NULL;

    size_t new_size;
    if (!tensor_checked_numel(new_ndim, new_dims, &new_size) || new_size != (size_t)tensor_numel(a)) {
        fprintf(stderr, "ERROR: Reshape dimensions must match total element count.\n");
        return NULL;
    }

    int* new_strides = NULL;
    if (new_ndim > 0) {
        new_strides = (int*)malloc((size_t)new_ndim * sizeof(int));
        if (new_strides == NULL) return NULL;
        calc_strides(new_ndim, new_dims, new_strides);
    }

    if (is_contiguous(a)) {
        tensor* view = make_view(a, new_ndim, new_dims, new_strides, a->offset);
        free(new_strides);
        return view;
    }

    tensor* contig = t_clone(a);
    if (contig == NULL) {
        free(new_strides);
        return NULL;
    }

    free(contig->dims);
    free(contig->strides);
    contig->dims = NULL;
    contig->strides = NULL;
    contig->ndim = new_ndim;
    contig->offset = 0;

    if (tensor_copy_metadata(new_ndim, new_dims, new_strides, &contig->dims, &contig->strides) != 0) {
        free(new_strides);
        t_free(contig);
        return NULL;
    }

    free(new_strides);
    return contig;
}

tensor* t_slice(tensor* a, int dim, int start, int end) {
    if (!tensor_has_valid_metadata(a)) return NULL;

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for slice.\n");
        return NULL;
    }
    if (start < 0 || end > a->dims[dim] || start >= end) {
        fprintf(stderr, "ERROR: Invalid start/end for slice.\n");
        return NULL;
    }

    int* new_dims = (int*)malloc((size_t)a->ndim * sizeof(int));
    if (new_dims == NULL) return NULL;

    for (int i = 0; i < a->ndim; i++) {
        new_dims[i] = (i == dim) ? (end - start) : a->dims[i];
    }

    tensor* view = make_view(a, a->ndim, new_dims, a->strides, a->offset + start * a->strides[dim]);
    free(new_dims);
    return view;
}
tensor* t_squeeze(tensor* a, int dim) {
    if (!tensor_has_valid_metadata(a)) return NULL;
    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for squeeze.\n");
        return NULL;
    }
    if (a->dims[dim] != 1) {
        fprintf(stderr, "ERROR: Can only squeeze a dimension of size 1.\n");
        return NULL;
    }

    int* new_dims = (int*)malloc((size_t)(a->ndim - 1) * sizeof(int));
    if (new_dims == NULL) return NULL;

    int new_ndim = a->ndim - 1;
    int j = 0;
    int* new_strides = (int*)malloc((size_t)new_ndim * sizeof(int));
    if (new_strides == NULL) {
        free(new_dims);
        return NULL;
    }
    j = 0;
    for (int i=0; i<a->ndim ; i++){
        if (i != dim){
            new_dims[j] = a->dims[i];
            new_strides[j] = a->strides[i];
            j++;
        }
    }

    tensor* view = make_view(a, new_ndim, new_dims, new_strides, a->offset);
    free(new_dims);
    free(new_strides);
    return view;
}

tensor* t_unsqueeze(tensor* a, int dim) {
    if (!tensor_has_valid_metadata(a)) return NULL;
    if (dim < 0 || dim > a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for unsqueeze.\n");
        return NULL;
    }

    int new_ndim = a->ndim + 1;
    int* new_dims = (int*)malloc((size_t)new_ndim * sizeof(int));
    int* new_strides = (int*)malloc((size_t)new_ndim * sizeof(int));
    if (new_dims == NULL || new_strides == NULL) {
        free(new_dims);
        free(new_strides);
        return NULL;
    }

    for (int i = 0, j = 0; i < new_ndim; ++i) {
        if (i == dim) {
            new_dims[i] = 1;
            /* Keep the usual contiguous-stride convention when possible. */
            if (dim < a->ndim) {
                if (a->strides[dim] > INT_MAX / a->dims[dim]) {
                    free(new_dims);
                    free(new_strides);
                    return NULL;
                }
                new_strides[i] = a->strides[dim] * a->dims[dim];
            } else {
                new_strides[i] = 1;
            }
        } else {
            new_dims[i] = a->dims[j];
            new_strides[i] = a->strides[j];
            ++j;
        }
    }

    tensor* view = make_view(a, new_ndim, new_dims, new_strides, a->offset);
    free(new_dims);
    free(new_strides);
    return view;
}
