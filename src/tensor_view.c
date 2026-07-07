#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"

static tensor* make_view(tensor* base, int ndim, const int* dims, const int* strides, int offset) {
    if (base == NULL || base->storage == NULL) return NULL;
    if (ndim < 0 || (ndim > 0 && (dims == NULL || strides == NULL))) return NULL;

    tensor* view = (tensor*)malloc(sizeof(tensor));
    if (view == NULL) return NULL;

    view->storage = NULL;
    view->dims = NULL;
    view->strides = NULL;
    view->ndim = ndim;
    view->offset = offset;

    if (ndim > 0) {
        view->dims = (int*)malloc(ndim * sizeof(int));
        view->strides = (int*)malloc(ndim * sizeof(int));
        if (view->dims == NULL || view->strides == NULL) {
            t_free(view);
            return NULL;
        }

        for (int i = 0; i < ndim; i++) {
            view->dims[i] = dims[i];
            view->strides[i] = strides[i];
        }
    }

    add_ref_count(base->storage, view);
    return view;
}

tensor* t_transpose(tensor* a, int dim0, int dim1) {
    if (a == NULL) return NULL;
    if (dim0 < 0 || dim0 >= a->ndim || dim1 < 0 || dim1 >= a->ndim) {
        fprintf(stderr, "ERROR: Transpose dimensions out of bounds.\n");
        return NULL;
    }

    int* dims = NULL;
    int* strides = NULL;
    if (a->ndim > 0) {
        dims = (int*)malloc(a->ndim * sizeof(int));
        strides = (int*)malloc(a->ndim * sizeof(int));
        if (dims == NULL || strides == NULL) {
            free(dims);
            free(strides);
            return NULL;
        }

        for (int i = 0; i < a->ndim; i++) {
            dims[i] = a->dims[i];
            strides[i] = a->strides[i];
        }

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
    if (t == NULL) return NULL;
    if (is_contiguous(t)) {
        return make_view(t, t->ndim, t->dims, t->strides, t->offset);
    }

    return t_clone(t);
}

tensor* t_reshape(tensor* a, int new_ndim, int* new_dims) {
    if (a == NULL || a->storage == NULL) return NULL;
    if (new_ndim < 0 || (new_ndim > 0 && new_dims == NULL)) return NULL;

    int new_size = 1;
    for (int i = 0; i < new_ndim; ++i) {
        new_size *= new_dims[i];
    }
    if (new_size != tensor_numel(a)) {
        fprintf(stderr, "ERROR: Reshape dimensions must match total element count.\n");
        return NULL;
    }

    int* new_strides = NULL;
    if (new_ndim > 0) {
        new_strides = (int*)malloc(new_ndim * sizeof(int));
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

    if (new_ndim > 0) {
        contig->dims = (int*)malloc(new_ndim * sizeof(int));
        contig->strides = (int*)malloc(new_ndim * sizeof(int));
        if (contig->dims == NULL || contig->strides == NULL) {
            free(new_strides);
            t_free(contig);
            return NULL;
        }
        for (int i = 0; i < new_ndim; i++) {
            contig->dims[i] = new_dims[i];
            contig->strides[i] = new_strides[i];
        }
    }

    free(new_strides);
    return contig;
}

tensor* t_slice(tensor* a, int dim, int start, int end) {
    if (a == NULL) return NULL;

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "ERROR: Invalid dimension for slice.\n");
        return NULL;
    }
    if (start < 0 || end > a->dims[dim] || start >= end) {
        fprintf(stderr, "ERROR: Invalid start/end for slice.\n");
        return NULL;
    }

    int* new_dims = (int*)malloc(a->ndim * sizeof(int));
    if (new_dims == NULL) return NULL;

    for (int i = 0; i < a->ndim; i++) {
        new_dims[i] = (i == dim) ? (end - start) : a->dims[i];
    }

    tensor* view = make_view(a, a->ndim, new_dims, a->strides, a->offset + start * a->strides[dim]);
    free(new_dims);
    return view;
}
