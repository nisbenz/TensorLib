#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"

tensor* t_transpose(tensor* a, int dim0, int dim1) {
    if (a == NULL) return NULL;
    if (dim0 < 0 || dim0 >= a->ndim || dim1 < 0 || dim1 >= a->ndim) {
        fprintf(stderr, "ERROR: Transpose dimensions out of bounds.\n");
        return NULL;
    }

    tensor* b = (tensor*)malloc(sizeof(tensor));
    if (b == NULL) return NULL;

    b->storage = NULL;
    b->dims = NULL;
    b->strides = NULL;
    b->ndim = a->ndim;

    if (a->ndim > 0) {
        b->dims = (int*)malloc(a->ndim * sizeof(int));
        b->strides = (int*)malloc(a->ndim * sizeof(int));
        if (b->dims == NULL || b->strides == NULL) {
            t_free(b);
            return NULL;
        }

        for (int i = 0; i < a->ndim; i++) {
            b->dims[i] = a->dims[i];
            b->strides[i] = a->strides[i];
        }

        b->dims[dim0] = a->dims[dim1];
        b->dims[dim1] = a->dims[dim0];

        b->strides[dim0] = a->strides[dim1];
        b->strides[dim1] = a->strides[dim0];
    }

    add_ref_count(a->storage, b);
    return b;
}

tensor* t_contiguous(tensor* t) {
    if (t == NULL) return NULL;
    if (is_contiguous(t)) {
        tensor* view = (tensor*)malloc(sizeof(tensor));
        if (view == NULL) return NULL;

        view->storage = NULL;
        view->dims = NULL;
        view->strides = NULL;
        view->ndim = t->ndim;

        if (t->ndim > 0) {
            view->dims = (int*)malloc(t->ndim * sizeof(int));
            view->strides = (int*)malloc(t->ndim * sizeof(int));
            if (view->dims == NULL || view->strides == NULL) {
                t_free(view);
                return NULL;
            }
            for(int i = 0; i < t->ndim; i++) {
                view->dims[i] = t->dims[i];
                view->strides[i] = t->strides[i];
            }
        }

        add_ref_count(t->storage, view);
        return view;
    }

    tensor* flat_tensor = t_alloc(t->ndim, t->dims);
    if (flat_tensor == NULL) return NULL;

    int* coords = (int*)calloc(t->ndim, sizeof(int));
    if (coords == NULL) {
        t_free(flat_tensor);
        return NULL;
    }

    int total_elements = flat_tensor->storage->size;
    for (int i = 0; i < total_elements; i++) {
        int old_flat_idx = get_flat_index_nd(t, coords);
        flat_tensor->storage->data[i] = t->storage->data[old_flat_idx];
        advance_coords(coords, t->dims, t->ndim);
    }

    free(coords);
    return flat_tensor;
}

tensor* t_reshape(tensor* a, int new_ndim, int* new_dims) {
    if (a == NULL || a->storage == NULL) return NULL;
    if (new_ndim < 0) return NULL;
    if (new_ndim > 0 && new_dims == NULL) return NULL;

    int new_size = 1;
    for (int i = 0; i < new_ndim; ++i) {
        new_size *= new_dims[i];
    }
    if (new_size != a->storage->size) {
        fprintf(stderr, "ERROR: Reshape dimensions must match total element count.\n");
        return NULL;
    }


    tensor* contig = t_contiguous(a);
    if (contig == NULL) return NULL;

    // 3. Allocate wrapper for the view
    tensor* view = (tensor*)malloc(sizeof(tensor));
    if (view == NULL) {
        t_free(contig);
        return NULL;
    }
    view->ndim = new_ndim;
    view->dims = NULL;
    view->strides = NULL;


    view->storage = contig->storage;

    // 5. Allocate and copy new dimensions
    if (new_ndim > 0) {
        view->dims = (int*)malloc(new_ndim * sizeof(int));
        view->strides = (int*)malloc(new_ndim * sizeof(int));
        if (view->dims == NULL || view->strides == NULL) {

            free(view->dims);
            free(view->strides);
            free(view);
            t_free(contig);
            return NULL;
        }
        for (int i = 0; i < new_ndim; ++i) {
            view->dims[i] = new_dims[i];
        }

        // 6. Automatically recalculate row-major strides using our helper!
        calc_strides(view->ndim, view->dims, view->strides);
    }

    free(contig->dims);
    free(contig->strides);
    free(contig);

    return view;
}