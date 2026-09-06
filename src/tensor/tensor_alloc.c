#include <stdlib.h>
#include <string.h>
#include "../../include/tensorlib/tensor.h"
#include "tensor_alloc_internal.h"

static int stats_enabled;
static volatile unsigned long long stats_allocations;
static volatile unsigned long long stats_frees;
static volatile size_t stats_allocated_bytes;
static volatile size_t stats_live_bytes;
static volatile size_t stats_peak_live_bytes;

static void record_storage_alloc(size_t bytes)
{
    if (!stats_enabled) return;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    ++stats_allocations;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    stats_allocated_bytes += bytes;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    stats_live_bytes += bytes;
#ifdef _OPENMP
#pragma omp critical(tensor_alloc_stats_peak)
#endif
    if (stats_live_bytes > stats_peak_live_bytes) {
        stats_peak_live_bytes = stats_live_bytes;
    }
}

static void record_storage_free(size_t bytes)
{
    if (!stats_enabled) return;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    ++stats_frees;
#ifdef _OPENMP
#pragma omp atomic update
#endif
    stats_live_bytes -= bytes;
}

void tensor_alloc_stats_enable(int enabled)
{
    stats_enabled = enabled != 0;
}

void tensor_alloc_stats_reset(void)
{
    stats_allocations = 0;
    stats_frees = 0;
    stats_allocated_bytes = 0;
    stats_live_bytes = 0;
    stats_peak_live_bytes = 0;
}

void tensor_alloc_stats_reset_counters(void)
{
    stats_allocations = 0;
    stats_frees = 0;
    stats_allocated_bytes = 0;
    stats_peak_live_bytes = stats_live_bytes;
}

void tensor_alloc_stats_read(tensor_alloc_stats* result)
{
    if (result == NULL) return;
    result->allocations = stats_allocations;
    result->frees = stats_frees;
    result->allocated_bytes = stats_allocated_bytes;
    result->live_bytes = stats_live_bytes;
    result->peak_live_bytes = stats_peak_live_bytes;
}

void add_ref_count(Storage* a, tensor* b) {
    if (a != NULL && b != NULL) {
        a->ref_count++;
        b->storage = a;
    }
}

Storage* s_alloc(int ndim, const int* dims) {
    size_t count;
    if (!tensor_checked_numel(ndim, dims, &count)) return NULL;

    Storage* s = (Storage*)malloc(sizeof(Storage));
    if (s == NULL) return NULL;
    s->ref_count = 1;
    s->size = (int)count;
    s->version = 0;
    s->data = (float*)malloc(count * sizeof(float));
    if (s->data == NULL) {
        free(s);
        return NULL;
    }
    record_storage_alloc(count * sizeof(float));
    return s;
}

tensor* t_alloc(int ndim, const int* dims) {
    size_t count;
    if (!tensor_checked_numel(ndim, dims, &count)) return NULL;
    (void)count;

    tensor* a = (tensor*)calloc(1, sizeof(tensor));
    if (a == NULL) return NULL;
    a->ndim = ndim;

    if (ndim > 0) {
        int* strides = (int*)malloc((size_t)ndim * sizeof(int));
        if (strides == NULL) {
            t_free(a);
            return NULL;
        }
        calc_strides(ndim, dims, strides);
        if (tensor_copy_metadata(ndim, dims, strides, &a->dims, &a->strides) != 0) {
            free(strides);
            t_free(a);
            return NULL;
        }
        free(strides);
    }

    a->storage = s_alloc(ndim, dims);
    if (a->storage == NULL) {
        t_free(a);
        return NULL;
    }
    return a;
}

void t_free(tensor* t) {
    if (t == NULL) return;
    if (t->storage != NULL) {
        if (t->storage->ref_count > 1) {
            t->storage->ref_count--;
        } else {
            record_storage_free((size_t)t->storage->size * sizeof(float));
            free(t->storage->data);
            free(t->storage);
        }
    }
    free(t->dims);
    free(t->strides);
    free(t);
}

int init_t(tensor* c, tensor* ref) {
    if (c == NULL || !tensor_has_valid_shape(ref)) return 1;
    int total_elements = tensor_numel(ref);
    if (total_elements == 0) return 1;

    c->storage = NULL;
    c->dims = NULL;
    c->strides = NULL;
    c->ndim = ref->ndim;
    c->offset = 0;

    c->storage = (Storage*)malloc(sizeof(Storage));
    if (c->storage == NULL) return 1;
    c->storage->ref_count = 1;
    c->storage->size = total_elements;
    c->storage->version = 0;
    c->storage->data = (float*)calloc((size_t)total_elements, sizeof(float));
    if (c->storage->data == NULL) {
        free(c->storage);
        c->storage = NULL;
        return 1;
    }
    record_storage_alloc((size_t)total_elements * sizeof(float));

    if (ref->ndim > 0) {
        int* strides = (int*)malloc((size_t)ref->ndim * sizeof(int));
        if (strides == NULL) {
            free(c->storage->data);
            free(c->storage);
            c->storage = NULL;
            return 1;
        }
        calc_strides(ref->ndim, ref->dims, strides);
        if (tensor_copy_metadata(ref->ndim, ref->dims, strides, &c->dims, &c->strides) != 0) {
            free(strides);
            free(c->storage->data);
            free(c->storage);
            c->storage = NULL;
            return 1;
        }
        free(strides);
    }
    return 0;
}

tensor* t_clone(tensor* t) {
    if (!tensor_has_valid_metadata(t)) return NULL;
    tensor* a = t_alloc(t->ndim, t->dims);
    if (a == NULL) return NULL;

    int total_elements = tensor_numel(t);

    /* Contiguous clones are plain memory copies. */
    if (is_contiguous(t)) {
        memcpy(a->storage->data, t->storage->data + t->offset,
               (size_t)total_elements * sizeof(float));
        return a;
    }

    if (t->ndim == 2 && t->strides[0] == 1 &&
        t->strides[1] >= t->dims[0]) {
        const int tile = 32;
        for (int row_block = 0; row_block < t->dims[0]; row_block += tile) {
            int row_end = row_block + tile;
            if (row_end > t->dims[0]) row_end = t->dims[0];
            for (int column_block = 0; column_block < t->dims[1];
                 column_block += tile) {
                int column_end = column_block + tile;
                if (column_end > t->dims[1]) column_end = t->dims[1];
                for (int row = row_block; row < row_end; ++row) {
                    for (int column = column_block; column < column_end; ++column) {
                        a->storage->data[row * t->dims[1] + column] =
                            t->storage->data[t->offset + row * t->strides[0] +
                                             column * t->strides[1]];
                    }
                }
            }
        }
        return a;
    }

    int* coords = (int*)calloc((size_t)t->ndim, sizeof(int));
    int src_idx = t->offset;
    if (t->ndim > 0 && coords == NULL) {
        t_free(a);
        return NULL;
    }

    for (int i = 0; i < total_elements; i++) {
        a->storage->data[i] = t->storage->data[src_idx];
        for (int axis = t->ndim - 1; axis >= 0; --axis) {
            coords[axis]++;
            src_idx += t->strides[axis];
            if (coords[axis] < t->dims[axis]) break;
            coords[axis] = 0;
            src_idx -= t->dims[axis] * t->strides[axis];
        }
    }
    free(coords);
    return a;
}

void tensor_mark_modified(tensor* value) {
    if (value == NULL || value->storage == NULL) return;
    value->storage->version++;
}
