#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H


typedef struct {
    float* data;
    int ref_count;
    int size;
} Storage;

typedef struct {
    Storage* storage;
    int ndim;
    int* dims;
    int* strides;
} tensor;

void add_ref_count(Storage* a, tensor* b);
tensor* t_transpose(tensor* a, int dim0, int dim1);
int same_shape(tensor* a, tensor* b);
int same_stride(tensor* a, tensor* b);
int init_t(tensor* c, tensor* ref);
tensor* t_add(tensor* a, tensor* b);
void advance_coords(int* coords, const int* dims, int ndim);
int get_flat_index_nd(tensor* t, int* coords);
Storage* s_alloc(int ndim, const int* dims);
tensor* t_alloc(int ndim, const int* dims);
void t_free(tensor* t);
tensor* t_clone(tensor* t);
int is_contiguous(tensor* t);
tensor* t_contiguous(tensor* t);
void calc_strides(int ndim, int* dims, int* strides);

#endif //TENSORLIB_TENSOR_H
