#ifndef TENSORLIB_TENSOR_H
#define TENSORLIB_TENSOR_H

#include <stddef.h>

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
    int offset;
} tensor;

void add_ref_count(Storage* a, tensor* b);
tensor* t_transpose(tensor* a, int dim0, int dim1);
int same_shape(tensor* a, tensor* b);
int same_stride(tensor* a, tensor* b);
int init_t(tensor* c, tensor* ref);
tensor* t_add(tensor* a, tensor* b);
tensor* t_sub(tensor* a, tensor* b);
tensor* t_mul(tensor* a, tensor* b);
tensor* t_div(tensor* a, tensor* b);
void advance_coords(int* coords, const int* dims, int ndim);
int get_flat_index_nd(tensor* t, int* coords);
int tensor_numel(tensor* t);
int tensor_checked_numel(int ndim, const int* dims, size_t* result);
int tensor_has_valid_shape(const tensor* t);
int tensor_has_valid_layout(const tensor* t);
int tensor_has_valid_metadata(const tensor* t);
int tensor_copy_metadata(int ndim, const int* dims, const int* strides, int** out_dims, int** out_strides);
/* Scalars use ndim == 0. Dimensions must be positive; zero-sized tensors are unsupported. */
Storage* s_alloc(int ndim, const int* dims);
tensor* t_alloc(int ndim, const int* dims);
void t_free(tensor* t);
tensor* t_clone(tensor* t);
int is_contiguous(tensor* t);
tensor* t_contiguous(tensor* t);
void calc_strides(int ndim, const int* dims, int* strides);
tensor* t_reshape(tensor* a, int new_ndim, int* new_dims);
tensor* t_squeeze(tensor* a, int dim);
tensor* t_unsqueeze(tensor* a, int dim);
tensor* t_expand(tensor* a, int new_ndim, const int* new_dims);
tensor* t_slice(tensor* a, int dim, int start, int end);
tensor* t_exp(tensor* t);
tensor* t_log(tensor* t);
tensor* t_relu(tensor* t);
tensor* t_tanh(tensor* t);
tensor* t_sigmoid(tensor* t);
tensor* t_pow(tensor* t, float exponent);
tensor* t_neg(tensor* t);
tensor* t_sqrt(tensor* t);
tensor* t_gelu(tensor* t);
tensor* t_sum(tensor* a, int dim);
tensor* t_mean(tensor* a, int dim);
tensor* t_max(tensor* a, int dim);
tensor* t_matmul(tensor* a, tensor* b);
#endif //TENSORLIB_TENSOR_H
