#ifndef TENSORLIB_AUTOGRAD_INTERNAL_H
#define TENSORLIB_AUTOGRAD_INTERNAL_H

#include "../../include/tensorlib/autograd.h"

ag_tensor* ag_make_result(tensor* output,
                          ag_op operation,
                          int input_count,
                          ag_tensor* const* inputs,
                          ag_backward_fn backward,
                          void* context,
                          void (*free_context)(void*));

tensor* ag_full_like(const tensor* reference, float value);

ag_tensor* ag_matmul_packed_rhs(
    const ag_tensor* a,
    const ag_tensor* b,
    const tensor_matmul_packed_rhs* packed_rhs
);

#endif
