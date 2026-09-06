#ifndef TENSORLIB_TENSOR_MATMUL_INTERNAL_H
#define TENSORLIB_TENSOR_MATMUL_INTERNAL_H

#include "../../include/tensorlib/tensor.h"

tensor* tensor_matmul_backward_rhs(const tensor* lhs,
                                   const tensor* output_gradient,
                                   const tensor* rhs);

#endif
