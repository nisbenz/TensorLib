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

#endif
