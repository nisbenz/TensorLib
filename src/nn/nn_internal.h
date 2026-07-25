#ifndef TENSORLIB_NN_INTERNAL_H
#define TENSORLIB_NN_INTERNAL_H

#include "../../include/tensorlib/nn.h"

int nn_module_init_base(nn_module* module,
                        const char* type_name,
                        const char* name,
                        nn_module_forward_fn forward,
                        nn_module_destroy_fn destroy);
void nn_module_destroy_base(nn_module* module);

#endif
