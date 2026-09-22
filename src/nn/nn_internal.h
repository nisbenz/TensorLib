#ifndef TENSORLIB_NN_INTERNAL_H
#define TENSORLIB_NN_INTERNAL_H

#include "../../include/tensorlib/nn.h"

int nn_module_init_base(nn_module* module,
                        const char* type_name,
                        const char* name,
                        nn_module_forward_fn forward,
                        nn_module_destroy_fn destroy);
void nn_module_destroy_base(nn_module* module);

int nn_module_is_valid(const nn_module* module);
int nn_adamw_config_is_valid(const nn_adamw_config* config);
char* nn_qualified_name(const char* module_name, const char* suffix);
char* nn_indexed_name(const char* module_name,
                      const char* collection,
                      size_t index);
nn_parameter* nn_create_named_parameter(const char* module_name,
                                         const char* suffix,
                                         int ndim,
                                         const int* dims,
                                         nn_init_kind initializer,
                                         nn_rng* rng);
int nn_register_owned_parameter(nn_module* module, nn_parameter** parameter);
int nn_register_owned_child(nn_module* module, nn_module* child);

#endif
