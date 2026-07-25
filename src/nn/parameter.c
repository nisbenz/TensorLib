#include <stdlib.h>
#include <string.h>

#include "../../include/tensorlib/nn.h"

static char* nn_copy_string(const char* value)
{
    size_t length;
    char* copy;

    if (value == NULL) return NULL;
    length = strlen(value);
    copy = (char*)malloc(length + 1);
    if (copy != NULL) memcpy(copy, value, length + 1);
    return copy;
}

nn_parameter* nn_parameter_create(const char* name,
                                  int ndim,
                                  const int* dims,
                                  int trainable,
                                  nn_init_kind initializer,
                                  nn_rng* rng)
{
    nn_parameter* parameter;
    tensor* value;

    (void)rng;
    if (name == NULL || name[0] == '\0' || initializer != NN_INIT_ZERO) {
        return NULL;
    }

    parameter = (nn_parameter*)calloc(1, sizeof(*parameter));
    if (parameter == NULL) return NULL;
    parameter->name = nn_copy_string(name);
    if (parameter->name == NULL) {
        free(parameter);
        return NULL;
    }

    value = t_alloc(ndim, dims);
    if (value == NULL) {
        free(parameter->name);
        free(parameter);
        return NULL;
    }
    for (int i = 0; i < tensor_numel(value); ++i) {
        value->storage->data[i] = 0.0f;
    }
    parameter->value = ag_from_owned_tensor(value, trainable != 0);
    if (parameter->value == NULL) {
        free(parameter->name);
        free(parameter);
        return NULL;
    }
    parameter->trainable = trainable != 0;
    return parameter;
}

void nn_parameter_destroy(nn_parameter* parameter)
{
    if (parameter == NULL) return;
    ag_tensor_release(parameter->value);
    free(parameter->name);
    free(parameter);
}
