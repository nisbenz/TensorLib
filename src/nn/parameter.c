#include <math.h>
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

static int nn_parameter_fans(int ndim,
                             const int* dims,
                             float* fan_in,
                             float* fan_out)
{
    double receptive = 1.0;

    if (ndim <= 0 || dims == NULL || fan_in == NULL || fan_out == NULL) {
        return -1;
    }
    if (ndim == 1) {
        *fan_in = (float)dims[0];
        *fan_out = (float)dims[0];
        return 0;
    }
    for (int i = 2; i < ndim; ++i) receptive *= (double)dims[i];
    *fan_in = (float)((double)dims[1] * receptive);
    *fan_out = (float)((double)dims[0] * receptive);
    if (!isfinite(*fan_in) || !isfinite(*fan_out) ||
        *fan_in <= 0.0f || *fan_out <= 0.0f) {
        return -1;
    }
    return 0;
}

static int nn_initialize_parameter(tensor* value,
                                   nn_init_kind initializer,
                                   nn_rng* rng)
{
    float fan_in;
    float fan_out;
    float scale;
    int count;

    if (value == NULL) return -1;
    count = tensor_numel(value);
    if (initializer == NN_INIT_ZERO || initializer == NN_INIT_ONE) {
        float fill = initializer == NN_INIT_ZERO ? 0.0f : 1.0f;
        for (int i = 0; i < count; ++i) value->storage->data[i] = fill;
        return 0;
    }
    if (rng == NULL ||
        nn_parameter_fans(value->ndim, value->dims, &fan_in, &fan_out) != 0) {
        return -1;
    }

    switch (initializer) {
        case NN_INIT_XAVIER_UNIFORM:
            scale = sqrtf(6.0f / (fan_in + fan_out));
            for (int i = 0; i < count; ++i) {
                value->storage->data[i] = nn_rng_uniform(rng, -scale, scale);
            }
            return 0;
        case NN_INIT_XAVIER_NORMAL:
            scale = sqrtf(2.0f / (fan_in + fan_out));
            for (int i = 0; i < count; ++i) {
                value->storage->data[i] = nn_rng_normal(rng, 0.0f, scale);
            }
            return 0;
        case NN_INIT_HE_UNIFORM:
            scale = sqrtf(6.0f / fan_in);
            for (int i = 0; i < count; ++i) {
                value->storage->data[i] = nn_rng_uniform(rng, -scale, scale);
            }
            return 0;
        case NN_INIT_HE_NORMAL:
            scale = sqrtf(2.0f / fan_in);
            for (int i = 0; i < count; ++i) {
                value->storage->data[i] = nn_rng_normal(rng, 0.0f, scale);
            }
            return 0;
        default:
            return -1;
    }
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

    if (name == NULL || name[0] == '\0') {
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
    if (nn_initialize_parameter(value, initializer, rng) != 0) {
        t_free(value);
        free(parameter->name);
        free(parameter);
        return NULL;
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
