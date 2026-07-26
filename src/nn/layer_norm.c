#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nn_internal.h"

static char* layer_norm_parameter_name(const char* module_name,
                                       const char* suffix)
{
    size_t module_length;
    size_t suffix_length;
    char* result;

    if (module_name == NULL || suffix == NULL) return NULL;
    module_length = strlen(module_name);
    suffix_length = strlen(suffix);
    if (module_length > SIZE_MAX - suffix_length - 2) return NULL;
    result = (char*)malloc(module_length + suffix_length + 2);
    if (result == NULL) return NULL;
    memcpy(result, module_name, module_length);
    result[module_length] = '.';
    memcpy(result + module_length + 1, suffix, suffix_length + 1);
    return result;
}

static ag_tensor* layer_norm_module_forward(const nn_module* module,
                                            const ag_tensor* input)
{
    return nn_layer_norm_forward((const nn_layer_norm*)module, input);
}

static void layer_norm_module_destroy(nn_module* module)
{
    nn_layer_norm_destroy((nn_layer_norm*)module);
}

static nn_parameter* create_parameter(const char* module_name,
                                      const char* suffix,
                                      int width,
                                      nn_init_kind initializer)
{
    char* name = layer_norm_parameter_name(module_name, suffix);
    nn_parameter* result;
    int dims[1] = {width};

    if (name == NULL) return NULL;
    result = nn_parameter_create(name, 1, dims, 1, initializer, NULL);
    free(name);
    return result;
}

nn_layer_norm* nn_layer_norm_create(const char* name,
                                    int normalized_width,
                                    float epsilon,
                                    int affine)
{
    nn_layer_norm* layer;

    if (normalized_width <= 0 || !isfinite(epsilon) || epsilon <= 0.0f) {
        return NULL;
    }
    layer = (nn_layer_norm*)calloc(1, sizeof(*layer));
    if (layer == NULL) return NULL;
    if (nn_module_init_base(&layer->base, "LayerNorm", name,
                            layer_norm_module_forward,
                            layer_norm_module_destroy) != 0) {
        free(layer);
        return NULL;
    }
    layer->normalized_width = normalized_width;
    layer->epsilon = epsilon;
    layer->affine = affine != 0;
    if (!layer->affine) return layer;

    layer->weight = create_parameter(
        name, "weight", normalized_width, NN_INIT_ONE);
    if (layer->weight == NULL ||
        nn_module_register_parameter(&layer->base, layer->weight) != 0) {
        if (layer->weight != NULL) {
            nn_parameter_destroy(layer->weight);
            layer->weight = NULL;
        }
        goto fail;
    }
    layer->bias = create_parameter(
        name, "bias", normalized_width, NN_INIT_ZERO);
    if (layer->bias == NULL ||
        nn_module_register_parameter(&layer->base, layer->bias) != 0) {
        if (layer->bias != NULL) {
            nn_parameter_destroy(layer->bias);
            layer->bias = NULL;
        }
        goto fail;
    }
    return layer;

fail:
    nn_module_destroy_base(&layer->base);
    free(layer);
    return NULL;
}

void nn_layer_norm_destroy(nn_layer_norm* layer)
{
    if (layer == NULL) return;
    nn_module_destroy_base(&layer->base);
    free(layer);
}

static ag_tensor* scalar(float value)
{
    tensor* raw = t_alloc(0, NULL);

    if (raw == NULL) return NULL;
    raw->storage->data[0] = value;
    return ag_from_owned_tensor(raw, 0);
}

ag_tensor* nn_layer_norm_forward(const nn_layer_norm* layer,
                                 const ag_tensor* input)
{
    ag_tensor* mean = NULL;
    ag_tensor* centered = NULL;
    ag_tensor* squared = NULL;
    ag_tensor* variance = NULL;
    ag_tensor* epsilon = NULL;
    ag_tensor* stabilized = NULL;
    ag_tensor* deviation = NULL;
    ag_tensor* normalized = NULL;
    ag_tensor* scaled = NULL;
    ag_tensor* result = NULL;
    int axis;

    if (layer == NULL || input == NULL ||
        !tensor_has_valid_metadata(input->value) ||
        input->value->ndim < 1 ||
        input->value->dims[input->value->ndim - 1] !=
            layer->normalized_width) {
        return NULL;
    }
    axis = input->value->ndim - 1;
    mean = ag_mean(input, axis, 1);
    if (mean == NULL) goto cleanup;
    centered = ag_sub(input, mean);
    if (centered == NULL) goto cleanup;
    squared = ag_mul(centered, centered);
    if (squared == NULL) goto cleanup;
    variance = ag_mean(squared, axis, 1);
    if (variance == NULL) goto cleanup;
    epsilon = scalar(layer->epsilon);
    if (epsilon == NULL) goto cleanup;
    stabilized = ag_add(variance, epsilon);
    if (stabilized == NULL) goto cleanup;
    deviation = ag_sqrt(stabilized);
    if (deviation == NULL) goto cleanup;
    normalized = ag_div(centered, deviation);
    if (normalized == NULL) goto cleanup;
    if (!layer->affine) {
        result = normalized;
        normalized = NULL;
        goto cleanup;
    }
    if (layer->weight == NULL || layer->bias == NULL ||
        layer->weight->value == NULL || layer->bias->value == NULL) {
        goto cleanup;
    }
    scaled = ag_mul(normalized, layer->weight->value);
    if (scaled == NULL) goto cleanup;
    result = ag_add(scaled, layer->bias->value);

cleanup:
    ag_tensor_release(scaled);
    ag_tensor_release(normalized);
    ag_tensor_release(deviation);
    ag_tensor_release(stabilized);
    ag_tensor_release(epsilon);
    ag_tensor_release(variance);
    ag_tensor_release(squared);
    ag_tensor_release(centered);
    ag_tensor_release(mean);
    return result;
}
