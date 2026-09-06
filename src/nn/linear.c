#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nn_internal.h"
#include "../../include/tensorlib/autograd_internal.h"

static char* nn_parameter_name(const char* module_name, const char* suffix)
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

static ag_tensor* nn_linear_module_forward(const nn_module* module,
                                           const ag_tensor* input)
{
    return nn_linear_forward((const nn_linear*)module, input);
}

static void nn_linear_module_destroy(nn_module* module)
{
    nn_linear_destroy((nn_linear*)module);
}

nn_linear* nn_linear_create(const char* name,
                            int in_features,
                            int out_features,
                            int use_bias,
                            nn_init_kind weight_init,
                            nn_init_kind bias_init,
                            nn_rng* rng)
{
    int weight_dims[2];
    int bias_dims[1];
    char* parameter_name;
    nn_linear* layer;

    if (in_features <= 0 || out_features <= 0) return NULL;
    layer = (nn_linear*)calloc(1, sizeof(*layer));
    if (layer == NULL) return NULL;
    if (nn_module_init_base(&layer->base,
                            "Linear",
                            name,
                            nn_linear_module_forward,
                            nn_linear_module_destroy) != 0) {
        free(layer);
        return NULL;
    }
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->use_bias = use_bias != 0;

    weight_dims[0] = out_features;
    weight_dims[1] = in_features;
    parameter_name = nn_parameter_name(name, "weight");
    if (parameter_name == NULL) goto fail;
    layer->weight = nn_parameter_create(
        parameter_name, 2, weight_dims, 1, weight_init, rng);
    free(parameter_name);
    if (layer->weight == NULL) goto fail;
    if (nn_module_register_parameter(&layer->base, layer->weight) != 0) {
        nn_parameter_destroy(layer->weight);
        layer->weight = NULL;
        goto fail;
    }

    if (layer->use_bias) {
        bias_dims[0] = out_features;
        parameter_name = nn_parameter_name(name, "bias");
        if (parameter_name == NULL) goto fail;
        layer->bias = nn_parameter_create(
            parameter_name, 1, bias_dims, 1, bias_init, rng);
        free(parameter_name);
        if (layer->bias == NULL) goto fail;
        if (nn_module_register_parameter(&layer->base, layer->bias) != 0) {
            nn_parameter_destroy(layer->bias);
            layer->bias = NULL;
            goto fail;
        }
    }
    return layer;

fail:
    nn_module_destroy_base(&layer->base);
    free(layer);
    return NULL;
}

void nn_linear_destroy(nn_linear* layer)
{
    if (layer == NULL) return;
    t_free_matmul_packed_rhs(layer->packed_weight);
    nn_module_destroy_base(&layer->base);
    free(layer);
}

ag_tensor* nn_linear_forward(const nn_linear* layer, const ag_tensor* input)
{
    ag_tensor* transposed;
    ag_tensor* product;
    ag_tensor* result;
    tensor* input_value;
    uint64_t weight_version;
    nn_linear* mutable_layer;

    if (layer == NULL || input == NULL || layer->weight == NULL ||
        layer->weight->value == NULL) {
        return NULL;
    }
    input_value = input->value;
    mutable_layer = (nn_linear*)layer;
    if (!tensor_has_valid_metadata(input_value) || input_value->ndim < 1 ||
        input_value->dims[input_value->ndim - 1] != layer->in_features) {
        return NULL;
    }

    transposed = ag_transpose(layer->weight->value, 0, 1);
    if (transposed == NULL) return NULL;
    weight_version = layer->weight->value->value->storage->version;
    if (mutable_layer->packed_weight == NULL ||
        mutable_layer->packed_weight_version != weight_version) {
        tensor_matmul_packed_rhs* packed =
            t_pack_matmul_rhs(transposed->value);
        if (packed != NULL) {
            t_free_matmul_packed_rhs(mutable_layer->packed_weight);
            mutable_layer->packed_weight = packed;
            mutable_layer->packed_weight_version = weight_version;
        }
    }
    product = input_value->ndim >= 2 &&
              mutable_layer->packed_weight != NULL &&
              mutable_layer->packed_weight_version == weight_version
            ? ag_matmul_packed_rhs(input, transposed,
                                   mutable_layer->packed_weight)
            : ag_matmul(input, transposed);
    ag_tensor_release(transposed);
    if (product == NULL) return NULL;
    if (!layer->use_bias) return product;
    if (layer->bias == NULL || layer->bias->value == NULL) {
        ag_tensor_release(product);
        return NULL;
    }
    result = ag_add(product, layer->bias->value);
    ag_tensor_release(product);
    return result;
}
