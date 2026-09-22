#include <math.h>
#include <stdlib.h>

#include "nn_internal.h"
#include "../../include/tensorlib/autograd_internal.h"

static ag_tensor* layer_norm_module_forward(const nn_module* module,
                                            const ag_tensor* input)
{
    return nn_layer_norm_forward((const nn_layer_norm*)module, input);
}

static void layer_norm_module_destroy(nn_module* module)
{
    nn_layer_norm_destroy((nn_layer_norm*)module);
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

    int parameter_dims[1] = {normalized_width};
    layer->weight = nn_create_named_parameter(
        name, "weight", 1, parameter_dims, NN_INIT_ONE, NULL);
    if (nn_register_owned_parameter(&layer->base, &layer->weight) != 0) {
        goto fail;
    }
    layer->bias = nn_create_named_parameter(
        name, "bias", 1, parameter_dims, NN_INIT_ZERO, NULL);
    if (nn_register_owned_parameter(&layer->base, &layer->bias) != 0) {
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

ag_tensor* nn_layer_norm_forward(const nn_layer_norm* layer,
                                 const ag_tensor* input)
{
    if (layer == NULL || input == NULL ||
        !tensor_has_valid_metadata(input->value) ||
        input->value->ndim < 1 ||
        input->value->dims[input->value->ndim - 1] !=
            layer->normalized_width) {
        return NULL;
    }
    if (layer->weight == NULL || layer->bias == NULL ||
        layer->weight->value == NULL || layer->bias->value == NULL) {
        return layer->affine ? NULL : ag_layer_norm(input, NULL, NULL,
                                                    layer->epsilon);
    }
    return layer->affine
         ? ag_layer_norm(input, layer->weight->value,
                         layer->bias->value, layer->epsilon)
         : ag_layer_norm(input, NULL, NULL, layer->epsilon);
}
