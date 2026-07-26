#include <math.h>
#include <stdlib.h>

#include "nn_internal.h"

static ag_tensor* dropout_module_forward(const nn_module* module,
                                         const ag_tensor* input)
{
    return nn_dropout_forward((const nn_dropout*)module, input);
}

static void dropout_module_destroy(nn_module* module)
{
    nn_dropout_destroy((nn_dropout*)module);
}

nn_dropout* nn_dropout_create(const char* name,
                              float probability,
                              nn_rng* rng)
{
    nn_dropout* layer;

    if (!isfinite(probability) || probability < 0.0f ||
        probability >= 1.0f || (probability > 0.0f && rng == NULL)) {
        return NULL;
    }
    layer = (nn_dropout*)calloc(1, sizeof(*layer));
    if (layer == NULL) return NULL;
    if (nn_module_init_base(&layer->base, "Dropout", name,
                            dropout_module_forward,
                            dropout_module_destroy) != 0) {
        free(layer);
        return NULL;
    }
    layer->probability = probability;
    layer->rng = rng;
    return layer;
}

void nn_dropout_destroy(nn_dropout* layer)
{
    if (layer == NULL) return;
    nn_module_destroy_base(&layer->base);
    free(layer);
}

ag_tensor* nn_dropout_forward(const nn_dropout* layer,
                              const ag_tensor* input)
{
    tensor* mask_value;
    ag_tensor* mask;
    ag_tensor* result;
    float scale;
    int count;

    if (layer == NULL || input == NULL ||
        !tensor_has_valid_metadata(input->value)) {
        return NULL;
    }
    if (!nn_module_is_training(&layer->base) ||
        layer->probability == 0.0f) {
        ag_tensor_retain((ag_tensor*)input);
        return (ag_tensor*)input;
    }
    if (layer->rng == NULL) return NULL;
    mask_value = t_alloc(input->value->ndim, input->value->dims);
    if (mask_value == NULL) return NULL;
    scale = 1.0f / (1.0f - layer->probability);
    count = tensor_numel(mask_value);
    for (int i = 0; i < count; ++i) {
        mask_value->storage->data[i] =
            nn_rng_uniform(layer->rng, 0.0f, 1.0f) < layer->probability
                ? 0.0f : scale;
    }
    mask = ag_from_owned_tensor(mask_value, 0);
    if (mask == NULL) return NULL;
    result = ag_mul(input, mask);
    ag_tensor_release(mask);
    return result;
}
