#include <stdlib.h>

#include "nn_internal.h"

static ag_tensor* embedding_module_forward(const nn_module* module,
                                           const ag_tensor* input)
{
    return nn_embedding_forward((const nn_embedding*)module, input);
}

static void embedding_module_destroy(nn_module* module)
{
    nn_embedding_destroy((nn_embedding*)module);
}

nn_embedding* nn_embedding_create(const char* name,
                                  int vocabulary_size,
                                  int embedding_width,
                                  nn_init_kind weight_init,
                                  nn_rng* rng)
{
    nn_embedding* layer;
    char* parameter_name;
    int dims[2];

    if (vocabulary_size <= 0 || embedding_width <= 0) return NULL;
    layer = (nn_embedding*)calloc(1, sizeof(*layer));
    if (layer == NULL) return NULL;
    if (nn_module_init_base(&layer->base, "Embedding", name,
                            embedding_module_forward,
                            embedding_module_destroy) != 0) {
        free(layer);
        return NULL;
    }
    layer->vocabulary_size = vocabulary_size;
    layer->embedding_width = embedding_width;
    dims[0] = vocabulary_size;
    dims[1] = embedding_width;
    parameter_name = nn_qualified_name(name, "weight");
    if (parameter_name == NULL) goto fail;
    layer->weight = nn_parameter_create(
        parameter_name, 2, dims, 1, weight_init, rng);
    free(parameter_name);
    if (nn_register_owned_parameter(&layer->base, &layer->weight) != 0) {
        goto fail;
    }
    return layer;

fail:
    nn_module_destroy_base(&layer->base);
    free(layer);
    return NULL;
}

void nn_embedding_destroy(nn_embedding* layer)
{
    if (layer == NULL) return;
    nn_module_destroy_base(&layer->base);
    free(layer);
}

ag_tensor* nn_embedding_forward(const nn_embedding* layer,
                                const ag_tensor* indices)
{
    if (layer == NULL || indices == NULL || indices->requires_grad ||
        layer->weight == NULL || layer->weight->value == NULL) {
        return NULL;
    }
    return ag_gather_rows(layer->weight->value, indices->value);
}
