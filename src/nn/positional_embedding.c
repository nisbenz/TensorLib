#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_internal.h"

static ag_tensor* positional_embedding_module_forward(
    const nn_module* module,
    const ag_tensor* input)
{
    return nn_positional_embedding_forward(
        (const nn_positional_embedding*)module, input);
}

static void positional_embedding_module_destroy(nn_module* module)
{
    nn_positional_embedding_destroy((nn_positional_embedding*)module);
}

static char* positional_table_name(const char* name)
{
    int required;
    char* result;

    if (name == NULL) return NULL;
    required = snprintf(NULL, 0, "%s.table", name);
    if (required < 0 || (size_t)required == SIZE_MAX) return NULL;
    result = (char*)malloc((size_t)required + 1);
    if (result != NULL) {
        snprintf(result, (size_t)required + 1, "%s.table", name);
    }
    return result;
}

nn_positional_embedding* nn_positional_embedding_create(
    const char* name,
    int context_length,
    int embedding_width,
    nn_init_kind weight_init,
    nn_rng* rng)
{
    nn_positional_embedding* layer;
    char* table_name;

    if (context_length <= 0 || embedding_width <= 0) return NULL;
    layer = (nn_positional_embedding*)calloc(1, sizeof(*layer));
    if (layer == NULL) return NULL;
    if (nn_module_init_base(&layer->base,
                            "PositionalEmbedding",
                            name,
                            positional_embedding_module_forward,
                            positional_embedding_module_destroy) != 0) {
        free(layer);
        return NULL;
    }
    layer->context_length = context_length;
    layer->embedding_width = embedding_width;
    table_name = positional_table_name(name);
    if (table_name == NULL) goto fail;
    layer->table = nn_embedding_create(
        table_name, context_length, embedding_width, weight_init, rng);
    free(table_name);
    if (layer->table == NULL) goto fail;
    if (nn_module_register_child(&layer->base, &layer->table->base) != 0) {
        nn_embedding_destroy(layer->table);
        layer->table = NULL;
        goto fail;
    }
    return layer;

fail:
    nn_module_destroy_base(&layer->base);
    free(layer);
    return NULL;
}

void nn_positional_embedding_destroy(nn_positional_embedding* layer)
{
    if (layer == NULL) return;
    nn_module_destroy_base(&layer->base);
    free(layer);
}

ag_tensor* nn_positional_embedding_forward(
    const nn_positional_embedding* layer,
    const ag_tensor* token_embeddings)
{
    tensor* position_values;
    ag_tensor* position_ids;
    ag_tensor* positions;
    ag_tensor* result;
    int position_dims[1];
    int sequence;

    if (layer == NULL || token_embeddings == NULL ||
        !tensor_has_valid_metadata(token_embeddings->value) ||
        token_embeddings->value->ndim != 3 ||
        token_embeddings->value->dims[2] != layer->embedding_width ||
        layer->table == NULL) {
        return NULL;
    }
    sequence = token_embeddings->value->dims[1];
    if (sequence > layer->context_length) return NULL;
    position_dims[0] = sequence;
    position_values = t_alloc(1, position_dims);
    if (position_values == NULL) return NULL;
    for (int position = 0; position < sequence; ++position) {
        position_values->storage->data[position] = (float)position;
    }
    position_ids = ag_from_owned_tensor(position_values, 0);
    if (position_ids == NULL) return NULL;
    positions = nn_embedding_forward(layer->table, position_ids);
    ag_tensor_release(position_ids);
    if (positions == NULL) return NULL;
    result = ag_add(token_embeddings, positions);
    ag_tensor_release(positions);
    return result;
}
