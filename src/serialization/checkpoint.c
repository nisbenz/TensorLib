#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "../../include/tensorlib/nn.h"

#define CHECKPOINT_VERSION UINT32_C(1)
#define CHECKPOINT_FLAG_ADAMW UINT32_C(1)
#define CHECKPOINT_FLAG_RNG UINT32_C(2)
#define CHECKPOINT_MAX_NAME UINT32_C(1048576)
#define CHECKPOINT_MAX_NDIM UINT32_C(64)

static const unsigned char checkpoint_magic[8] = {
    'T', 'L', 'C', 'K', 'P', 'T', 0, 0
};
static const unsigned char legacy_magic[8] = {
    'T', 'L', 'W', 'E', 'I', 'G', 'H', 'T'
};

typedef struct {
    char* name;
    int ndim;
    int* dims;
    size_t count;
    float* values;
    float* first_moment;
    float* second_moment;
    uint64_t step;
} saved_parameter;

typedef struct {
    saved_parameter* parameters;
    size_t parameter_count;
    uint32_t flags;
    nn_adamw_config config;
    uint64_t rng_state;
} saved_checkpoint;

static int tensor_flat_index(const tensor* value, int flat)
{
    int index = value->offset;
    int remaining = flat;

    for (int dim = value->ndim - 1; dim >= 0; --dim) {
        int coordinate = remaining % value->dims[dim];
        remaining /= value->dims[dim];
        index += coordinate * value->strides[dim];
    }
    return index;
}

static int module_valid(const nn_module* module)
{
    if (module == NULL ||
        module->parameter_count > module->parameter_capacity ||
        module->child_count > module->child_capacity ||
        (module->parameter_count > 0 && module->parameters == NULL) ||
        (module->child_count > 0 && module->children == NULL)) {
        return 0;
    }
    for (size_t i = 0; i < module->parameter_count; ++i) {
        if (module->parameters[i] == NULL) return 0;
    }
    for (size_t i = 0; i < module->child_count; ++i) {
        if (!module_valid(module->children[i])) return 0;
    }
    return 1;
}

static int write_bytes(FILE* file, const void* data, size_t size)
{
    return size == 0 || fwrite(data, 1, size, file) == size;
}

static int read_bytes(FILE* file, void* data, size_t size)
{
    return size == 0 || fread(data, 1, size, file) == size;
}

static int write_u32(FILE* file, uint32_t value)
{
    unsigned char bytes[4];
    for (int i = 0; i < 4; ++i) {
        bytes[i] = (unsigned char)(value >> (8 * i));
    }
    return write_bytes(file, bytes, sizeof(bytes));
}

static int read_u32(FILE* file, uint32_t* value)
{
    unsigned char bytes[4];
    uint32_t result = 0;

    if (!read_bytes(file, bytes, sizeof(bytes))) return 0;
    for (int i = 0; i < 4; ++i) {
        result |= (uint32_t)bytes[i] << (8 * i);
    }
    *value = result;
    return 1;
}

static int write_u64(FILE* file, uint64_t value)
{
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = (unsigned char)(value >> (8 * i));
    }
    return write_bytes(file, bytes, sizeof(bytes));
}

static int read_u64(FILE* file, uint64_t* value)
{
    unsigned char bytes[8];
    uint64_t result = 0;

    if (!read_bytes(file, bytes, sizeof(bytes))) return 0;
    for (int i = 0; i < 8; ++i) {
        result |= (uint64_t)bytes[i] << (8 * i);
    }
    *value = result;
    return 1;
}

static int write_f32(FILE* file, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return write_u32(file, bits);
}

static int read_f32(FILE* file, float* value)
{
    uint32_t bits;
    if (!read_u32(file, &bits)) return 0;
    memcpy(value, &bits, sizeof(bits));
    return 1;
}

static int write_string(FILE* file, const char* value)
{
    size_t length;

    if (value == NULL || value[0] == '\0') return 0;
    length = strlen(value);
    if (length > UINT32_MAX) return 0;
    return write_u32(file, (uint32_t)length) &&
           write_bytes(file, value, length);
}

static char* read_string(FILE* file)
{
    uint32_t length;
    char* result;

    if (!read_u32(file, &length) || length == 0 ||
        length > CHECKPOINT_MAX_NAME) {
        return NULL;
    }
    result = (char*)malloc((size_t)length + 1);
    if (result == NULL) return NULL;
    if (!read_bytes(file, result, length)) {
        free(result);
        return NULL;
    }
    result[length] = '\0';
    return result;
}

static int adamw_config_valid(const nn_adamw_config* config)
{
    return isfinite(config->learning_rate) && config->learning_rate > 0.0f &&
           isfinite(config->beta1) &&
           config->beta1 >= 0.0f && config->beta1 < 1.0f &&
           isfinite(config->beta2) &&
           config->beta2 >= 0.0f && config->beta2 < 1.0f &&
           isfinite(config->epsilon) && config->epsilon > 0.0f &&
           isfinite(config->weight_decay) && config->weight_decay >= 0.0f &&
           isfinite(config->max_grad_norm) && config->max_grad_norm >= 0.0f;
}

static int optimizer_matches(const nn_adamw* optimizer,
                             const nn_module* module)
{
    size_t count;

    if (optimizer == NULL || optimizer->module != module ||
        !adamw_config_valid(&optimizer->config) || !module_valid(module)) {
        return 0;
    }
    count = nn_module_parameter_count(module);
    if (count == SIZE_MAX || count != optimizer->parameter_count ||
        (count > 0 &&
         (optimizer->parameters == NULL ||
          optimizer->first_moments == NULL ||
          optimizer->second_moments == NULL ||
          optimizer->steps == NULL))) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(module, i);
        if (parameter == NULL || optimizer->parameters[i] != parameter ||
            parameter->value == NULL ||
            !tensor_has_valid_metadata(parameter->value->value) ||
            !tensor_has_valid_metadata(optimizer->first_moments[i]) ||
            !tensor_has_valid_metadata(optimizer->second_moments[i]) ||
            !same_shape(parameter->value->value,
                        optimizer->first_moments[i]) ||
            !same_shape(parameter->value->value,
                        optimizer->second_moments[i])) {
            return 0;
        }
    }
    return 1;
}

static int parameters_unique_and_valid(const nn_module* module)
{
    size_t count = nn_module_parameter_count(module);

    if (count == SIZE_MAX || count > UINT32_MAX) return 0;
    for (size_t i = 0; i < count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(module, i);
        if (parameter == NULL || parameter->name == NULL ||
            parameter->name[0] == '\0' || parameter->value == NULL ||
            !tensor_has_valid_metadata(parameter->value->value)) {
            return 0;
        }
        for (size_t earlier = 0; earlier < i; ++earlier) {
            if (strcmp(parameter->name,
                       nn_module_parameter_at(module, earlier)->name) == 0) {
                return 0;
            }
        }
    }
    return 1;
}

static int write_tensor_values(FILE* file, const tensor* value)
{
    int count = tensor_numel((tensor*)value);
    for (int i = 0; i < count; ++i) {
        float element =
            value->storage->data[tensor_flat_index(value, i)];
        if (!isfinite(element) || !write_f32(file, element)) return 0;
    }
    return 1;
}

static int write_parameter(FILE* file, const nn_parameter* parameter)
{
    const tensor* value = parameter->value->value;

    if (!write_string(file, parameter->name) ||
        !write_u32(file, (uint32_t)value->ndim)) {
        return 0;
    }
    for (int dim = 0; dim < value->ndim; ++dim) {
        if (!write_u32(file, (uint32_t)value->dims[dim])) return 0;
    }
    return write_u64(file, (uint64_t)tensor_numel((tensor*)value)) &&
           write_tensor_values(file, value);
}

static int write_optimizer(FILE* file, const nn_adamw* optimizer)
{
    const nn_adamw_config* config = &optimizer->config;

    if (!write_f32(file, config->learning_rate) ||
        !write_f32(file, config->beta1) ||
        !write_f32(file, config->beta2) ||
        !write_f32(file, config->epsilon) ||
        !write_f32(file, config->weight_decay) ||
        !write_f32(file, config->max_grad_norm) ||
        !write_u32(file, (uint32_t)optimizer->parameter_count)) {
        return 0;
    }
    for (size_t i = 0; i < optimizer->parameter_count; ++i) {
        for (int element = 0;
             element < tensor_numel(optimizer->second_moments[i]);
             ++element) {
            float value = optimizer->second_moments[i]->storage->data[
                tensor_flat_index(optimizer->second_moments[i], element)];
            if (!isfinite(value) || value < 0.0f) return 0;
        }
        if (!write_string(file, optimizer->parameters[i]->name) ||
            !write_u64(file, optimizer->steps[i]) ||
            !write_u64(file, (uint64_t)tensor_numel(
                optimizer->first_moments[i])) ||
            !write_tensor_values(file, optimizer->first_moments[i]) ||
            !write_tensor_values(file, optimizer->second_moments[i])) {
            return 0;
        }
    }
    return 1;
}

static char* temporary_path(const char* path)
{
    static const char suffix[] = ".tmp";
    size_t length;
    char* result;

    if (path == NULL || path[0] == '\0') return NULL;
    length = strlen(path);
    if (length > SIZE_MAX - sizeof(suffix)) return NULL;
    result = (char*)malloc(length + sizeof(suffix));
    if (result != NULL) {
        memcpy(result, path, length);
        memcpy(result + length, suffix, sizeof(suffix));
    }
    return result;
}

static int replace_file(const char* temporary, const char* destination)
{
#ifdef _WIN32
    return MoveFileExA(temporary, destination,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(temporary, destination) == 0;
#endif
}

int nn_checkpoint_save(const char* path,
                       const nn_module* module,
                       const nn_adamw* optimizer,
                       const nn_rng* rng)
{
    char* temp_path;
    FILE* file;
    uint32_t flags = 0;
    size_t count;
    int success = 0;

    if (!module_valid(module) || !parameters_unique_and_valid(module) ||
        (optimizer != NULL && !optimizer_matches(optimizer, module))) {
        return -1;
    }
    temp_path = temporary_path(path);
    if (temp_path == NULL) return -1;
    remove(temp_path);
    file = fopen(temp_path, "wb");
    if (file == NULL) {
        free(temp_path);
        return -1;
    }
    if (optimizer != NULL) flags |= CHECKPOINT_FLAG_ADAMW;
    if (rng != NULL) flags |= CHECKPOINT_FLAG_RNG;
    count = nn_module_parameter_count(module);
    if (!write_bytes(file, checkpoint_magic, sizeof(checkpoint_magic)) ||
        !write_u32(file, CHECKPOINT_VERSION) ||
        !write_u32(file, flags) ||
        !write_u32(file, (uint32_t)count)) {
        goto cleanup;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!write_parameter(file, nn_module_parameter_at(module, i))) {
            goto cleanup;
        }
    }
    if (optimizer != NULL && !write_optimizer(file, optimizer)) goto cleanup;
    if (rng != NULL && !write_u64(file, rng->state)) goto cleanup;
    {
        int flush_status = fflush(file);
        int close_status = fclose(file);
        file = NULL;
        if (flush_status != 0 || close_status != 0) goto cleanup;
    }
    if (!replace_file(temp_path, path)) goto cleanup;
    success = 1;

cleanup:
    if (file != NULL) fclose(file);
    if (!success) remove(temp_path);
    free(temp_path);
    return success ? 0 : -1;
}

static void free_checkpoint(saved_checkpoint* checkpoint)
{
    if (checkpoint == NULL) return;
    for (size_t i = 0; i < checkpoint->parameter_count; ++i) {
        free(checkpoint->parameters[i].second_moment);
        free(checkpoint->parameters[i].first_moment);
        free(checkpoint->parameters[i].values);
        free(checkpoint->parameters[i].dims);
        free(checkpoint->parameters[i].name);
    }
    free(checkpoint->parameters);
    memset(checkpoint, 0, sizeof(*checkpoint));
}

static int read_parameter(FILE* file, saved_parameter* parameter)
{
    uint32_t ndim;
    uint64_t serialized_count;
    size_t checked_count;

    parameter->name = read_string(file);
    if (parameter->name == NULL || !read_u32(file, &ndim) ||
        ndim > CHECKPOINT_MAX_NDIM) {
        return 0;
    }
    parameter->ndim = (int)ndim;
    if (ndim > 0) {
        parameter->dims = (int*)malloc((size_t)ndim * sizeof(*parameter->dims));
        if (parameter->dims == NULL) return 0;
    }
    for (uint32_t dim = 0; dim < ndim; ++dim) {
        uint32_t size;
        if (!read_u32(file, &size) || size == 0 || size > INT32_MAX) return 0;
        parameter->dims[dim] = (int)size;
    }
    if (!read_u64(file, &serialized_count) ||
        serialized_count > INT32_MAX ||
        !tensor_checked_numel(parameter->ndim, parameter->dims,
                              &checked_count) ||
        serialized_count != checked_count) {
        return 0;
    }
    parameter->count = checked_count;
    parameter->values =
        (float*)malloc(parameter->count * sizeof(*parameter->values));
    if (parameter->values == NULL) return 0;
    for (size_t i = 0; i < parameter->count; ++i) {
        if (!read_f32(file, &parameter->values[i]) ||
            !isfinite(parameter->values[i])) {
            return 0;
        }
    }
    return 1;
}

static saved_parameter* find_saved_parameter(saved_checkpoint* checkpoint,
                                             const char* name)
{
    for (size_t i = 0; i < checkpoint->parameter_count; ++i) {
        if (strcmp(checkpoint->parameters[i].name, name) == 0) {
            return &checkpoint->parameters[i];
        }
    }
    return NULL;
}

static int read_optimizer(FILE* file, saved_checkpoint* checkpoint)
{
    uint32_t count;
    nn_adamw_config* config = &checkpoint->config;

    if (!read_f32(file, &config->learning_rate) ||
        !read_f32(file, &config->beta1) ||
        !read_f32(file, &config->beta2) ||
        !read_f32(file, &config->epsilon) ||
        !read_f32(file, &config->weight_decay) ||
        !read_f32(file, &config->max_grad_norm) ||
        !adamw_config_valid(config) ||
        !read_u32(file, &count) || count != checkpoint->parameter_count) {
        return 0;
    }
    for (uint32_t i = 0; i < count; ++i) {
        char* name = read_string(file);
        saved_parameter* parameter;
        uint64_t element_count;

        if (name == NULL) return 0;
        parameter = find_saved_parameter(checkpoint, name);
        free(name);
        if (parameter == NULL || parameter->first_moment != NULL ||
            !read_u64(file, &parameter->step) ||
            !read_u64(file, &element_count) ||
            element_count != parameter->count) {
            return 0;
        }
        parameter->first_moment =
            (float*)malloc(parameter->count * sizeof(float));
        parameter->second_moment =
            (float*)malloc(parameter->count * sizeof(float));
        if (parameter->first_moment == NULL ||
            parameter->second_moment == NULL) {
            return 0;
        }
        for (size_t element = 0; element < parameter->count; ++element) {
            if (!read_f32(file, &parameter->first_moment[element]) ||
                !isfinite(parameter->first_moment[element])) {
                return 0;
            }
        }
        for (size_t element = 0; element < parameter->count; ++element) {
            if (!read_f32(file, &parameter->second_moment[element]) ||
                !isfinite(parameter->second_moment[element]) ||
                parameter->second_moment[element] < 0.0f) {
                return 0;
            }
        }
    }
    return 1;
}

static int read_new_checkpoint(FILE* file, saved_checkpoint* checkpoint)
{
    uint32_t version;
    uint32_t parameter_count;

    if (!read_u32(file, &version) || version != CHECKPOINT_VERSION ||
        !read_u32(file, &checkpoint->flags) ||
        (checkpoint->flags & ~(CHECKPOINT_FLAG_ADAMW |
                               CHECKPOINT_FLAG_RNG)) != 0 ||
        !read_u32(file, &parameter_count)) {
        return 0;
    }
    checkpoint->parameter_count = parameter_count;
    if (parameter_count > 0) {
        checkpoint->parameters = (saved_parameter*)calloc(
            parameter_count, sizeof(*checkpoint->parameters));
        if (checkpoint->parameters == NULL) return 0;
    }
    for (uint32_t i = 0; i < parameter_count; ++i) {
        if (!read_parameter(file, &checkpoint->parameters[i])) return 0;
        for (uint32_t earlier = 0; earlier < i; ++earlier) {
            if (strcmp(checkpoint->parameters[i].name,
                       checkpoint->parameters[earlier].name) == 0) {
                return 0;
            }
        }
    }
    if ((checkpoint->flags & CHECKPOINT_FLAG_ADAMW) != 0 &&
        !read_optimizer(file, checkpoint)) {
        return 0;
    }
    if ((checkpoint->flags & CHECKPOINT_FLAG_RNG) != 0 &&
        !read_u64(file, &checkpoint->rng_state)) {
        return 0;
    }
    return fgetc(file) == EOF && !ferror(file);
}

static int checkpoint_matches_live(const saved_checkpoint* checkpoint,
                                   nn_module* module,
                                   nn_adamw* optimizer,
                                   nn_rng* rng)
{
    size_t live_count;

    if (!module_valid(module) || !parameters_unique_and_valid(module) ||
        (((checkpoint->flags & CHECKPOINT_FLAG_ADAMW) != 0) !=
         (optimizer != NULL)) ||
        (((checkpoint->flags & CHECKPOINT_FLAG_RNG) != 0) != (rng != NULL)) ||
        (optimizer != NULL && !optimizer_matches(optimizer, module))) {
        return 0;
    }
    live_count = nn_module_parameter_count(module);
    if (live_count != checkpoint->parameter_count) return 0;
    for (size_t i = 0; i < live_count; ++i) {
        nn_parameter* live = nn_module_parameter_at(module, i);
        saved_parameter* saved = find_saved_parameter(
            (saved_checkpoint*)checkpoint, live->name);
        if (saved == NULL || live->value->value->ndim != saved->ndim ||
            (optimizer != NULL &&
             (saved->first_moment == NULL ||
              saved->second_moment == NULL))) {
            return 0;
        }
        for (int dim = 0; dim < saved->ndim; ++dim) {
            if (live->value->value->dims[dim] != saved->dims[dim]) return 0;
        }
    }
    return 1;
}

static void apply_values(tensor* destination,
                         const float* values,
                         size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        destination->storage->data[
            tensor_flat_index(destination, (int)i)] = values[i];
    }
    tensor_mark_modified(destination);
}

static void apply_checkpoint(const saved_checkpoint* checkpoint,
                             nn_module* module,
                             nn_adamw* optimizer,
                             nn_rng* rng)
{
    for (size_t i = 0; i < checkpoint->parameter_count; ++i) {
        const saved_parameter* saved = &checkpoint->parameters[i];
        nn_parameter* live = NULL;
        size_t live_index = 0;

        for (; live_index < nn_module_parameter_count(module); ++live_index) {
            nn_parameter* candidate =
                nn_module_parameter_at(module, live_index);
            if (strcmp(candidate->name, saved->name) == 0) {
                live = candidate;
                break;
            }
        }
        apply_values(live->value->value, saved->values, saved->count);
        if (optimizer != NULL) {
            optimizer->steps[live_index] = saved->step;
            apply_values(optimizer->first_moments[live_index],
                         saved->first_moment, saved->count);
            apply_values(optimizer->second_moments[live_index],
                         saved->second_moment, saved->count);
        }
    }
    if (optimizer != NULL) optimizer->config = checkpoint->config;
    if (rng != NULL) rng->state = checkpoint->rng_state;
}

int nn_checkpoint_load(const char* path,
                       nn_module* module,
                       nn_adamw* optimizer,
                       nn_rng* rng)
{
    unsigned char magic[8];
    saved_checkpoint checkpoint = {0};
    FILE* file;
    int success = 0;

    if (path == NULL || path[0] == '\0') return -1;
    file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (!read_bytes(file, magic, sizeof(magic))) goto cleanup;
    if (memcmp(magic, checkpoint_magic, sizeof(magic)) == 0) {
        if (!read_new_checkpoint(file, &checkpoint)) goto cleanup;
    } else if (memcmp(magic, legacy_magic, sizeof(magic)) == 0) {
        uint32_t version;
        uint32_t count;
        if (optimizer != NULL || rng != NULL ||
            !read_u32(file, &version) || version != 1 ||
            !read_u32(file, &count)) {
            goto cleanup;
        }
        checkpoint.parameter_count = count;
        if (count > 0) {
            checkpoint.parameters = (saved_parameter*)calloc(
                count, sizeof(*checkpoint.parameters));
            if (checkpoint.parameters == NULL) goto cleanup;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (!read_parameter(file, &checkpoint.parameters[i])) goto cleanup;
            for (uint32_t earlier = 0; earlier < i; ++earlier) {
                if (strcmp(checkpoint.parameters[i].name,
                           checkpoint.parameters[earlier].name) == 0) {
                    goto cleanup;
                }
            }
        }
        if (fgetc(file) != EOF || ferror(file)) goto cleanup;
    } else {
        goto cleanup;
    }
    if (!checkpoint_matches_live(
            &checkpoint, module, optimizer, rng)) {
        goto cleanup;
    }
    apply_checkpoint(&checkpoint, module, optimizer, rng);
    success = 1;

cleanup:
    fclose(file);
    free_checkpoint(&checkpoint);
    return success ? 0 : -1;
}
