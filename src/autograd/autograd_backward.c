#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "./../../include/tensorlib/autograd_internal.h"
#include "../tensor/parallel.h"

#define AG_REDUCTION_MIN_PARALLEL_ELEMENTS (1 << 20)

typedef struct {
    ag_tensor** values;
    int count;
    int capacity;
} tensor_list;

typedef struct {
    ag_node** values;
    int count;
    int capacity;
} node_list;

static int backward_stats_enabled;
static ag_backward_stats backward_stats;

static double backward_now(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
#endif
}

static double backward_elapsed(double started)
{
    double elapsed = backward_now() - started;
    return elapsed > 0.0 ? elapsed : 0.0;
}

static void sum_contiguous_suffix(const float* source,
                                  float* destination,
                                  int outer_count,
                                  int result_count,
                                  float scale)
{
    int tasks = outer_count > result_count ? outer_count : result_count;
    int threads = tensorlib_parallel_threads(
        (long long)outer_count * result_count,
        AG_REDUCTION_MIN_PARALLEL_ELEMENTS, tasks);
#ifndef _OPENMP
    (void)threads;
#endif
    if (threads <= 1) {
        for (int outer = 0; outer < outer_count; ++outer) {
            for (int index = 0; index < result_count; ++index) {
                destination[index] +=
                    scale * source[outer * result_count + index];
            }
        }
        return;
    }
#ifdef _OPENMP
    if (outer_count >= threads * 4 && result_count <= (1 << 14)) {
        float* partials = (float*)calloc(
            (size_t)threads * (size_t)result_count, sizeof(float));
        if (partials != NULL) {
#pragma omp parallel num_threads(threads)
            {
                int thread = omp_get_thread_num();
                float* partial = partials +
                    (size_t)thread * (size_t)result_count;
#pragma omp for schedule(static)
                for (int outer = 0; outer < outer_count; ++outer) {
                    for (int index = 0; index < result_count; ++index) {
                        partial[index] +=
                            source[outer * result_count + index];
                    }
                }
#pragma omp for schedule(static)
                for (int index = 0; index < result_count; ++index) {
                    float sum = 0.0f;
                    for (int owner = 0; owner < threads; ++owner) {
                        sum += partials[(size_t)owner * (size_t)result_count +
                                        (size_t)index];
                    }
                    destination[index] = scale * sum;
                }
            }
            free(partials);
            return;
        }
    }
#pragma omp parallel for if(threads > 1) schedule(static) num_threads(threads)
    for (int index = 0; index < result_count; ++index) {
        float sum = 0.0f;
        for (int outer = 0; outer < outer_count; ++outer) {
            sum += source[outer * result_count + index];
        }
        destination[index] = scale * sum;
    }
#endif
}

static int contiguous_reduction_layout(const tensor* source,
                                       const tensor* target,
                                       int* prefix_count,
                                       int* reduction_count,
                                       int* suffix_count)
{
    if (!is_contiguous((tensor*)source)) return 0;
    int rank_offset = source->ndim - target->ndim;
    int phase = 0;
    *prefix_count = 1;
    *reduction_count = 1;
    *suffix_count = 1;
    for (int axis = 0; axis < source->ndim; ++axis) {
        int target_axis = axis - rank_offset;
        int preserved = target_axis >= 0 &&
            target->dims[target_axis] == source->dims[axis];
        int reduced = target_axis < 0 || target->dims[target_axis] == 1;
        if (!preserved && !reduced) return 0;
        if (preserved) {
            if (phase == 1) phase = 2;
            if (phase == 0) *prefix_count *= source->dims[axis];
            else *suffix_count *= source->dims[axis];
        } else {
            if (phase == 2) return 0;
            phase = 1;
            *reduction_count *= source->dims[axis];
        }
    }
    return *reduction_count > 1;
}

static void sum_contiguous_reduction(const float* source,
                                     float* destination,
                                     int prefix_count,
                                     int reduction_count,
                                     int suffix_count,
                                     float scale)
{
    for (int prefix = 0; prefix < prefix_count; ++prefix) {
        sum_contiguous_suffix(
            source + (size_t)prefix * (size_t)reduction_count *
                (size_t)suffix_count,
            destination + (size_t)prefix * (size_t)suffix_count,
            reduction_count, suffix_count, scale);
    }
}

void ag_backward_stats_enable(int enabled)
{
    backward_stats_enabled = enabled != 0;
}

void ag_backward_stats_reset(void)
{
    memset(&backward_stats, 0, sizeof(backward_stats));
}

void ag_backward_stats_read(ag_backward_stats* output)
{
    if (output != NULL) *output = backward_stats;
}

void ag_backward_stats_record_matmul(int packed_dinput, int fast_dweight)
{
    if (!backward_stats_enabled) return;
    if (packed_dinput >= 0) {
        if (packed_dinput) ++backward_stats.matmul_packed_dinput;
        else ++backward_stats.matmul_generic_dinput;
    }
    if (fast_dweight >= 0) {
        if (fast_dweight) ++backward_stats.matmul_fast_dweight;
        else ++backward_stats.matmul_fallback_dweight;
    }
}

static int append_tensor(tensor_list* list, ag_tensor* value) {
    if (list->count == list->capacity) {
        int capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        ag_tensor** values = (ag_tensor**)realloc(list->values,
                                                  (size_t)capacity * sizeof(*values));
        if (values == NULL) return 1;
        list->values = values;
        list->capacity = capacity;
    }
    list->values[list->count++] = value;
    return 0;
}

static int append_node(node_list* list, ag_node* node) {
    if (list->count == list->capacity) {
        int capacity = list->capacity == 0 ? 16 : list->capacity * 2;
        ag_node** values = (ag_node**)realloc(list->values,
                                              (size_t)capacity * sizeof(*values));
        if (values == NULL) return 1;
        list->values = values;
        list->capacity = capacity;
    }
    list->values[list->count++] = node;
    return 0;
}

static int collect_graph(ag_tensor* value, tensor_list* tensors, node_list* nodes) {
    if (value == NULL || value->graph_index >= 0) return value == NULL;
    value->graph_index = tensors->count;
    if (append_tensor(tensors, value) != 0) return 1;
    if (value->creator == NULL) return 0;
    for (int i = 0; i < value->creator->input_count; ++i) {
        if (collect_graph(value->creator->inputs[i], tensors, nodes) != 0) return 1;
    }
    return append_node(nodes, value->creator);
}

static int graph_versions_match(const node_list* nodes) {
    for (int node_index = 0; node_index < nodes->count; ++node_index) {
        const ag_node* node = nodes->values[node_index];
        if (node == NULL || node->output == NULL ||
            !tensor_has_valid_metadata(node->output->value) ||
            node->output->value->storage->version != node->output_version) {
            return 0;
        }
        for (int input_index = 0; input_index < node->input_count; ++input_index) {
            if (node->inputs[input_index] == NULL ||
                !tensor_has_valid_metadata(node->inputs[input_index]->value) ||
                node->inputs[input_index]->value->storage->version !=
                    node->input_versions[input_index]) {
                return 0;
            }
        }
    }
    return 1;
}

tensor* ag_sum_to_shape(const tensor* source, const tensor* target, float scale)
{
    if (!tensor_has_valid_metadata(source) || !tensor_has_valid_metadata(target) ||
        source->ndim < target->ndim) return NULL;
    for (int axis = 0; axis < target->ndim; ++axis) {
        int source_axis = source->ndim - target->ndim + axis;
        if (target->dims[axis] != 1 &&
            target->dims[axis] != source->dims[source_axis]) return NULL;
    }
    if (source->ndim == target->ndim &&
        same_shape((tensor*)source, (tensor*)target)) {
        if (scale == 1.0f) return t_contiguous((tensor*)source);
        if (scale == -1.0f) return t_neg((tensor*)source);
        return t_mul_scalar((tensor*)source, scale);
    }

    tensor* result = t_alloc(target->ndim, target->dims);
    if (result == NULL) return NULL;
    int result_count = tensor_numel(result);
    for (int index = 0; index < result_count; ++index) {
        result->storage->data[result->offset + index] = 0.0f;
    }

    int prefix_count;
    int reduction_count;
    int suffix_count;
    if (result_count > 0 && contiguous_reduction_layout(
            source, target, &prefix_count, &reduction_count, &suffix_count)) {
        if (backward_stats_enabled) {
            ++backward_stats.reduction_fast_calls;
            backward_stats.reduction_fast_elements +=
                (unsigned long long)tensor_numel((tensor*)source);
        }
        const float* values = source->storage->data + source->offset;
        float* destination = result->storage->data + result->offset;
        sum_contiguous_reduction(values, destination, prefix_count,
                                 reduction_count, suffix_count, scale);
        return result;
    }

    int* coords = source->ndim > 0
                ? (int*)calloc((size_t)source->ndim, sizeof(*coords)) : NULL;
    if (backward_stats_enabled) {
        ++backward_stats.reduction_generic_calls;
        backward_stats.reduction_generic_elements +=
            (unsigned long long)tensor_numel((tensor*)source);
    }
    if (source->ndim > 0 && coords == NULL) {
        t_free(result);
        return NULL;
    }
    int source_count = tensor_numel((tensor*)source);
    for (int index = 0; index < source_count; ++index) {
        int result_index = result->offset;
        for (int axis = 0; axis < target->ndim; ++axis) {
            int source_axis = source->ndim - target->ndim + axis;
            int coordinate = target->dims[axis] == 1 ? 0 : coords[source_axis];
            result_index += coordinate * result->strides[axis];
        }
        result->storage->data[result_index] += scale *
            source->storage->data[get_flat_index_nd((tensor*)source, coords)];
        advance_coords(coords, source->dims, source->ndim);
    }
    free(coords);
    return result;
}

static tensor* reduce_to_shape(tensor* contribution, const tensor* target,
                               int* coords, int coord_capacity) {
    if (!tensor_has_valid_metadata(contribution) || !tensor_has_valid_metadata(target) ||
        contribution->ndim < target->ndim) {
        t_free(contribution);
        return NULL;
    }

    if (contribution->ndim == target->ndim &&
        same_shape(contribution, (tensor*)target)) return contribution;

    for (int target_axis = 0; target_axis < target->ndim; ++target_axis) {
        int contribution_axis = contribution->ndim - target->ndim + target_axis;
        if (target->dims[target_axis] != 1 &&
            target->dims[target_axis] != contribution->dims[contribution_axis]) {
            t_free(contribution);
            return NULL;
        }
    }
    tensor* reduced = t_alloc(target->ndim, target->dims);
    if (reduced == NULL) {
        t_free(reduced);
        t_free(contribution);
        return NULL;
    }
    int reduced_count = tensor_numel(reduced);
    for (int index = 0; index < reduced_count; ++index) {
        reduced->storage->data[reduced->offset + index] = 0.0f;
    }

    int prefix_count;
    int reduction_count;
    int suffix_count;
    if (reduced_count > 0 && contiguous_reduction_layout(
            contribution, target, &prefix_count, &reduction_count,
            &suffix_count)) {
        if (backward_stats_enabled) {
            ++backward_stats.reduction_fast_calls;
            backward_stats.reduction_fast_elements +=
                (unsigned long long)tensor_numel(contribution);
        }
        const float* source = contribution->storage->data + contribution->offset;
        float* destination = reduced->storage->data + reduced->offset;
        sum_contiguous_reduction(source, destination, prefix_count,
                                 reduction_count, suffix_count, 1.0f);
        t_free(contribution);
        return reduced;
    }

    if (contribution->ndim > coord_capacity ||
        (contribution->ndim > 0 && coords == NULL)) {
        t_free(reduced);
        t_free(contribution);
        return NULL;
    }
    memset(coords, 0, (size_t)contribution->ndim * sizeof(*coords));
    if (backward_stats_enabled) {
        ++backward_stats.reduction_generic_calls;
        backward_stats.reduction_generic_elements +=
            (unsigned long long)tensor_numel(contribution);
    }

    int contribution_count = tensor_numel(contribution);
    for (int index = 0; index < contribution_count; ++index) {
        int reduced_index = reduced->offset;
        for (int target_axis = 0; target_axis < target->ndim; ++target_axis) {
            int contribution_axis = contribution->ndim - target->ndim + target_axis;
            int coordinate = target->dims[target_axis] == 1
                           ? 0 : coords[contribution_axis];
            reduced_index += coordinate * reduced->strides[target_axis];
        }
        reduced->storage->data[reduced_index] +=
            contribution->storage->data[get_flat_index_nd(contribution, coords)];
        advance_coords(coords, contribution->dims, contribution->ndim);
    }
    t_free(contribution);
    return reduced;
}

static int add_in_place(tensor* destination, const tensor* source,
                        int* coords, int coord_capacity)
{
    if (!tensor_has_valid_metadata(destination) ||
        !tensor_has_valid_metadata(source) ||
        !same_shape(destination, (tensor*)source) ||
        destination->storage == source->storage ||
        destination->storage->ref_count != 1) return 1;

    int count = tensor_numel(destination);
    if (is_contiguous(destination) && is_contiguous((tensor*)source) &&
        destination->offset == 0 && source->offset == 0) {
        for (int index = 0; index < count; ++index) {
            destination->storage->data[index] += source->storage->data[index];
        }
        return 0;
    }

    if (destination->ndim > coord_capacity ||
        (destination->ndim > 0 && coords == NULL)) return 1;
    memset(coords, 0, (size_t)destination->ndim * sizeof(*coords));
    for (int index = 0; index < count; ++index) {
        int destination_index = get_flat_index_nd(destination, coords);
        int source_index = get_flat_index_nd((tensor*)source, coords);
        destination->storage->data[destination_index] +=
            source->storage->data[source_index];
        advance_coords(coords, destination->dims, destination->ndim);
    }
    return 0;
}

static int accumulate_pass_gradient(tensor** destination,
                                    tensor* contribution,
                                    const tensor* target,
                                    int* coords,
                                    int coord_capacity) {
    double started = backward_stats_enabled ? backward_now() : 0.0;
    tensor* reduced = reduce_to_shape(contribution, target, coords,
                                      coord_capacity);
    if (backward_stats_enabled) {
        backward_stats.reduction_seconds += backward_elapsed(started);
        started = backward_now();
    }
    if (reduced == NULL) return 1;
    if (*destination == NULL) {
        *destination = reduced;
        if (backward_stats_enabled) {
            backward_stats.accumulation_seconds += backward_elapsed(started);
        }
        return 0;
    }
    if (add_in_place(*destination, reduced, coords, coord_capacity) == 0) {
        t_free(reduced);
        if (backward_stats_enabled) {
            backward_stats.accumulation_seconds += backward_elapsed(started);
        }
        return 0;
    }
    tensor* sum = t_add(*destination, reduced);
    t_free(reduced);
    if (sum == NULL) return 1;
    t_free(*destination);
    *destination = sum;
    if (backward_stats_enabled) {
        backward_stats.accumulation_seconds += backward_elapsed(started);
    }
    return 0;
}

static int merge_persistent_gradients(const tensor_list* tensors,
                                      tensor** pass_gradients,
                                      tensor** merged,
                                      ag_grad_retention retention) {
    for (int i = 0; i < tensors->count; ++i) {
        ag_tensor* value = tensors->values[i];
        if (retention == AG_GRAD_RETAIN_LEAVES && value->creator != NULL) continue;
        if (!value->requires_grad || pass_gradients[i] == NULL) continue;
        if (value->grad == NULL && pass_gradients[i]->storage->ref_count == 1 &&
            is_contiguous(pass_gradients[i]) &&
            pass_gradients[i]->offset == 0) {
            merged[i] = pass_gradients[i];
            pass_gradients[i] = NULL;
        } else {
            merged[i] = value->grad == NULL
                      ? t_clone(pass_gradients[i])
                      : t_add(value->grad, pass_gradients[i]);
        }
        if (merged[i] == NULL) {
            for (int j = 0; j < tensors->count; ++j) t_free(merged[j]);
            return 1;
        }
    }

    for (int i = 0; i < tensors->count; ++i) {
        if (merged[i] == NULL) continue;
        t_free(tensors->values[i]->grad);
        tensors->values[i]->grad = merged[i];
        merged[i] = NULL;
    }
    return 0;
}

static void free_contributions(tensor** contributions, int count)
{
    if (contributions == NULL) return;
    for (int index = 0; index < count; ++index) {
        t_free(contributions[index]);
        contributions[index] = NULL;
    }
}

ag_backward_options ag_backward_default_options(void)
{
    ag_backward_options options = {AG_GRAD_RETAIN_ALL};
    return options;
}

int ag_backward_with_grad_ex(ag_tensor* output,
                             const tensor* output_gradient,
                             const ag_backward_options* options) {
    if (options == NULL ||
        (options->retention != AG_GRAD_RETAIN_ALL &&
         options->retention != AG_GRAD_RETAIN_LEAVES)) return 1;
    if (output == NULL || !output->requires_grad ||
        !tensor_has_valid_metadata(output->value) ||
        !tensor_has_valid_metadata(output_gradient) ||
        !same_shape(output->value, (tensor*)output_gradient)) return 1;

    tensor_list tensors = {0};
    node_list nodes = {0};
    tensor** pass_gradients = NULL;
    tensor** merged_gradients = NULL;
    tensor** contributions = NULL;
    int* reduction_coords = NULL;
    void* workspace = NULL;
    int contribution_capacity = 0;
    int reduction_coord_capacity = 0;
    int status = 1;
    double started = backward_stats_enabled ? backward_now() : 0.0;

    if (collect_graph(output, &tensors, &nodes) != 0) goto cleanup;
    if (!graph_versions_match(&nodes)) goto cleanup;
    if (backward_stats_enabled) {
        backward_stats.traversal_seconds += backward_elapsed(started);
        backward_stats.graph_tensors = (unsigned long)tensors.count;
        backward_stats.graph_nodes = (unsigned long)nodes.count;
    }

    for (int index = 0; index < tensors.count; ++index) {
        int ndim = tensors.values[index]->value->ndim;
        if (ndim > reduction_coord_capacity) reduction_coord_capacity = ndim;
    }
    for (int node_index = 0; node_index < nodes.count; ++node_index) {
        if (nodes.values[node_index]->input_count > contribution_capacity) {
            contribution_capacity = nodes.values[node_index]->input_count;
        }
    }
    size_t pointer_count = (size_t)tensors.count * 2u +
                           (size_t)contribution_capacity;
    size_t workspace_bytes = pointer_count * sizeof(tensor*) +
        (size_t)reduction_coord_capacity * sizeof(int);
    workspace = calloc(1, workspace_bytes);
    if (workspace == NULL) goto cleanup;
    pass_gradients = (tensor**)workspace;
    merged_gradients = pass_gradients + tensors.count;
    contributions = merged_gradients + tensors.count;
    reduction_coords = (int*)(contributions + contribution_capacity);

    {
        int output_index = output->graph_index;
        pass_gradients[output_index] = t_clone((tensor*)output_gradient);
        if (pass_gradients[output_index] == NULL) goto cleanup;
    }

    for (int node_index = nodes.count - 1; node_index >= 0; --node_index) {
        ag_node* node = nodes.values[node_index];
        int gradient_index = node->output->graph_index;
        if (gradient_index < 0 || pass_gradients[gradient_index] == NULL) goto cleanup;

        for (int input_index = 0; input_index < node->input_count; ++input_index) {
            contributions[input_index] = NULL;
        }

        started = backward_stats_enabled ? backward_now() : 0.0;
        int backward_status =
            node->backward(node, pass_gradients[gradient_index], contributions);
        if (backward_stats_enabled) {
            int operation = (int)node->operation;
            if (operation >= 0 && operation < AG_BACKWARD_OP_COUNT) {
                backward_stats.operation_seconds[operation] +=
                    backward_elapsed(started);
                ++backward_stats.operation_calls[operation];
            }
        }
        if (backward_status != 0) {
            free_contributions(contributions, node->input_count);
            goto cleanup;
        }

        for (int input_index = 0; input_index < node->input_count; ++input_index) {
            ag_tensor* input = node->inputs[input_index];
            if (!input->requires_grad) {
                t_free(contributions[input_index]);
                contributions[input_index] = NULL;
                continue;
            }
            int destination = input->graph_index;
            if (destination < 0 || contributions[input_index] == NULL ||
                accumulate_pass_gradient(&pass_gradients[destination],
                                         contributions[input_index], input->value,
                                         reduction_coords,
                                         reduction_coord_capacity) != 0) {
                contributions[input_index] = NULL;
                free_contributions(contributions, node->input_count);
                goto cleanup;
            }
            contributions[input_index] = NULL;
        }
    }

    started = backward_stats_enabled ? backward_now() : 0.0;
    status = merge_persistent_gradients(&tensors, pass_gradients,
                                        merged_gradients, options->retention);
    if (backward_stats_enabled) {
        backward_stats.merge_seconds += backward_elapsed(started);
    }

cleanup:
    free_contributions(contributions, contribution_capacity);
    if (pass_gradients != NULL) {
        for (int i = 0; i < tensors.count; ++i) t_free(pass_gradients[i]);
    }
    free(workspace);
    for (int i = 0; i < tensors.count; ++i) tensors.values[i]->graph_index = -1;
    free(nodes.values);
    free(tensors.values);
    return status;
}

int ag_backward_with_grad(ag_tensor* output, const tensor* output_gradient)
{
    ag_backward_options options = ag_backward_default_options();
    return ag_backward_with_grad_ex(output, output_gradient, &options);
}

int ag_backward_ex(ag_tensor* loss, const ag_backward_options* options) {
    if (loss == NULL || loss->value == NULL || loss->value->ndim != 0) return 1;
    tensor* seed = ag_full_like(loss->value, 1.0f);
    if (seed == NULL) return 1;
    int status = ag_backward_with_grad_ex(loss, seed, options);
    t_free(seed);
    return status;
}

int ag_backward(ag_tensor* loss) {
    ag_backward_options options = ag_backward_default_options();
    return ag_backward_ex(loss, &options);
}

void ag_zero_grad(ag_tensor* value) {
    if (value == NULL) return;
    t_free(value->grad);
    value->grad = NULL;
}

void ag_zero_grad_all(ag_tensor* root) {
    if (root == NULL) return;
    tensor_list tensors = {0};
    node_list nodes = {0};
    if (collect_graph(root, &tensors, &nodes) == 0) {
        for (int i = 0; i < tensors.count; ++i) {
            ag_zero_grad(tensors.values[i]);
            tensors.values[i]->graph_index = -1;
        }
    }
    for (int i = 0; i < tensors.count; ++i) tensors.values[i]->graph_index = -1;
    free(nodes.values);
    free(tensors.values);
}
