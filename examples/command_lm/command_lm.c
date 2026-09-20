#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <tensorlib/nn.h>
#include "command_dataset.h"
#include "command_tokenizer.h"

enum {
    COMMAND_CONTEXT = 256,
    COMMAND_CHANNELS = 512,
    COMMAND_HEADS = 8,
    COMMAND_LAYERS = 9
};

typedef struct {
    const char* corpus_path;
    const char* tokenizer_path;
    const char* checkpoint_path;
    const char* prompt;
    int generate_corpus;
    int steps;
    int batch_size;
    int eval_interval;
    int eval_batches;
    int log_interval;
    int generate_count;
    int threads;
    int resume;
    uint64_t seed;
    float learning_rate;
} command_options;

static void usage(const char* program)
{
    printf("Usage: %s CORPUS [options]\n\n"
           "Train a ~30M-parameter structured developer-command model.\n\n"
           "Options:\n"
           "  --generate-corpus N    create N synthetic records if missing\n"
           "  --tokenizer PATH       tokenizer file (default: command.tok)\n"
           "  --checkpoint PATH      checkpoint file (default: command.chk)\n"
           "  --resume               explicitly resume a compatible checkpoint\n"
           "  --steps N              updates (default: 50000)\n"
           "  --batch-size N         sequences per update (default: 16)\n"
           "  --threads N            OpenMP threads (default: 16)\n"
           "  --eval-interval N      validation/checkpoint interval (default: 500)\n"
           "  --eval-batches N       validation batches (default: 8)\n"
           "  --log-interval N      throughput log interval (default: 10)\n"
           "  --learning-rate X      AdamW rate (default: 0.0003)\n"
           "  --prompt TEXT          request to translate\n"
           "  --generate N            output tokens after training (default: 64)\n"
           "  --seed N               deterministic seed (default: 20260920)\n"
           "  --help                 show this message\n", program);
}

static int parse_int(const char* text, int minimum, int* result)
{
    char* end;
    long value;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || text[0] == '\0' || end[0] != '\0' ||
        value < minimum || value > INT32_MAX) return -1;
    *result = (int)value;
    return 0;
}

static int parse_u64(const char* text, uint64_t* result)
{
    char* end;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || text[0] == '\0' || end[0] != '\0') return -1;
    *result = (uint64_t)value;
    return 0;
}

static int parse_float(const char* text, float* result)
{
    char* end;
    double value;
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || text[0] == '\0' || end[0] != '\0' ||
        !isfinite(value) || value <= 0.0 || value > 1.0) return -1;
    *result = (float)value;
    return 0;
}

static int next_value(int argc, char** argv, int* index, const char** value)
{
    if (*index + 1 >= argc) return -1;
    *value = argv[++*index];
    return 0;
}

static double wall_seconds(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static int parse_options(int argc, char** argv, command_options* options)
{
    const char* value;
    memset(options, 0, sizeof(*options));
    options->tokenizer_path = "command.tok";
    options->checkpoint_path = "command.chk";
    options->steps = 50000;
    options->batch_size = 16;
    options->threads = 16;
    options->eval_interval = 500;
    options->eval_batches = 8;
    options->log_interval = 10;
    options->generate_count = 64;
    options->learning_rate = 3e-4f;
    options->seed = UINT64_C(20260920);
    for (int index = 1; index < argc; ++index) {
        const char* argument = argv[index];
        if (strcmp(argument, "--help") == 0) return 1;
        if (strcmp(argument, "--generate-corpus") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->generate_corpus) != 0) return -1;
        } else if (strcmp(argument, "--tokenizer") == 0) {
            if (next_value(argc, argv, &index, &options->tokenizer_path) != 0) return -1;
        } else if (strcmp(argument, "--checkpoint") == 0) {
            if (next_value(argc, argv, &index, &options->checkpoint_path) != 0) return -1;
        } else if (strcmp(argument, "--resume") == 0) {
            options->resume = 1;
        } else if (strcmp(argument, "--steps") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 0, &options->steps) != 0) return -1;
        } else if (strcmp(argument, "--batch-size") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->batch_size) != 0) return -1;
        } else if (strcmp(argument, "--threads") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->threads) != 0) return -1;
        } else if (strcmp(argument, "--eval-interval") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->eval_interval) != 0) return -1;
        } else if (strcmp(argument, "--eval-batches") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->eval_batches) != 0) return -1;
        } else if (strcmp(argument, "--log-interval") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->log_interval) != 0) return -1;
        } else if (strcmp(argument, "--learning-rate") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_float(value, &options->learning_rate) != 0) return -1;
        } else if (strcmp(argument, "--prompt") == 0) {
            if (next_value(argc, argv, &index, &options->prompt) != 0) return -1;
        } else if (strcmp(argument, "--generate") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 0, &options->generate_count) != 0) return -1;
        } else if (strcmp(argument, "--seed") == 0) {
            if (next_value(argc, argv, &index, &value) != 0 ||
                parse_u64(value, &options->seed) != 0) return -1;
        } else if (argument[0] == '-' || options->corpus_path != NULL) {
            return -1;
        } else {
            options->corpus_path = argument;
        }
    }
    return options->corpus_path == NULL ? -1 : 0;
}

static size_t parameter_count(const nn_module* module)
{
    size_t total = 0;
    size_t count = nn_module_parameter_count(module);
    for (size_t index = 0; index < count; ++index) {
        nn_parameter* parameter = nn_module_parameter_at(module, index);
        int values = tensor_numel(parameter->value->value);
        if (values < 0 || (size_t)values > SIZE_MAX - total) return SIZE_MAX;
        total += (size_t)values;
    }
    return total;
}

typedef struct {
    uint16_t* ids;
    size_t length;
    size_t train_length;
} token_stream;

static int tokenize_corpus(const command_corpus* corpus,
                           const command_tokenizer* tokenizer,
                           token_stream* stream)
{
    size_t capacity;
    if (corpus == NULL || tokenizer == NULL || stream == NULL) return -1;
    memset(stream, 0, sizeof(*stream));
    capacity = corpus->size == 0 ? 1 : corpus->size;
    stream->ids = (uint16_t*)malloc(capacity * sizeof(*stream->ids));
    if (stream->ids == NULL || command_tokenizer_encode(
            tokenizer, corpus->bytes, corpus->size, stream->ids, capacity,
            &stream->length) != 0) {
        free(stream->ids);
        memset(stream, 0, sizeof(*stream));
        return -1;
    }
    stream->train_length = stream->length * 9 / 10;
    return stream->train_length > COMMAND_CONTEXT + 1 &&
           stream->length - stream->train_length > COMMAND_CONTEXT + 1 ? 0 : -1;
}

static void destroy_tokens(token_stream* stream)
{
    if (stream == NULL) return;
    free(stream->ids);
    memset(stream, 0, sizeof(*stream));
}

static size_t random_start(nn_rng* rng, size_t available)
{
    double sample = nn_rng_uniform(rng, 0.0f, 1.0f);
    size_t result = (size_t)(sample * (double)available);
    return result < available ? result : available - 1;
}

static int make_batch(const token_stream* stream, int training, int batch_size,
                      int batch_index, nn_rng* rng, ag_tensor** inputs,
                      tensor** targets)
{
    size_t region_start;
    size_t region_length;
    size_t available;
    int dims[2] = {batch_size, COMMAND_CONTEXT};
    tensor* input_values = NULL;
    tensor* target_values = NULL;
    if (stream == NULL || inputs == NULL || targets == NULL ||
        batch_size <= 0) return -1;
    region_start = training ? 0 : stream->train_length;
    region_length = training ? stream->train_length
                             : stream->length - stream->train_length;
    if (region_length <= COMMAND_CONTEXT + 1) return -1;
    available = region_length - COMMAND_CONTEXT - 1;
    *inputs = NULL;
    *targets = NULL;
    input_values = t_alloc(2, dims);
    target_values = t_alloc(2, dims);
    if (input_values == NULL || target_values == NULL) goto fail;
    for (int row = 0; row < batch_size; ++row) {
        size_t start = training
            ? random_start(rng, available)
            : (((size_t)batch_index * (size_t)batch_size + (size_t)row) *
               COMMAND_CONTEXT) % available;
        start += region_start;
        for (int column = 0; column < COMMAND_CONTEXT; ++column) {
            size_t offset = (size_t)row * COMMAND_CONTEXT + (size_t)column;
            input_values->storage->data[offset] = (float)stream->ids[start + (size_t)column];
            target_values->storage->data[offset] = (float)stream->ids[start + (size_t)column + 1];
        }
    }
    *inputs = ag_from_owned_tensor(input_values, 0);
    input_values = NULL;
    *targets = target_values;
    return *inputs == NULL ? -1 : 0;
fail:
    t_free(input_values);
    t_free(target_values);
    return -1;
}

static int evaluate(nn_decoder* model, const token_stream* stream,
                    int batch_size, int batches, float* result)
{
    double total = 0.0;
    ag_tensor* inputs = NULL;
    tensor* targets = NULL;
    ag_tensor* loss = NULL;
    nn_module_set_training(&model->base, 0);
    for (int batch = 0; batch < batches; ++batch) {
        inputs = NULL;
        targets = NULL;
        loss = NULL;
        if (make_batch(stream, 0, batch_size, batch, NULL, &inputs, &targets) != 0) goto fail;
        loss = nn_decoder_loss(model, inputs, targets);
        if (loss == NULL) goto fail;
        total += loss->value->storage->data[loss->value->offset];
        ag_tensor_release(loss);
        ag_tensor_release(inputs);
        t_free(targets);
        loss = NULL;
        inputs = NULL;
        targets = NULL;
    }
    nn_module_set_training(&model->base, 1);
    *result = (float)(total / (double)batches);
    return 0;
fail:
    ag_tensor_release(loss);
    ag_tensor_release(inputs);
    t_free(targets);
    nn_module_set_training(&model->base, 1);
    return -1;
}

static int safe_command_text(const unsigned char* text, size_t length)
{
    static const char* denied[] = {"rm", "rmdir", "mv", "cp", "chmod", "chown",
        "dd", "mkfs", "shred", "kill", "sudo", "su", "ssh", "scp", "curl",
        "wget", "nc", "bash", "sh", "python", "perl", "ruby", "eval", "exec",
        "install", "uninstall", "remove", "delete", "destroy", "purge", "erase",
        "wipe", "format", "push", "reset", "rebase", "checkout", "clone", "commit",
        "merge", "pull", "apply", "upgrade", "update", "start", "stop", "restart",
        "enable", "disable", "deploy", "create", "drop", "truncate", "terminate",
        "revoke", "grant", "upload", "download"};
    const unsigned char marker[] = "COMMAND: ";
    const unsigned char* start = NULL;
    for (size_t index = 0; index + sizeof(marker) - 1 <= length; ++index) {
        if (memcmp(text + index, marker, sizeof(marker) - 1) == 0) {
            start = text + index + sizeof(marker) - 1;
            break;
        }
    }
    if (start == NULL) return 0;
    for (const unsigned char* cursor = start; cursor < text + length; ++cursor) {
        if (*cursor == ';' || *cursor == '&' || *cursor == '$' || *cursor == '`' ||
            *cursor == '<' || *cursor == '>' || *cursor == '(' || *cursor == ')') return 0;
    }
    for (size_t item = 0; item < sizeof(denied) / sizeof(denied[0]); ++item) {
        size_t width = strlen(denied[item]);
        for (const unsigned char* cursor = start; cursor + width <= text + length; ++cursor) {
            if ((cursor == start || cursor[-1] == ' ' || cursor[-1] == '|') &&
                memcmp(cursor, denied[item], width) == 0 &&
                (cursor + width == text + length || cursor[width] == ' ' || cursor[width] == '|')) return 0;
        }
    }
    return start < text + length && *start > ' ';
}

static int generate_output(nn_decoder* model, const command_tokenizer* tokenizer,
                           const char* prompt, int count)
{
    uint16_t ids[COMMAND_CONTEXT];
    unsigned char decoded[COMMAND_CONTEXT * COMMAND_TOKENIZER_MAX_TOKEN_BYTES + 1];
    size_t length = 0;
    if (command_tokenizer_encode(tokenizer, (const unsigned char*)prompt,
                                 strlen(prompt), ids, COMMAND_CONTEXT,
                                 &length) != 0 || length == 0) return -1;
    nn_module_set_training(&model->base, 0);
    for (int step = 0; step < count; ++step) {
        int dims[2] = {1, (int)length};
        tensor* values = t_alloc(2, dims);
        ag_tensor* input = NULL;
        ag_tensor* logits = NULL;
        uint16_t next = 0;
        float best = -INFINITY;
        if (values == NULL) goto fail;
        for (size_t index = 0; index < length; ++index) {
            values->storage->data[index] = (float)ids[index];
        }
        input = ag_from_owned_tensor(values, 0);
        if (input == NULL) goto fail;
        logits = nn_decoder_forward(model, input);
        if (logits == NULL) {
            ag_tensor_release(input);
            goto fail;
        }
        for (int token = 0; token < COMMAND_TOKENIZER_VOCAB; ++token) {
            int offset = logits->value->offset + (int)(length - 1) * logits->value->strides[1] +
                         token * logits->value->strides[2];
            float value = logits->value->storage->data[offset];
            if (value > best) { best = value; next = (uint16_t)token; }
        }
        ag_tensor_release(logits);
        ag_tensor_release(input);
        if (length == COMMAND_CONTEXT) memmove(ids, ids + 1, (length - 1) * sizeof(*ids));
        if (length < COMMAND_CONTEXT) ++length;
        ids[length - 1] = next;
    }
    if (command_tokenizer_decode(tokenizer, ids, length, decoded, sizeof(decoded),
                                 &length) != 0) goto fail;
    printf("\nGenerated request/command:\n%.*s\n", (int)length, decoded);
    printf("Command validation: %s\n", safe_command_text(decoded, length) ? "accepted" : "rejected");
    nn_module_set_training(&model->base, 1);
    return 0;
fail:
    nn_module_set_training(&model->base, 1);
    return -1;
}

int main(int argc, char** argv)
{
    command_options options;
    command_corpus corpus = {0};
    command_tokenizer tokenizer = {0};
    token_stream stream = {0};
    nn_decoder_config model_config;
    nn_adamw_config optimizer_config;
    nn_rng rng;
    nn_decoder* model = NULL;
    nn_adamw* optimizer = NULL;
    int parse_status;
    int status = EXIT_FAILURE;

    parse_status = parse_options(argc, argv, &options);
    if (parse_status != 0) {
        usage(argv[0]);
        return parse_status > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(options.threads);
#endif
    if (options.generate_corpus > 0) {
        if (command_corpus_generate(&corpus, (size_t)options.generate_corpus,
                                     options.seed) != 0 ||
            command_corpus_save(&corpus, options.corpus_path) != 0) {
            fprintf(stderr, "Could not generate corpus '%s'.\n", options.corpus_path);
            return EXIT_FAILURE;
        }
    } else if (command_corpus_load(&corpus, options.corpus_path) != 0) {
        fprintf(stderr, "Could not load corpus '%s'.\n", options.corpus_path);
        return EXIT_FAILURE;
    }
    command_tokenizer_init(&tokenizer);
    if (command_tokenizer_load(&tokenizer, options.tokenizer_path) == 0 &&
        !options.resume) {
        fprintf(stderr, "Tokenizer exists; use --resume or a new path.\n");
        goto cleanup;
    }
    if (tokenizer.merge_count == 0) {
        if (command_tokenizer_train(&tokenizer, corpus.bytes, corpus.train_size,
                                    COMMAND_TOKENIZER_VOCAB) != 0 ||
            command_tokenizer_save(&tokenizer, options.tokenizer_path) != 0) {
            fprintf(stderr, "Could not prepare tokenizer '%s'.\n", options.tokenizer_path);
            goto cleanup;
        }
    }
    if (tokenize_corpus(&corpus, &tokenizer, &stream) != 0) {
        fprintf(stderr, "Corpus is too short after tokenization.\n");
        goto cleanup;
    }
    model_config.vocabulary_size = COMMAND_TOKENIZER_VOCAB;
    model_config.context_length = COMMAND_CONTEXT;
    model_config.channels = COMMAND_CHANNELS;
    model_config.head_count = COMMAND_HEADS;
    model_config.layer_count = COMMAND_LAYERS;
    model_config.dropout_probability = 0.1f;
    model_config.layer_norm_epsilon = 1e-5f;
    optimizer_config = nn_adamw_default_config();
    optimizer_config.learning_rate = options.learning_rate;
    optimizer_config.weight_decay = 0.01f;
    optimizer_config.max_grad_norm = 1.0f;
    nn_rng_seed(&rng, options.seed);
    model = nn_decoder_create("command_lm", &model_config, &rng);
    optimizer = model == NULL ? NULL : nn_adamw_create(&model->base, &optimizer_config);
    if (model == NULL || optimizer == NULL) {
        fprintf(stderr, "Could not allocate command model.\n");
        goto cleanup;
    }
    if (options.resume) {
        if (nn_checkpoint_load(options.checkpoint_path, &model->base,
                               optimizer, &rng) != 0) {
            fprintf(stderr, "Could not resume checkpoint '%s'.\n",
                    options.checkpoint_path);
            goto cleanup;
        }
        printf("Resumed checkpoint: %s\n", options.checkpoint_path);
    } else {
        FILE* existing = fopen(options.checkpoint_path, "rb");
        if (existing != NULL) {
            fclose(existing);
            fprintf(stderr, "Checkpoint exists; use --resume or a new path.\n");
            goto cleanup;
        }
        printf("Starting a new model.\n");
    }
    printf("Corpus: %zu bytes, %zu tokens; model parameters: %zu (%.2fM)\n",
           corpus.size, stream.length, parameter_count(&model->base),
           (double)parameter_count(&model->base) / 1e6);
    printf("Training: steps=%d batch=%d threads=%d lr=%.6g\n",
           options.steps, options.batch_size, options.threads,
           (double)options.learning_rate);
    for (int step = 1; step <= options.steps; ++step) {
        ag_backward_options backward_options = ag_backward_default_options();
        ag_tensor* inputs = NULL;
        tensor* targets = NULL;
        ag_tensor* loss = NULL;
        float value;
        double started = wall_seconds();
        nn_adamw_zero_grad(optimizer);
        backward_options.retention = AG_GRAD_RETAIN_LEAVES;
        if (make_batch(&stream, 1, options.batch_size, step, &rng,
                       &inputs, &targets) != 0 ||
            (loss = nn_decoder_loss(model, inputs, targets)) == NULL ||
            ag_backward_ex(loss, &backward_options) != 0 ||
            nn_adamw_step(optimizer) != 0) {
            fprintf(stderr, "Training failed at step %d.\n", step);
            ag_tensor_release(loss); ag_tensor_release(inputs); t_free(targets);
            goto cleanup;
        }
        value = loss->value->storage->data[loss->value->offset];
        ag_tensor_release(loss); ag_tensor_release(inputs); t_free(targets);
        {
            double seconds = wall_seconds() - started;
            double tokens_per_second = seconds > 0.0
                ? (double)options.batch_size * COMMAND_CONTEXT / seconds : 0.0;
            if (step == 1 || step % options.log_interval == 0) {
                printf("step %d/%d train_loss=%.5f step_time=%.3fs tokens/s=%.1f\n",
                       step, options.steps, (double)value, seconds,
                       tokens_per_second);
            }
        }
        if (step == 1 || step % options.eval_interval == 0) {
            float validation;
            if (evaluate(model, &stream, options.batch_size,
                         options.eval_batches, &validation) != 0 ||
                nn_checkpoint_save(options.checkpoint_path, &model->base,
                                   optimizer, &rng) != 0) goto cleanup;
            printf("step %d/%d train_loss=%.5f validation_loss=%.5f\n",
                   step, options.steps, (double)value, (double)validation);
        }
    }
    if (options.prompt == NULL) {
        options.prompt = "REQUEST: describe a safe command\nCOMMAND: ";
    }
    if (options.generate_count > 0 &&
        generate_output(model, &tokenizer, options.prompt,
                        options.generate_count) != 0) {
        fprintf(stderr, "Generation failed.\n");
        goto cleanup;
    }
    status = EXIT_SUCCESS;
cleanup:
    nn_adamw_destroy(optimizer);
    nn_decoder_destroy(model);
    destroy_tokens(&stream);
    command_tokenizer_destroy(&tokenizer);
    command_corpus_destroy(&corpus);
    return status;
}
