#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <nn.h>

enum {
    TINY_LM_VOCABULARY = 256,
    TINY_LM_CONTEXT = 128,
    TINY_LM_CHANNELS = 192,
    TINY_LM_HEADS = 6,
    TINY_LM_LAYERS = 4
};

typedef struct {
    unsigned char* bytes;
    size_t size;
    size_t train_size;
} byte_corpus;

typedef struct {
    const char* corpus_path;
    const char* checkpoint_path;
    const char* prompt;
    int steps;
    int batch_size;
    int eval_interval;
    int eval_batches;
    int log_interval;
    int generate_count;
    int top_k;
    float temperature;
    float learning_rate;
    uint64_t seed;
    int resume;
} tiny_lm_options;

static void print_usage(const char* program)
{
    printf(
        "Usage: %s CORPUS [options]\n"
        "\n"
        "Train a ~1.9M-parameter byte-level decoder language model.\n"
        "\n"
        "Options:\n"
        "  --steps N             optimizer updates (default: 1000)\n"
        "  --batch-size N        sequences per update (default: 1)\n"
        "  --learning-rate X     AdamW learning rate (default: 0.0003)\n"
        "  --eval-interval N     validate/checkpoint every N steps (default: 100)\n"
        "  --eval-batches N      deterministic validation batches (default: 2)\n"
        "  --log-interval N      training log interval (default: 10)\n"
        "  --checkpoint PATH     checkpoint path (default: tiny_lm.chk)\n"
        "  --resume              load checkpoint before training\n"
        "  --prompt TEXT         generation prompt (default: corpus prefix)\n"
        "  --generate N          bytes to generate after training (default: 200)\n"
        "  --temperature X       0 for greedy decoding (default: 0.8)\n"
        "  --top-k N             sampling candidates, 0 disables (default: 40)\n"
        "  --seed N              deterministic seed (default: 20260726)\n"
        "  --help                show this message\n",
        program);
}

static int parse_int(const char* text, int minimum, int* result)
{
    char* end;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || text[0] == '\0' || end[0] != '\0' ||
        value < minimum || value > INT32_MAX) {
        return -1;
    }
    *result = (int)value;
    return 0;
}

static int parse_float(const char* text, float minimum, float* result)
{
    char* end;
    double value;

    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || text[0] == '\0' || end[0] != '\0' ||
        !isfinite(value) || value < minimum || value > FLT_MAX) {
        return -1;
    }
    *result = (float)value;
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

static int option_value(int argc,
                        char** argv,
                        int* index,
                        const char** result)
{
    if (*index + 1 >= argc) return -1;
    *result = argv[++*index];
    return 0;
}

static int parse_options(int argc, char** argv, tiny_lm_options* options)
{
    const char* value;

    memset(options, 0, sizeof(*options));
    options->checkpoint_path = "tiny_lm.chk";
    options->steps = 1000;
    options->batch_size = 1;
    options->eval_interval = 100;
    options->eval_batches = 2;
    options->log_interval = 10;
    options->generate_count = 200;
    options->top_k = 40;
    options->temperature = 0.8f;
    options->learning_rate = 3e-4f;
    options->seed = UINT64_C(20260726);

    for (int index = 1; index < argc; ++index) {
        const char* argument = argv[index];
        if (strcmp(argument, "--help") == 0) return 1;
        if (strcmp(argument, "--resume") == 0) {
            options->resume = 1;
        } else if (strcmp(argument, "--steps") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 0, &options->steps) != 0) return -1;
        } else if (strcmp(argument, "--batch-size") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->batch_size) != 0) return -1;
        } else if (strcmp(argument, "--learning-rate") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_float(value, 0.0f, &options->learning_rate) != 0 ||
                options->learning_rate == 0.0f) return -1;
        } else if (strcmp(argument, "--eval-interval") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->eval_interval) != 0) return -1;
        } else if (strcmp(argument, "--eval-batches") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->eval_batches) != 0) return -1;
        } else if (strcmp(argument, "--log-interval") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 1, &options->log_interval) != 0) return -1;
        } else if (strcmp(argument, "--checkpoint") == 0) {
            if (option_value(argc, argv, &index, &value) != 0) return -1;
            options->checkpoint_path = value;
        } else if (strcmp(argument, "--prompt") == 0) {
            if (option_value(argc, argv, &index, &value) != 0) return -1;
            options->prompt = value;
        } else if (strcmp(argument, "--generate") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 0, &options->generate_count) != 0) return -1;
        } else if (strcmp(argument, "--temperature") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_float(value, 0.0f, &options->temperature) != 0) return -1;
        } else if (strcmp(argument, "--top-k") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_int(value, 0, &options->top_k) != 0 ||
                options->top_k > TINY_LM_VOCABULARY) return -1;
        } else if (strcmp(argument, "--seed") == 0) {
            if (option_value(argc, argv, &index, &value) != 0 ||
                parse_u64(value, &options->seed) != 0) return -1;
        } else if (argument[0] == '-' || options->corpus_path != NULL) {
            return -1;
        } else {
            options->corpus_path = argument;
        }
    }
    return options->corpus_path == NULL ? -1 : 0;
}

static int load_corpus(const char* path, byte_corpus* corpus)
{
    FILE* file = NULL;
    long length;
    size_t minimum = 2 * ((size_t)TINY_LM_CONTEXT + 1);

    memset(corpus, 0, sizeof(*corpus));
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Could not open corpus '%s': %s\n",
                path, strerror(errno));
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Could not measure corpus '%s'.\n", path);
        fclose(file);
        return -1;
    }
    if ((size_t)length < minimum) {
        fprintf(stderr,
                "Corpus needs at least %zu bytes for context %d and a "
                "train/validation split.\n",
                minimum, TINY_LM_CONTEXT);
        fclose(file);
        return -1;
    }
    corpus->bytes = (unsigned char*)malloc((size_t)length);
    if (corpus->bytes == NULL ||
        fread(corpus->bytes, 1, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "Could not read corpus '%s'.\n", path);
        free(corpus->bytes);
        memset(corpus, 0, sizeof(*corpus));
        fclose(file);
        return -1;
    }
    fclose(file);
    corpus->size = (size_t)length;
    corpus->train_size = corpus->size * 9 / 10;
    if (corpus->train_size <= TINY_LM_CONTEXT ||
        corpus->size - corpus->train_size <= TINY_LM_CONTEXT) {
        free(corpus->bytes);
        memset(corpus, 0, sizeof(*corpus));
        return -1;
    }
    return 0;
}

static size_t random_start(nn_rng* rng, size_t available)
{
    double sample = nn_rng_uniform(rng, 0.0f, 1.0f);
    size_t result = (size_t)(sample * (double)available);
    return result < available ? result : available - 1;
}

static int make_batch(const byte_corpus* corpus,
                      int training,
                      int batch_size,
                      int batch_index,
                      nn_rng* rng,
                      ag_tensor** inputs,
                      tensor** targets)
{
    const unsigned char* region = training
        ? corpus->bytes : corpus->bytes + corpus->train_size;
    size_t region_size = training
        ? corpus->train_size : corpus->size - corpus->train_size;
    size_t available = region_size - TINY_LM_CONTEXT;
    int dims[2] = {batch_size, TINY_LM_CONTEXT};
    tensor* input_values = NULL;
    tensor* target_values = NULL;

    *inputs = NULL;
    *targets = NULL;
    input_values = t_alloc(2, dims);
    target_values = t_alloc(2, dims);
    if (input_values == NULL || target_values == NULL) goto fail;
    for (int row = 0; row < batch_size; ++row) {
        size_t start;
        if (training) {
            start = random_start(rng, available);
        } else {
            uint64_t sequence =
                (uint64_t)(unsigned)batch_index * (uint64_t)(unsigned)batch_size +
                (uint64_t)(unsigned)row;
            start = (size_t)((sequence * UINT64_C(2654435761)) % available);
        }
        for (int column = 0; column < TINY_LM_CONTEXT; ++column) {
            size_t destination =
                (size_t)row * TINY_LM_CONTEXT + (size_t)column;
            input_values->storage->data[destination] =
                (float)region[start + (size_t)column];
            target_values->storage->data[destination] =
                (float)region[start + (size_t)column + 1];
        }
    }
    *inputs = ag_from_owned_tensor(input_values, 0);
    input_values = NULL;
    *targets = target_values;
    if (*inputs == NULL) goto fail;
    return 0;

fail:
    t_free(target_values);
    t_free(input_values);
    t_free(*targets);
    ag_tensor_release(*inputs);
    *targets = NULL;
    *inputs = NULL;
    return -1;
}

static int evaluate(nn_decoder* model,
                    const byte_corpus* corpus,
                    int batch_size,
                    int batch_count,
                    float* average_loss)
{
    double total = 0.0;

    nn_module_set_training(&model->base, 0);
    for (int batch = 0; batch < batch_count; ++batch) {
        ag_tensor* inputs = NULL;
        tensor* targets = NULL;
        ag_tensor* loss = NULL;

        if (make_batch(corpus, 0, batch_size, batch, NULL,
                       &inputs, &targets) != 0) goto fail;
        loss = nn_decoder_loss(model, inputs, targets);
        if (loss == NULL) goto fail;
        total += loss->value->storage->data[loss->value->offset];
        ag_tensor_release(loss);
        t_free(targets);
        ag_tensor_release(inputs);
        continue;

fail:
        ag_tensor_release(loss);
        t_free(targets);
        ag_tensor_release(inputs);
        nn_module_set_training(&model->base, 1);
        return -1;
    }
    nn_module_set_training(&model->base, 1);
    *average_loss = (float)(total / batch_count);
    return 0;
}

static size_t model_parameter_count(const nn_module* module)
{
    size_t total = 0;
    size_t tensors = nn_module_parameter_count(module);

    for (size_t index = 0; index < tensors; ++index) {
        nn_parameter* parameter = nn_module_parameter_at(module, index);
        int count = tensor_numel(parameter->value->value);
        if (count < 0 || (size_t)count > SIZE_MAX - total) return SIZE_MAX;
        total += (size_t)count;
    }
    return total;
}

static float logit_at(const ag_tensor* logits, int time, int token)
{
    const tensor* value = logits->value;
    int index = value->offset +
                (time - 1) * value->strides[1] +
                token * value->strides[2];
    return value->storage->data[index];
}

static int sample_next(const ag_tensor* logits,
                       int time,
                       float temperature,
                       int top_k,
                       nn_rng* rng)
{
    int order[TINY_LM_VOCABULARY];
    int count = top_k == 0 ? TINY_LM_VOCABULARY : top_k;
    double weights[TINY_LM_VOCABULARY];
    double sum = 0.0;

    for (int token = 0; token < TINY_LM_VOCABULARY; ++token) {
        order[token] = token;
    }
    for (int left = 1; left < TINY_LM_VOCABULARY; ++left) {
        int token = order[left];
        float score = logit_at(logits, time, token);
        int position = left;
        while (position > 0 &&
               logit_at(logits, time, order[position - 1]) < score) {
            order[position] = order[position - 1];
            --position;
        }
        order[position] = token;
    }
    if (temperature == 0.0f) return order[0];
    {
        double maximum = logit_at(logits, time, order[0]) / temperature;
        for (int index = 0; index < count; ++index) {
            double score = logit_at(logits, time, order[index]) / temperature;
            weights[index] = exp(score - maximum);
            sum += weights[index];
        }
    }
    {
        double choice = nn_rng_uniform(rng, 0.0f, 1.0f) * sum;
        for (int index = 0; index < count; ++index) {
            choice -= weights[index];
            if (choice <= 0.0) return order[index];
        }
    }
    return order[count - 1];
}

static int generate(nn_decoder* model,
                    const byte_corpus* corpus,
                    const tiny_lm_options* options)
{
    const unsigned char* initial;
    size_t initial_size;
    unsigned char context[TINY_LM_CONTEXT];
    size_t context_size;
    nn_rng sampling_rng;

    if (options->generate_count == 0) return 0;
    if (options->prompt != NULL) {
        initial = (const unsigned char*)options->prompt;
        initial_size = strlen(options->prompt);
    } else {
        initial = corpus->bytes;
        initial_size = corpus->train_size < 48 ? corpus->train_size : 48;
    }
    if (initial_size == 0) {
        initial = (const unsigned char*)"\n";
        initial_size = 1;
    }
    context_size = initial_size < TINY_LM_CONTEXT
        ? initial_size : TINY_LM_CONTEXT;
    memcpy(context, initial + initial_size - context_size, context_size);
    nn_rng_seed(&sampling_rng, options->seed ^ UINT64_C(0x47454E4552415445));
    nn_module_set_training(&model->base, 0);

    printf("\n--- generation ---\n");
    fwrite(initial, 1, initial_size, stdout);
    for (int generated = 0; generated < options->generate_count; ++generated) {
        int dims[2] = {1, (int)context_size};
        tensor* raw = t_alloc(2, dims);
        ag_tensor* input;
        ag_tensor* logits;
        int next;

        if (raw == NULL) return -1;
        for (size_t index = 0; index < context_size; ++index) {
            raw->storage->data[index] = (float)context[index];
        }
        input = ag_from_owned_tensor(raw, 0);
        logits = input != NULL ? nn_decoder_forward(model, input) : NULL;
        if (logits == NULL) {
            ag_tensor_release(input);
            return -1;
        }
        next = sample_next(logits, (int)context_size,
                           options->temperature, options->top_k,
                           &sampling_rng);
        fputc(next, stdout);
        if (context_size < TINY_LM_CONTEXT) {
            context[context_size++] = (unsigned char)next;
        } else {
            memmove(context, context + 1, TINY_LM_CONTEXT - 1);
            context[TINY_LM_CONTEXT - 1] = (unsigned char)next;
        }
        ag_tensor_release(logits);
        ag_tensor_release(input);
    }
    printf("\n--- end generation ---\n");
    nn_module_set_training(&model->base, 1);
    return 0;
}

int main(int argc, char** argv)
{
    tiny_lm_options options;
    byte_corpus corpus;
    nn_decoder_config model_config;
    nn_adamw_config optimizer_config;
    nn_rng rng;
    nn_decoder* model = NULL;
    nn_adamw* optimizer = NULL;
    size_t parameters;
    double running_loss = 0.0;
    int running_steps = 0;
    int parse_status;
    int status = EXIT_FAILURE;

    parse_status = parse_options(argc, argv, &options);
    if (parse_status != 0) {
        print_usage(argv[0]);
        return parse_status > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (load_corpus(options.corpus_path, &corpus) != 0) return EXIT_FAILURE;

    model_config.vocabulary_size = TINY_LM_VOCABULARY;
    model_config.context_length = TINY_LM_CONTEXT;
    model_config.channels = TINY_LM_CHANNELS;
    model_config.head_count = TINY_LM_HEADS;
    model_config.layer_count = TINY_LM_LAYERS;
    model_config.dropout_probability = 0.1f;
    model_config.layer_norm_epsilon = 1e-5f;
    optimizer_config = nn_adamw_default_config();
    optimizer_config.learning_rate = options.learning_rate;
    optimizer_config.weight_decay = 0.01f;
    optimizer_config.max_grad_norm = 1.0f;
    nn_rng_seed(&rng, options.seed);

    model = nn_decoder_create("tiny_lm", &model_config, &rng);
    optimizer = model != NULL
        ? nn_adamw_create(&model->base, &optimizer_config) : NULL;
    if (model == NULL || optimizer == NULL) {
        fprintf(stderr, "Could not allocate the TinyLM model and optimizer.\n");
        goto cleanup;
    }
    if (options.resume &&
        nn_checkpoint_load(options.checkpoint_path, &model->base,
                           optimizer, &rng) != 0) {
        fprintf(stderr, "Could not resume checkpoint '%s'.\n",
                options.checkpoint_path);
        goto cleanup;
    }

    parameters = model_parameter_count(&model->base);
    printf("Corpus: %zu bytes (%zu train, %zu validation)\n",
           corpus.size, corpus.train_size, corpus.size - corpus.train_size);
    printf("TinyLM: L=%d C=%d H=%d T=%d V=%d, %zu parameters (%.3fM)\n",
           TINY_LM_LAYERS, TINY_LM_CHANNELS, TINY_LM_HEADS,
           TINY_LM_CONTEXT, TINY_LM_VOCABULARY,
           parameters, (double)parameters / 1e6);
    printf("Training: steps=%d batch=%d lr=%.6g checkpoint=%s%s\n",
           options.steps, options.batch_size, (double)options.learning_rate,
           options.checkpoint_path, options.resume ? " (resumed)" : "");

    for (int step = 1; step <= options.steps; ++step) {
        ag_tensor* inputs = NULL;
        tensor* targets = NULL;
        ag_tensor* loss = NULL;
        clock_t start = clock();
        float loss_value;
        double seconds;
        double tokens_per_second;

        nn_adamw_zero_grad(optimizer);
        if (make_batch(&corpus, 1, options.batch_size, step, &rng,
                       &inputs, &targets) != 0) {
            fprintf(stderr, "Could not construct training batch %d.\n", step);
            goto cleanup;
        }
        loss = nn_decoder_loss(model, inputs, targets);
        if (loss == NULL || ag_backward(loss) != 0 ||
            nn_adamw_step(optimizer) != 0) {
            fprintf(stderr, "Training failed at step %d.\n", step);
            ag_tensor_release(loss);
            t_free(targets);
            ag_tensor_release(inputs);
            goto cleanup;
        }
        loss_value = loss->value->storage->data[loss->value->offset];
        ag_tensor_release(loss);
        t_free(targets);
        ag_tensor_release(inputs);
        seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
        tokens_per_second = seconds > 0.0
            ? (double)options.batch_size * TINY_LM_CONTEXT / seconds : 0.0;
        running_loss += loss_value;
        ++running_steps;

        if (step == 1 || step % options.log_interval == 0) {
            printf("step %6d/%d train_loss=%.5f step_time=%.3fs "
                   "tokens/s=%.1f\n",
                   step, options.steps,
                   running_loss / running_steps, seconds, tokens_per_second);
            running_loss = 0.0;
            running_steps = 0;
        }
        if (step % options.eval_interval == 0 || step == options.steps) {
            float validation_loss;
            if (evaluate(model, &corpus, options.batch_size,
                         options.eval_batches, &validation_loss) != 0) {
                fprintf(stderr, "Validation failed at step %d.\n", step);
                goto cleanup;
            }
            printf("step %6d/%d validation_loss=%.5f\n",
                   step, options.steps, (double)validation_loss);
            if (nn_checkpoint_save(options.checkpoint_path, &model->base,
                                   optimizer, &rng) != 0) {
                fprintf(stderr, "Could not save checkpoint '%s'.\n",
                        options.checkpoint_path);
                goto cleanup;
            }
            printf("checkpoint saved: %s\n", options.checkpoint_path);
        }
    }
    if (options.steps == 0) {
        float validation_loss;
        if (evaluate(model, &corpus, options.batch_size,
                     options.eval_batches, &validation_loss) != 0) {
            fprintf(stderr, "Validation failed.\n");
            goto cleanup;
        }
        printf("validation_loss=%.5f\n", (double)validation_loss);
    }
    if (generate(model, &corpus, &options) != 0) {
        fprintf(stderr, "Generation failed.\n");
        goto cleanup;
    }
    status = EXIT_SUCCESS;

cleanup:
    nn_adamw_destroy(optimizer);
    nn_decoder_destroy(model);
    free(corpus.bytes);
    return status;
}
