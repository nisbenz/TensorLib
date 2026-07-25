#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./../../include/tensorlib/nn.h"

#define MNIST_ROWS 28
#define MNIST_COLUMNS 28
#define MNIST_FEATURES (MNIST_ROWS * MNIST_COLUMNS)
#define MNIST_CLASSES 10

typedef struct {
    ag_tensor* images;
    tensor* labels;
    int count;
} mnist_dataset;

static uint32_t read_u32_be(FILE* file, int* ok)
{
    unsigned char bytes[4];

    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        *ok = 0;
        return 0;
    }
    return ((uint32_t)bytes[0] << 24) |
           ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) |
           (uint32_t)bytes[3];
}

static int load_mnist(const char* image_path,
                      const char* label_path,
                      mnist_dataset* dataset)
{
    FILE* images = NULL;
    FILE* labels = NULL;
    uint32_t image_count;
    uint32_t label_count;
    uint32_t rows;
    uint32_t columns;
    size_t image_bytes;
    unsigned char* encoded_images = NULL;
    unsigned char* encoded_labels = NULL;
    tensor* image_values = NULL;
    tensor* label_values = NULL;
    int ok = 1;

    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->count = 0;
    images = fopen(image_path, "rb");
    if (images == NULL) {
        fprintf(stderr, "Could not open MNIST images '%s': %s\n",
                image_path, strerror(errno));
        goto fail;
    }
    labels = fopen(label_path, "rb");
    if (labels == NULL) {
        fprintf(stderr,
                "Could not open MNIST labels '%s': %s\n"
                "Place t10k-labels.idx1-ubyte beside the image file.\n",
                label_path, strerror(errno));
        goto fail;
    }

    if (read_u32_be(images, &ok) != UINT32_C(2051)) {
        fprintf(stderr, "Invalid MNIST image-file magic number.\n");
        goto fail;
    }
    image_count = read_u32_be(images, &ok);
    rows = read_u32_be(images, &ok);
    columns = read_u32_be(images, &ok);
    if (!ok || image_count == 0 || image_count > INT32_MAX ||
        rows != MNIST_ROWS || columns != MNIST_COLUMNS) {
        fprintf(stderr, "Expected non-empty 28x28 MNIST images.\n");
        goto fail;
    }
    if (read_u32_be(labels, &ok) != UINT32_C(2049)) {
        fprintf(stderr, "Invalid MNIST label-file magic number.\n");
        goto fail;
    }
    label_count = read_u32_be(labels, &ok);
    if (!ok || label_count != image_count) {
        fprintf(stderr, "MNIST image and label counts do not match.\n");
        goto fail;
    }

    image_bytes = (size_t)image_count * MNIST_FEATURES;
    encoded_images = (unsigned char*)malloc(image_bytes);
    encoded_labels = (unsigned char*)malloc((size_t)image_count);
    if (encoded_images == NULL || encoded_labels == NULL) {
        fprintf(stderr, "Not enough memory to load MNIST.\n");
        goto fail;
    }
    if (fread(encoded_images, 1, image_bytes, images) != image_bytes ||
        fread(encoded_labels, 1, image_count, labels) != image_count) {
        fprintf(stderr, "MNIST data files are truncated.\n");
        goto fail;
    }
    for (uint32_t i = 0; i < label_count; ++i) {
        if (encoded_labels[i] >= MNIST_CLASSES) {
            fprintf(stderr, "MNIST label %u is outside [0, 9].\n",
                    (unsigned)encoded_labels[i]);
            goto fail;
        }
    }
    {
        int image_dims[2] = {(int)image_count, MNIST_FEATURES};
        int label_dims[1] = {(int)image_count};
        image_values = t_alloc(2, image_dims);
        label_values = t_alloc(1, label_dims);
    }
    if (image_values == NULL || label_values == NULL) {
        fprintf(stderr, "Not enough memory to decode MNIST.\n");
        goto fail;
    }
    for (size_t i = 0; i < image_bytes; ++i) {
        image_values->storage->data[i] = (float)encoded_images[i] / 255.0f;
    }
    for (uint32_t i = 0; i < label_count; ++i) {
        label_values->storage->data[i] = (float)encoded_labels[i];
    }
    dataset->images = ag_from_owned_tensor(image_values, 0);
    image_values = NULL;
    if (dataset->images == NULL) goto fail;
    dataset->labels = label_values;
    label_values = NULL;

    dataset->count = (int)image_count;
    free(encoded_labels);
    free(encoded_images);
    fclose(labels);
    fclose(images);
    return 0;

fail:
    t_free(label_values);
    t_free(image_values);
    free(encoded_labels);
    free(encoded_images);
    t_free(dataset->labels);
    ag_tensor_release(dataset->images);
    dataset->labels = NULL;
    dataset->images = NULL;
    if (labels != NULL) fclose(labels);
    if (images != NULL) fclose(images);
    return -1;
}

static void free_mnist(mnist_dataset* dataset)
{
    t_free(dataset->labels);
    ag_tensor_release(dataset->images);
    dataset->labels = NULL;
    dataset->images = NULL;
    dataset->count = 0;
}

static int make_batch(const mnist_dataset* dataset,
                      int start,
                      int count,
                      ag_tensor** inputs,
                      tensor** targets)
{
    *inputs = NULL;
    *targets = NULL;
    *inputs = ag_slice(dataset->images, 0, start, start + count);
    *targets = t_slice(dataset->labels, 0, start, start + count);
    if (*inputs == NULL || *targets == NULL) {
        t_free(*targets);
        ag_tensor_release(*inputs);
        *targets = NULL;
        *inputs = NULL;
        return -1;
    }
    return 0;
}

static int train_epoch(nn_mlp* model,
                       nn_sgd* optimizer,
                       const mnist_dataset* dataset,
                       int train_count,
                       int batch_size,
                       float* average_loss)
{
    double weighted_loss = 0.0;

    for (int start = 0; start < train_count; start += batch_size) {
        int count = batch_size;
        ag_tensor* inputs = NULL;
        tensor* targets = NULL;
        ag_tensor* logits = NULL;
        ag_tensor* loss = NULL;

        if (count > train_count - start) count = train_count - start;
        if (make_batch(dataset, start, count, &inputs, &targets) != 0) goto fail;
        logits = nn_mlp_forward(model, inputs);
        loss = nn_cross_entropy(logits, targets);
        if (logits == NULL || loss == NULL || ag_backward(loss) != 0 ||
            nn_sgd_step(optimizer) != 0) {
            goto fail;
        }
        weighted_loss +=
            (double)loss->value->storage->data[loss->value->offset] * count;
        nn_sgd_zero_grad(optimizer);
        ag_tensor_release(loss);
        ag_tensor_release(logits);
        t_free(targets);
        ag_tensor_release(inputs);
        continue;

fail:
        ag_tensor_release(loss);
        ag_tensor_release(logits);
        t_free(targets);
        ag_tensor_release(inputs);
        return -1;
    }
    *average_loss = (float)(weighted_loss / train_count);
    return 0;
}

static int evaluate(nn_mlp* model,
                    const mnist_dataset* dataset,
                    int start_index,
                    int test_count,
                    int batch_size,
                    float* average_loss,
                    float* accuracy,
                    int confusion[MNIST_CLASSES][MNIST_CLASSES])
{
    double weighted_loss = 0.0;
    int correct = 0;

    for (int actual = 0; actual < MNIST_CLASSES; ++actual) {
        for (int predicted = 0; predicted < MNIST_CLASSES; ++predicted) {
            confusion[actual][predicted] = 0;
        }
    }
    for (int offset = 0; offset < test_count; offset += batch_size) {
        int count = batch_size;
        ag_tensor* inputs = NULL;
        tensor* targets = NULL;
        ag_tensor* logits = NULL;
        ag_tensor* loss = NULL;

        if (count > test_count - offset) count = test_count - offset;
        if (make_batch(dataset, start_index + offset, count,
                       &inputs, &targets) != 0) {
            goto fail;
        }
        logits = nn_mlp_forward(model, inputs);
        loss = nn_cross_entropy(logits, targets);
        if (logits == NULL || loss == NULL) goto fail;
        weighted_loss +=
            (double)loss->value->storage->data[loss->value->offset] * count;
        for (int row = 0; row < count; ++row) {
            int predicted = 0;
            float best = logits->value->storage->data[
                logits->value->offset + row * MNIST_CLASSES];
            for (int class_index = 1;
                 class_index < MNIST_CLASSES;
                 ++class_index) {
                float candidate = logits->value->storage->data[
                    logits->value->offset +
                    row * MNIST_CLASSES + class_index];
                if (candidate > best) {
                    best = candidate;
                    predicted = class_index;
                }
            }
            int actual = (int)dataset->labels->storage->data[
                dataset->labels->offset + start_index + offset + row];
            ++confusion[actual][predicted];
            if (predicted == actual) ++correct;
        }
        ag_tensor_release(loss);
        ag_tensor_release(logits);
        t_free(targets);
        ag_tensor_release(inputs);
        continue;

fail:
        ag_tensor_release(loss);
        ag_tensor_release(logits);
        t_free(targets);
        ag_tensor_release(inputs);
        return -1;
    }
    *average_loss = (float)(weighted_loss / test_count);
    *accuracy = (float)correct / test_count;
    return 0;
}

static void print_confusion_matrix(
    int confusion[MNIST_CLASSES][MNIST_CLASSES])
{
    printf("\nConfusion matrix (rows=actual, columns=predicted):\n");
    printf("       0    1    2    3    4    5    6    7    8    9   accuracy\n");
    for (int actual = 0; actual < MNIST_CLASSES; ++actual) {
        int total = 0;
        for (int predicted = 0; predicted < MNIST_CLASSES; ++predicted) {
            total += confusion[actual][predicted];
        }
        printf("%d  ", actual);
        for (int predicted = 0; predicted < MNIST_CLASSES; ++predicted) {
            printf("%5d", confusion[actual][predicted]);
        }
        printf("   %6.2f%%\n",
               total > 0
                   ? 100.0 * confusion[actual][actual] / total
                   : 0.0);
    }
}

/*
 * TensorLib does not currently provide parameter serialization. This compact
 * binary format stores enough metadata to validate and reload each parameter:
 * magic, version, parameter count, then name, shape, and contiguous float data.
 */
static int save_weights(const nn_module* module, const char* path)
{
    static const unsigned char magic[8] = {
        'T', 'L', 'W', 'E', 'I', 'G', 'H', 'T'
    };
    const uint32_t version = 1;
    size_t parameter_count = nn_module_parameter_count(module);
    uint32_t serialized_count;
    FILE* file;

    if (parameter_count > UINT32_MAX) return -1;
    serialized_count = (uint32_t)parameter_count;
    file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "Could not create weights file '%s': %s\n",
                path, strerror(errno));
        return -1;
    }
    if (fwrite(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        fwrite(&version, sizeof(version), 1, file) != 1 ||
        fwrite(&serialized_count, sizeof(serialized_count), 1, file) != 1) {
        goto fail;
    }
    for (size_t i = 0; i < parameter_count; ++i) {
        nn_parameter* parameter = nn_module_parameter_at(module, i);
        tensor* value;
        size_t name_length;
        uint32_t serialized_name_length;
        uint32_t ndim;
        uint64_t element_count;

        if (parameter == NULL || parameter->name == NULL ||
            parameter->value == NULL ||
            !tensor_has_valid_metadata(parameter->value->value)) {
            goto fail;
        }
        value = parameter->value->value;
        name_length = strlen(parameter->name);
        if (name_length > UINT32_MAX || value->ndim < 0 ||
            !is_contiguous(value)) {
            goto fail;
        }
        serialized_name_length = (uint32_t)name_length;
        ndim = (uint32_t)value->ndim;
        element_count = (uint64_t)tensor_numel(value);
        if (fwrite(&serialized_name_length,
                   sizeof(serialized_name_length), 1, file) != 1 ||
            fwrite(parameter->name, 1, name_length, file) != name_length ||
            fwrite(&ndim, sizeof(ndim), 1, file) != 1) {
            goto fail;
        }
        for (int dim = 0; dim < value->ndim; ++dim) {
            int32_t size = value->dims[dim];
            if (fwrite(&size, sizeof(size), 1, file) != 1) goto fail;
        }
        if (fwrite(&element_count, sizeof(element_count), 1, file) != 1 ||
            fwrite(value->storage->data + value->offset,
                   sizeof(float), (size_t)element_count, file) !=
                (size_t)element_count) {
            goto fail;
        }
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Could not finish weights file '%s'.\n", path);
        return -1;
    }
    return 0;

fail:
    fclose(file);
    fprintf(stderr, "Failed while writing weights file '%s'.\n", path);
    return -1;
}

static int parse_positive_int(const char* text, int* result)
{
    char* end;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0' ||
        value <= 0 || value > INT32_MAX) {
        return -1;
    }
    *result = (int)value;
    return 0;
}

int main(int argc, char** argv)
{
    const char* image_path = argc > 1
        ? argv[1] : "examples/mnsit/t10k-images.idx3-ubyte";
    const char* label_path = argc > 2
        ? argv[2] : "examples/mnsit/t10k-labels.idx1-ubyte";
    const char* weights_path = argc > 5
        ? argv[5] : "examples/mnsit/mnist_mlp.weights";
    int epochs = 10;
    int batch_size = 64;
    int train_count;
    int test_count;
    int hidden_sizes[] = {128};
    nn_activation activations[] = {
        nn_activation_relu(),
        nn_activation_custom("identity", NULL, NULL)
    };
    nn_mlp_config config = {
        MNIST_FEATURES,
        hidden_sizes,
        1,
        MNIST_CLASSES,
        activations,
        1,
        NN_INIT_HE_NORMAL,
        NN_INIT_ZERO
    };
    mnist_dataset dataset;
    nn_rng rng;
    nn_mlp* model = NULL;
    nn_sgd* optimizer = NULL;
    int confusion[MNIST_CLASSES][MNIST_CLASSES];
    int status = EXIT_FAILURE;

    if ((argc > 3 && parse_positive_int(argv[3], &epochs) != 0) ||
        (argc > 4 && parse_positive_int(argv[4], &batch_size) != 0) ||
        argc > 6) {
        fprintf(stderr,
                "Usage: %s [images] [labels] [epochs] [batch-size] [weights]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (load_mnist(image_path, label_path, &dataset) != 0) return EXIT_FAILURE;
    train_count = dataset.count * 3 / 4;
    test_count = dataset.count - train_count;
    if (train_count == 0 || test_count == 0) {
        fprintf(stderr, "MNIST dataset is too small for a 75/25 split.\n");
        goto cleanup;
    }

    nn_rng_seed(&rng, UINT64_C(0x4D4E495354));
    model = nn_mlp_create("mnist_mlp", &config, &rng);
    optimizer = model != NULL ? nn_sgd_create(&model->base, 0.05f) : NULL;
    if (model == NULL || optimizer == NULL) {
        fprintf(stderr, "Could not create the MLP or SGD optimizer.\n");
        goto cleanup;
    }

    printf("Loaded %d 28x28 images: %d train (75%%), %d test (25%%)\n",
           dataset.count, train_count, test_count);
    printf("MLP: 784 input -> 128 ReLU -> 10 Softmax, "
           "cross-entropy, SGD(lr=0.05)\n");
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float train_loss;

        if (train_epoch(model, optimizer, &dataset, train_count,
                        batch_size, &train_loss) != 0) {
            fprintf(stderr, "Training failed in epoch %d.\n", epoch);
            goto cleanup;
        }
        printf("epoch %2d/%d  train_loss=%.6f\n",
               epoch, epochs, (double)train_loss);
    }
    {
        float test_loss;
        float test_accuracy;
        if (evaluate(model, &dataset, train_count, test_count, batch_size,
                     &test_loss, &test_accuracy, confusion) != 0) {
            fprintf(stderr, "Final evaluation failed.\n");
            goto cleanup;
        }
        printf("final test_loss=%.6f  test_accuracy=%.2f%%\n",
               (double)test_loss, (double)test_accuracy * 100.0);
    }
    print_confusion_matrix(confusion);
    if (save_weights(&model->base, weights_path) != 0) goto cleanup;
    printf("\nSaved %zu parameter tensors to %s\n",
           nn_module_parameter_count(&model->base), weights_path);
    status = EXIT_SUCCESS;

cleanup:
    nn_sgd_destroy(optimizer);
    nn_mlp_destroy(model);
    free_mnist(&dataset);
    return status;
}
