#include "command_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t next_random(uint64_t* state)
{
    *state ^= *state << 7;
    *state ^= *state >> 9;
    return (uint32_t)(*state);
}

static int append_text(command_corpus* corpus, size_t* capacity,
                       const char* text)
{
    size_t length = strlen(text);
    if (length > SIZE_MAX - corpus->size) return -1;
    if (corpus->size + length > *capacity) {
        size_t next = *capacity == 0 ? 4096 : *capacity;
        while (next < corpus->size + length) {
            if (next > SIZE_MAX / 2) return -1;
            next *= 2;
        }
        corpus->bytes = (unsigned char*)realloc(corpus->bytes, next);
        if (corpus->bytes == NULL) return -1;
        *capacity = next;
    }
    memcpy(corpus->bytes + corpus->size, text, length);
    corpus->size += length;
    return 0;
}

int command_corpus_generate(command_corpus* corpus, size_t examples,
                            uint64_t seed)
{
    static const char* repositories[] = {"tensorlib", "app", "service", "demo"};
    static const char* files[] = {"src/main.c", "README.md", "CMakeLists.txt", "tests"};
    static const char* requests[] = {
        "please %s in the %s repository",
        "run %s for project %s",
        "I need to %s on %s",
        "developer request: %s, target %s"
    };
    static const char* actions[][2] = {
        {"show the current status", "status"},
        {"show the pending diff", "diff"},
        {"show recent history", "log"},
        {"list branches", "branch"},
        {"configure the project", "configure"},
        {"build the project", "build"},
        {"run the tests", "test"},
        {"search for TODO", "search"},
        {"list files", "list"}
    };
    size_t capacity = 0;
    uint64_t state = seed == 0 ? UINT64_C(1) : seed;
    char line[512];
    if (corpus == NULL || examples == 0) return -1;
    memset(corpus, 0, sizeof(*corpus));
    for (size_t index = 0; index < examples; ++index) {
        size_t action = next_random(&state) % (sizeof(actions) / sizeof(actions[0]));
        size_t repository = next_random(&state) % (sizeof(repositories) / sizeof(repositories[0]));
        size_t file = next_random(&state) % (sizeof(files) / sizeof(files[0]));
        size_t request = next_random(&state) % (sizeof(requests) / sizeof(requests[0]));
        const char* target = action >= 7 ? files[file] : repositories[repository];
        (void)snprintf(line, sizeof(line), "REQUEST: ");
        if (append_text(corpus, &capacity, line) != 0) goto fail;
        (void)snprintf(line, sizeof(line), requests[request], actions[action][0], target);
        if (append_text(corpus, &capacity, line) != 0 ||
            append_text(corpus, &capacity, "\nCOMMAND: TOOL ") != 0) goto fail;
        if (action < 4) {
            (void)snprintf(line, sizeof(line), "git ACTION %s TARGET %s\n\n",
                           actions[action][1], repositories[repository]);
        } else if (action < 7) {
            (void)snprintf(line, sizeof(line), "cmake ACTION %s TARGET %s\n\n",
                           actions[action][1], repositories[repository]);
        } else if (action == 7) {
            (void)snprintf(line, sizeof(line), "grep ACTION search TARGET %s\n\n", file ? "TODO" : "FIXME");
        } else {
            (void)snprintf(line, sizeof(line), "files ACTION list TARGET %s\n\n", repositories[repository]);
        }
        if (append_text(corpus, &capacity, line) != 0) goto fail;
    }
    corpus->train_size = corpus->size * 9 / 10;
    return corpus->train_size > 256 ? 0 : -1;
fail:
    command_corpus_destroy(corpus);
    return -1;
}

int command_corpus_load(command_corpus* corpus, const char* path)
{
    FILE* file;
    long length;
    if (corpus == NULL || path == NULL) return -1;
    memset(corpus, 0, sizeof(*corpus));
    file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) <= 256 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) fclose(file);
        return -1;
    }
    corpus->bytes = (unsigned char*)malloc((size_t)length);
    if (corpus->bytes == NULL ||
        fread(corpus->bytes, 1, (size_t)length, file) != (size_t)length) {
        fclose(file);
        command_corpus_destroy(corpus);
        return -1;
    }
    fclose(file);
    corpus->size = (size_t)length;
    corpus->train_size = corpus->size * 9 / 10;
    return 0;
}

int command_corpus_save(const command_corpus* corpus, const char* path)
{
    FILE* file;
    if (corpus == NULL || corpus->bytes == NULL || path == NULL) return -1;
    file = fopen(path, "wb");
    if (file == NULL) return -1;
    if (fwrite(corpus->bytes, 1, corpus->size, file) != corpus->size) {
        fclose(file);
        return -1;
    }
    fclose(file);
    return 0;
}

void command_corpus_destroy(command_corpus* corpus)
{
    if (corpus == NULL) return;
    free(corpus->bytes);
    memset(corpus, 0, sizeof(*corpus));
}
