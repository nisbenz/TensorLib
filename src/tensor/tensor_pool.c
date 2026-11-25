#include "tensor_pool.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Smaller blocks are already cheap: malloc recycles them without page faults. */
#define POOL_MIN_BLOCK_BYTES ((size_t)1 << 16)
#define POOL_DEFAULT_MAX_BYTES ((size_t)1 << 28)
#define POOL_MAX_BLOCKS 512

/* Retaining freed blocks would hide use-after-free from sanitizers. */
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define POOL_ENABLED 0
#elif defined(__GNUC__) || defined(__clang__)
#define POOL_ENABLED 1
#else
#define POOL_ENABLED 0
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
#undef POOL_ENABLED
#define POOL_ENABLED 0
#endif
#endif

#if POOL_ENABLED
typedef struct {
    void* data;
    size_t bytes;
} pool_block;

/* Ordered oldest to newest; eviction takes the oldest. */
static pool_block blocks[POOL_MAX_BLOCKS];
static int block_count;
static size_t retained_bytes;
static size_t max_bytes;
static int initialized;
static char lock;

/* Never wait: a contended caller simply falls back to malloc/free. */
static int pool_try_lock(void)
{
    return !__atomic_test_and_set(&lock, __ATOMIC_ACQUIRE);
}

static void pool_unlock(void)
{
    __atomic_clear(&lock, __ATOMIC_RELEASE);
}

static void pool_initialize(void)
{
    const char* value;
    char* end;
    unsigned long long parsed;

    if (initialized) return;
    initialized = 1;
    max_bytes = POOL_DEFAULT_MAX_BYTES;
    value = getenv("TENSORLIB_POOL_MAX_BYTES");
    if (value == NULL || *value < '0' || *value > '9') return;
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno == 0 && *end == '\0' && parsed <= SIZE_MAX) {
        max_bytes = (size_t)parsed;
    }
}

static void* pool_remove(int index)
{
    void* data = blocks[index].data;
    retained_bytes -= blocks[index].bytes;
    --block_count;
    memmove(blocks + index, blocks + index + 1,
            (size_t)(block_count - index) * sizeof(*blocks));
    return data;
}
#endif

void* tensor_pool_alloc(size_t bytes)
{
    void* data = NULL;
#if POOL_ENABLED
    if (bytes >= POOL_MIN_BLOCK_BYTES && pool_try_lock()) {
        /* Newest first: most likely still in cache. */
        for (int index = block_count - 1; index >= 0; --index) {
            if (blocks[index].bytes != bytes) continue;
            data = pool_remove(index);
            break;
        }
        pool_unlock();
        if (data != NULL) return data;
    }
#endif
    data = malloc(bytes);
#if POOL_ENABLED
    if (data == NULL && tensor_pool_retained_bytes() > 0) {
        tensor_pool_trim();
        data = malloc(bytes);
    }
#endif
    return data;
}

void tensor_pool_free(void* data, size_t bytes)
{
#if POOL_ENABLED
    if (data != NULL && bytes >= POOL_MIN_BLOCK_BYTES && pool_try_lock()) {
        pool_initialize();
        if (bytes <= max_bytes) {
            while (block_count == POOL_MAX_BLOCKS ||
                   retained_bytes > max_bytes - bytes) {
                free(pool_remove(0));
            }
            blocks[block_count].data = data;
            blocks[block_count].bytes = bytes;
            ++block_count;
            retained_bytes += bytes;
            data = NULL;
        }
        pool_unlock();
    }
#else
    (void)bytes;
#endif
    free(data);
}

void tensor_pool_trim(void)
{
#if POOL_ENABLED
    if (!pool_try_lock()) return;
    while (block_count > 0) free(pool_remove(block_count - 1));
    pool_unlock();
#endif
}

size_t tensor_pool_retained_bytes(void)
{
    size_t result = 0;
#if POOL_ENABLED
    if (pool_try_lock()) {
        result = retained_bytes;
        pool_unlock();
    }
#endif
    return result;
}
