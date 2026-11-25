#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_common.h"
#include "../../../src/tensor/tensor_pool.h"

/* Modes: "default", "disabled" (limit 0), "small" (limit 200000 bytes). */
enum { BLOCK = 80000 };

static int pooling_active(void)
{
    void* data = tensor_pool_alloc(BLOCK);
    int active;
    tensor_pool_free(data, BLOCK);
    active = tensor_pool_retained_bytes() == BLOCK;
    tensor_pool_trim();
    return active;
}

TEST(test_reuses_only_exact_sizes)
{
    void* first = tensor_pool_alloc(BLOCK);
    ASSERT_NOT_NULL(first);
    tensor_pool_free(first, BLOCK);
    void* other = tensor_pool_alloc(BLOCK + 4);
    ASSERT_TRUE(other != first);
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), BLOCK);
    void* second = tensor_pool_alloc(BLOCK);
    ASSERT_TRUE(second == first);
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), 0);
    /* Pooled memory is ordinary malloc memory. */
    free(second);
    free(other);
}

TEST(test_small_and_null_blocks_bypass_pool)
{
    tensor_pool_free(tensor_pool_alloc(1024), 1024);
    tensor_pool_free(NULL, BLOCK);
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), 0);
}

TEST(test_block_count_is_bounded)
{
    void* data[600];
    for (int i = 0; i < 600; ++i) data[i] = tensor_pool_alloc(BLOCK);
    for (int i = 0; i < 600; ++i) tensor_pool_free(data[i], BLOCK);
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), 512 * BLOCK);
    tensor_pool_trim();
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), 0);
}

TEST(test_byte_limit_evicts_oldest)
{
    void* old = tensor_pool_alloc(BLOCK);
    void* middle = tensor_pool_alloc(BLOCK);
    void* newest = tensor_pool_alloc(BLOCK);
    tensor_pool_free(old, BLOCK);
    tensor_pool_free(middle, BLOCK);
    tensor_pool_free(newest, BLOCK);
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), 2 * BLOCK);
    void* reused = tensor_pool_alloc(BLOCK);
    ASSERT_TRUE(reused == newest);
    /* Larger than the whole budget: never retained. */
    tensor_pool_free(tensor_pool_alloc(300000), 300000);
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), BLOCK);
    tensor_pool_free(reused, BLOCK);
    tensor_pool_trim();
}

TEST(test_concurrent_blocks_stay_distinct)
{
    int wrong = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(8) reduction(+:wrong)
#endif
    for (int task = 0; task < 4000; ++task) {
        unsigned char* data = (unsigned char*)tensor_pool_alloc(BLOCK);
        if (data == NULL) { ++wrong; continue; }
        memset(data, task & 255, 256);
        for (int i = 0; i < 256; ++i) wrong += data[i] != (task & 255);
        tensor_pool_free(data, BLOCK);
    }
    ASSERT_EQ_INT(wrong, 0);
    tensor_pool_trim();
}

TEST(test_nothing_is_retained_when_disabled)
{
    tensor_pool_free(tensor_pool_alloc(BLOCK), BLOCK);
    ASSERT_EQ_INT(tensor_pool_retained_bytes(), 0);
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "default";
    int active = pooling_active();
    RUN_TEST(test_small_and_null_blocks_bypass_pool);
    RUN_TEST(test_concurrent_blocks_stay_distinct);
    if (strcmp(mode, "disabled") == 0) {
        RUN_TEST(test_nothing_is_retained_when_disabled);
    } else if (!active) {
        printf("Pooling unavailable in this build; behavior checks skipped.\n");
    } else if (strcmp(mode, "small") == 0) {
        RUN_TEST(test_byte_limit_evicts_oldest);
    } else {
        RUN_TEST(test_reuses_only_exact_sizes);
        RUN_TEST(test_block_count_is_bounded);
    }
    TEST_SUITE_SUMMARY();
}
