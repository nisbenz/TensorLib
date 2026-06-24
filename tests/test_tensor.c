#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/tensor.h"

// Simple testing harness macros
#define RUN_TEST(test_func) do { \
    printf("Running %s...\n", #test_func); \
    if (test_func() == 0) { \
        printf("  [PASS] %s\n", #test_func); \
        passed_tests++; \
    } else { \
        printf("  [FAIL] %s\n", #test_func); \
        failed_tests++; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("    Assertion FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))
#define ASSERT_EQUAL_INT(val, exp) ASSERT_TRUE((val) == (exp))
#define ASSERT_EQUAL_FLOAT(val, exp, eps) ASSERT_TRUE(fabs((val) - (exp)) < (eps))

static int passed_tests = 0;
static int failed_tests = 0;

// Test 1: Allocation and Deallocation
int test_alloc_free() {
    int dims[2] = {2, 3};
    tensor* t = t_alloc(2, dims);

    ASSERT_TRUE(t != NULL);
    ASSERT_TRUE(t->storage != NULL);
    ASSERT_EQUAL_INT(t->ndim, 2);
    ASSERT_EQUAL_INT(t->dims[0], 2);
    ASSERT_EQUAL_INT(t->dims[1], 3);
    
    // Check contiguous strides
    ASSERT_EQUAL_INT(t->strides[1], 1);
    ASSERT_EQUAL_INT(t->strides[0], 3);

    // Check storage size and reference count
    ASSERT_EQUAL_INT(t->storage->size, 6);
    ASSERT_EQUAL_INT(t->storage->ref_count, 1);

    t_free(t);
    return 0;
}

// Test 2: Reference Counting on Views
int test_ref_counting() {
    int dims[2] = {2, 3};
    tensor* t1 = t_alloc(2, dims);
    ASSERT_TRUE(t1 != NULL);
    ASSERT_EQUAL_INT(t1->storage->ref_count, 1);

    // Create transposed view (shares storage)
    tensor* t2 = t_transpose(t1, 0, 1);
    ASSERT_TRUE(t2 != NULL);
    ASSERT_TRUE(t1->storage == t2->storage);
    ASSERT_EQUAL_INT(t1->storage->ref_count, 2);

    // Free the original tensor, reference count should drop but storage remains
    t_free(t1);
    ASSERT_EQUAL_INT(t2->storage->ref_count, 1);

    // Free the view, this should free the storage
    t_free(t2);
    return 0;
}

// Test 3: Shape and Stride Comparisons
int test_shape_and_stride() {
    int dims1[2] = {2, 3};
    int dims2[2] = {2, 3};
    int dims3[2] = {3, 2};
    int dims4[3] = {2, 3, 1};

    tensor* t1 = t_alloc(2, dims1);
    tensor* t2 = t_alloc(2, dims2);
    tensor* t3 = t_alloc(2, dims3);
    tensor* t4 = t_alloc(3, dims4);

    ASSERT_TRUE(same_shape(t1, t2));
    ASSERT_FALSE(same_shape(t1, t3));
    ASSERT_FALSE(same_shape(t1, t4));

    ASSERT_TRUE(same_stride(t1, t2));
    
    // Create a transposed view of t1 (shape becomes 3x2, strides become 1, 3)
    tensor* t1_trans = t_transpose(t1, 0, 1);
    // t3 has shape 3x2, but contiguous strides (strides: 2, 1)
    ASSERT_TRUE(same_shape(t1_trans, t3));
    ASSERT_FALSE(same_stride(t1_trans, t3));

    t_free(t1);
    t_free(t2);
    t_free(t3);
    t_free(t4);
    t_free(t1_trans);
    return 0;
}

// Test 4: Contiguity Verification
int test_contiguity() {
    int dims[2] = {2, 3};
    tensor* t = t_alloc(2, dims);

    ASSERT_TRUE(is_contiguous(t));

    // Transposing the tensor makes it non-contiguous
    tensor* t_trans = t_transpose(t, 0, 1);
    ASSERT_FALSE(is_contiguous(t_trans));

    t_free(t);
    t_free(t_trans);
    return 0;
}

// Test 5: Cloning Tensors (Contiguous and Non-Contiguous)
int test_clone() {
    int dims[2] = {2, 3};
    tensor* t1 = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) {
        t1->storage->data[i] = (float)i;
    }

    // Clone contiguous tensor
    tensor* t1_clone = t_clone(t1);
    ASSERT_TRUE(t1_clone != NULL);
    ASSERT_TRUE(same_shape(t1, t1_clone));
    ASSERT_TRUE(same_stride(t1, t1_clone));
    ASSERT_TRUE(t1->storage != t1_clone->storage); // Should be independent storage
    ASSERT_EQUAL_INT(t1_clone->storage->ref_count, 1);

    for (int i = 0; i < 6; i++) {
        ASSERT_EQUAL_FLOAT(t1_clone->storage->data[i], t1->storage->data[i], 1e-5f);
    }

    // Clone non-contiguous (transposed) tensor
    tensor* t1_trans = t_transpose(t1, 0, 1);
    tensor* t1_trans_clone = t_clone(t1_trans);

    ASSERT_TRUE(t1_trans_clone != NULL);
    ASSERT_TRUE(same_shape(t1_trans, t1_trans_clone));
    ASSERT_TRUE(same_stride(t1_trans, t1_trans_clone));
    ASSERT_TRUE(t1_trans->storage != t1_trans_clone->storage); // Independent storage

    // Check that cloned values in their physical slots match the transposed source
    for (int i = 0; i < 6; i++) {
        ASSERT_EQUAL_FLOAT(t1_trans_clone->storage->data[i], t1_trans->storage->data[i], 1e-5f);
    }

    t_free(t1);
    t_free(t1_clone);
    t_free(t1_trans);
    t_free(t1_trans_clone);
    return 0;
}

// Test 6: Transposition Layout
int test_transposition() {
    int dims[2] = {2, 3};
    tensor* t = t_alloc(2, dims);
    
    // Set sequential values
    // [0.0, 1.0, 2.0]
    // [3.0, 4.0, 5.0]
    for (int i = 0; i < 6; i++) {
        t->storage->data[i] = (float)i;
    }

    // Transpose
    tensor* t_trans = t_transpose(t, 0, 1);
    ASSERT_EQUAL_INT(t_trans->ndim, 2);
    ASSERT_EQUAL_INT(t_trans->dims[0], 3);
    ASSERT_EQUAL_INT(t_trans->dims[1], 2);
    
    // Strides should be swapped from (3, 1) to (1, 3)
    ASSERT_EQUAL_INT(t_trans->strides[0], 1);
    ASSERT_EQUAL_INT(t_trans->strides[1], 3);

    // Verify logical values:
    // Transposed layout should represent:
    // [0.0, 3.0]
    // [1.0, 4.0]
    // [2.0, 5.0]
    int coords00[2] = {0, 0};
    int coords01[2] = {0, 1};
    int coords10[2] = {1, 0};
    int coords11[2] = {1, 1};
    int coords20[2] = {2, 0};
    int coords21[2] = {2, 1};

    ASSERT_EQUAL_FLOAT(t_trans->storage->data[get_flat_index_nd(t_trans, coords00)], 0.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_trans->storage->data[get_flat_index_nd(t_trans, coords01)], 3.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_trans->storage->data[get_flat_index_nd(t_trans, coords10)], 1.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_trans->storage->data[get_flat_index_nd(t_trans, coords11)], 4.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_trans->storage->data[get_flat_index_nd(t_trans, coords20)], 2.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_trans->storage->data[get_flat_index_nd(t_trans, coords21)], 5.0f, 1e-5f);

    t_free(t);
    t_free(t_trans);
    return 0;
}

// Test 7: Make Contiguous (t_contiguous)
int test_contiguous_conversion() {
    int dims[2] = {2, 3};
    tensor* t = t_alloc(2, dims);
    for (int i = 0; i < 6; i++) {
        t->storage->data[i] = (float)i;
    }

    // Already contiguous
    tensor* t_contig1 = t_contiguous(t);
    ASSERT_TRUE(is_contiguous(t_contig1));
    ASSERT_TRUE(t_contig1->storage == t->storage); // Should just increase ref_count

    // Make a non-contiguous transposed tensor
    tensor* t_trans = t_transpose(t, 0, 1);
    ASSERT_FALSE(is_contiguous(t_trans));

    // Convert to contiguous, which should allocate new storage and lay out elements flat
    tensor* t_contig2 = t_contiguous(t_trans);
    ASSERT_TRUE(is_contiguous(t_contig2));
    ASSERT_TRUE(t_contig2->storage != t_trans->storage); // New storage
    ASSERT_EQUAL_INT(t_contig2->storage->size, 6);

    // Verify flat physical layout of t_contig2:
    // Logically:
    // [0.0, 3.0]
    // [1.0, 4.0]
    // [2.0, 5.0]
    // Physically, since it is contiguous, it should be stored as:
    // 0.0, 3.0, 1.0, 4.0, 2.0, 5.0
    ASSERT_EQUAL_FLOAT(t_contig2->storage->data[0], 0.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_contig2->storage->data[1], 3.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_contig2->storage->data[2], 1.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_contig2->storage->data[3], 4.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_contig2->storage->data[4], 2.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(t_contig2->storage->data[5], 5.0f, 1e-5f);

    t_free(t);
    t_free(t_contig1);
    t_free(t_trans);
    t_free(t_contig2);
    return 0;
}

// Test 8: Contiguous Tensor Addition
int test_add_contiguous() {
    int dims[2] = {2, 3};
    tensor* a = t_alloc(2, dims);
    tensor* b = t_alloc(2, dims);

    for (int i = 0; i < 6; i++) {
        a->storage->data[i] = (float)i;
        b->storage->data[i] = (float)(10 - i);
    }

    tensor* c = t_add(a, b);
    ASSERT_TRUE(c != NULL);
    ASSERT_TRUE(is_contiguous(c));
    ASSERT_EQUAL_INT(c->storage->size, 6);

    // Since they are contiguous and have the same strides, they should be added element-wise physically
    for (int i = 0; i < 6; i++) {
        ASSERT_EQUAL_FLOAT(c->storage->data[i], 10.0f, 1e-5f);
    }

    t_free(a);
    t_free(b);
    t_free(c);
    return 0;
}

// Test 9: Non-Contiguous Tensor Addition (Fixed code path!)
int test_add_non_contiguous() {
    int dims_a[2] = {2, 3};
    tensor* a = t_alloc(2, dims_a);
    // a =
    // [0, 1, 2]
    // [3, 4, 5]
    for (int i = 0; i < 6; i++) {
        a->storage->data[i] = (float)i;
    }

    // We will transpose a to get a_trans (shape 3x2)
    // a_trans =
    // [0, 3]
    // [1, 4]
    // [2, 5]
    tensor* a_trans = t_transpose(a, 0, 1);

    // Create a contiguous b of shape 3x2
    // b =
    // [10, 20]
    // [30, 40]
    // [50, 60]
    int dims_b[2] = {3, 2};
    tensor* b = t_alloc(2, dims_b);
    for (int i = 0; i < 6; i++) {
        b->storage->data[i] = (float)((i + 1) * 10);
    }

    // Add them! Since strides are different:
    // a_trans has shape 3x2, strides (1, 3)
    // b has shape 3x2, strides (2, 1)
    // This forces the non-contiguous path we fixed!
    tensor* c = t_add(a_trans, b);
    ASSERT_TRUE(c != NULL);
    
    // The shape of c should be 3x2.
    // It should have the same strides as a_trans: (1, 3)
    ASSERT_EQUAL_INT(c->dims[0], 3);
    ASSERT_EQUAL_INT(c->dims[1], 2);
    ASSERT_EQUAL_INT(c->strides[0], 1);
    ASSERT_EQUAL_INT(c->strides[1], 3);

    // Logically, the sum should be:
    // [0 + 10, 3 + 20] = [10, 23]
    // [1 + 30, 4 + 40] = [31, 44]
    // [2 + 50, 5 + 60] = [52, 65]
    //
    // Since c has strides (1, 3), its physical layout is:
    // idx for (0,0) = 0
    // idx for (1,0) = 1
    // idx for (2,0) = 2
    // idx for (0,1) = 3
    // idx for (1,1) = 4
    // idx for (2,1) = 5
    //
    // So c->storage->data should contain physically:
    // index 0 (0,0) = 10.0
    // index 1 (1,0) = 31.0
    // index 2 (2,0) = 52.0
    // index 3 (0,1) = 23.0
    // index 4 (1,1) = 44.0
    // index 5 (2,1) = 65.0
    
    ASSERT_EQUAL_FLOAT(c->storage->data[0], 10.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(c->storage->data[1], 31.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(c->storage->data[2], 52.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(c->storage->data[3], 23.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(c->storage->data[4], 44.0f, 1e-5f);
    ASSERT_EQUAL_FLOAT(c->storage->data[5], 65.0f, 1e-5f);

    t_free(a);
    t_free(a_trans);
    t_free(b);
    t_free(c);
    return 0;
}

int main() {
    printf("===========================================\n");
    printf("        RUNNING TENSORLIB TEST SUITE       \n");
    printf("===========================================\n");

    RUN_TEST(test_alloc_free);
    RUN_TEST(test_ref_counting);
    RUN_TEST(test_shape_and_stride);
    RUN_TEST(test_contiguity);
    RUN_TEST(test_clone);
    RUN_TEST(test_transposition);
    RUN_TEST(test_contiguous_conversion);
    RUN_TEST(test_add_contiguous);
    RUN_TEST(test_add_non_contiguous);

    printf("===========================================\n");
    printf("TEST SUMMARY: %d passed, %d failed\n", passed_tests, failed_tests);
    printf("===========================================\n");

    return failed_tests > 0 ? 1 : 0;
}
