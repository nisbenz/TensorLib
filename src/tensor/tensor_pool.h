#ifndef TENSORLIB_TENSOR_POOL_H
#define TENSORLIB_TENSOR_POOL_H

#include <stddef.h>

/* Bounded cache of freed large buffers, matched by exact byte size. Training
 * repeats the same shapes every step, so reuse avoids returning pages to the
 * OS and faulting them back in. Blocks are ordinary malloc memory: a pooled
 * buffer may be released with free(), and any malloc block may be pooled.
 * Contents are uninitialized. TENSORLIB_POOL_MAX_BYTES bounds retained memory
 * (0 disables); it is read once. Sanitizer builds never retain blocks. */
void* tensor_pool_alloc(size_t bytes);
void tensor_pool_free(void* data, size_t bytes);

/* Release everything retained. */
void tensor_pool_trim(void);
size_t tensor_pool_retained_bytes(void);

#endif
