#include <tensorlib/tensor.h>

int main(void) {
    const int dims[] = {2, 2};
    tensor* value = t_alloc(2, dims);
    if (value == NULL) return 1;

    t_free(value);
    return 0;
}
