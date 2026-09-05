#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bench_harness.h"

static int counted_operation(void* opaque, double* checksum)
{
    int* calls = (int*)opaque;
    ++*calls;
    *checksum += 1.0;
    return 0;
}

int main(void)
{
    bench_profile quick = bench_profile_named("quick");
    bench_profile full = bench_profile_named("full");
    bench_profile smoke = bench_profile_named("smoke");
    bench_measurement result;
    int calls = 0;

    if (strcmp(quick.profile, "quick") != 0 || quick.sample_count != 5 ||
        strcmp(full.profile, "full") != 0 || full.sample_count != 15 ||
        strcmp(smoke.profile, "smoke") != 0 || smoke.sample_count != 1) {
        fprintf(stderr, "benchmark profile defaults are incorrect\n");
        return 1;
    }
    if (bench_now_seconds() <= 0.0 ||
        bench_measure(counted_operation, &calls, &smoke, &result) != 0) {
        fprintf(stderr, "smoke measurement failed\n");
        return 1;
    }
    if (calls != 2 || result.iterations_per_sample != 1 ||
        result.checksum != 1.0 || !isfinite(result.median_seconds) ||
        result.median_seconds < 0.0 || result.p95_seconds < 0.0) {
        fprintf(stderr, "smoke measurement produced invalid statistics\n");
        return 1;
    }
    printf("benchmark harness tests passed\n");
    return 0;
}
