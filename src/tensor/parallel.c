#include "parallel.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#ifdef _OPENMP
typedef struct {
    const char* name;
    tensorlib_parallel_kind family;
    int flops;
} policy_description;

static const policy_description descriptions[TENSORLIB_PARALLEL_COUNT] = {
    {"MEMORY", TENSORLIB_PARALLEL_MEMORY, 0},
    {"COMPUTE", TENSORLIB_PARALLEL_COMPUTE, 0},
    {"MATMUL", TENSORLIB_PARALLEL_MATMUL, 1},
    {"CLONE", TENSORLIB_PARALLEL_MEMORY, 0},
    {"PACK_RHS", TENSORLIB_PARALLEL_MEMORY, 0},
    {"GELU", TENSORLIB_PARALLEL_COMPUTE, 0},
    {"GELU_BACKWARD", TENSORLIB_PARALLEL_COMPUTE, 0},
    {"PACKED_MATMUL", TENSORLIB_PARALLEL_MATMUL, 1},
    {"BATCHED_MATMUL", TENSORLIB_PARALLEL_MATMUL, 1},
    {"ADAMW", TENSORLIB_PARALLEL_MEMORY, 0}
};

static int thread_limits[TENSORLIB_PARALLEL_COUNT];
static long long work_limits[TENSORLIB_PARALLEL_COUNT];
static int initialized;

/* Zero means inherit; never parse environment strings on the hot path. */
static long long read_limit(const char* name, long long maximum)
{
    const char* value = getenv(name);
    char* end;
    long long parsed;
    if (value == NULL || *value == '\0') return 0;
    for (const char* cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') goto invalid;
    }
    errno = 0;
    parsed = strtoll(value, &end, 10);
    if (errno == 0 && *end == '\0' && parsed > 0 && parsed <= maximum) {
        return parsed;
    }
invalid:
    fprintf(stderr, "TensorLib: invalid %s; using inherited/default setting\n",
            name);
    return 0;
}

static void initialize_policy(void)
{
    int ready;
#pragma omp atomic read seq_cst
    ready = initialized;
    if (ready) return;
#pragma omp critical(tensorlib_parallel_configuration)
    {
        if (!initialized) {
            for (int index = 0; index < TENSORLIB_PARALLEL_COUNT; ++index) {
                char name[80];
                int family = (int)descriptions[index].family;
                snprintf(name, sizeof(name), "TENSORLIB_%s_THREADS",
                         descriptions[index].name);
                thread_limits[index] = (int)read_limit(name, INT_MAX);
                snprintf(name, sizeof(name), "TENSORLIB_%s_MIN_%s",
                         descriptions[index].name,
                         descriptions[index].flops ? "FLOPS" : "ELEMENTS");
                work_limits[index] = read_limit(name, LLONG_MAX);
                if (thread_limits[index] == 0)
                    thread_limits[index] = thread_limits[family];
                if (work_limits[index] == 0)
                    work_limits[index] = work_limits[family];
            }
#pragma omp atomic write seq_cst
            initialized = 1;
        }
    }
}
#endif

int tensorlib_parallel_threads_for(tensorlib_parallel_kind kind,
                                  long long work, long long default_limit,
                                  int tasks)
{
#ifdef _OPENMP
    int maximum;
    long long limit;
    if (omp_in_parallel() || kind < 0 || kind >= TENSORLIB_PARALLEL_COUNT)
        return 1;
    initialize_policy();
    limit = work_limits[kind] > 0 ? work_limits[kind] : default_limit;
    if (work < limit) return 1;
    maximum = omp_get_max_threads();
    if (thread_limits[kind] > 0 && thread_limits[kind] < maximum)
        maximum = thread_limits[kind];
    if (maximum <= 1) return 1;
    return tasks > 0 && tasks < maximum ? tasks : maximum;
#else
    (void)kind;
    (void)work;
    (void)default_limit;
    (void)tasks;
    return 1;
#endif
}

void tensorlib_parallel_report(FILE* stream)
{
    if (stream == NULL) return;
#ifdef _OPENMP
    initialize_policy();
    fprintf(stream, "  TensorLib policy (0 = OpenMP/caller default):\n");
    for (int index = 0; index < TENSORLIB_PARALLEL_COUNT; ++index) {
        fprintf(stream, "    %s: threads=%d min_%s=%lld\n",
                descriptions[index].name, thread_limits[index],
                descriptions[index].flops ? "flops" : "elements",
                work_limits[index]);
    }
#else
    fprintf(stream, "  TensorLib policy: serial (OpenMP disabled)\n");
#endif
}
