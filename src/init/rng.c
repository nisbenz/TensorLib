#include <math.h>
#include <stdint.h>

#include "../../include/tensorlib/nn.h"

#define NN_RNG_INCREMENT UINT64_C(0x9E3779B97F4A7C15)
#define NN_RNG_FLOAT_SCALE (1.0f / 16777216.0f)

static uint64_t nn_rng_next_u64(nn_rng* rng)
{
    uint64_t value;

    rng->state += NN_RNG_INCREMENT;
    value = rng->state;
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

static float nn_rng_unit(nn_rng* rng)
{
    return (float)(nn_rng_next_u64(rng) >> 40) * NN_RNG_FLOAT_SCALE;
}

void nn_rng_seed(nn_rng* rng, uint64_t seed)
{
    if (rng != NULL) rng->state = seed;
}

float nn_rng_uniform(nn_rng* rng, float min, float max)
{
    if (rng == NULL || !isfinite(min) || !isfinite(max) || min > max) {
        return NAN;
    }
    if (min == max) return min;
    return min + (max - min) * nn_rng_unit(rng);
}

float nn_rng_normal(nn_rng* rng, float mean, float stddev)
{
    const float two_pi = 6.28318530717958647692f;
    float u1;
    float u2;

    if (rng == NULL || !isfinite(mean) || !isfinite(stddev) || stddev < 0.0f) {
        return NAN;
    }
    if (stddev == 0.0f) return mean;

    u1 = ((float)(nn_rng_next_u64(rng) >> 40) + 1.0f) / 16777217.0f;
    u2 = nn_rng_unit(rng);
    return mean + stddev * sqrtf(-2.0f * logf(u1)) * cosf(two_pi * u2);
}
