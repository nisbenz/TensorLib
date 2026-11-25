#ifndef TENSORLIB_TENSOR_MATMUL_TUNING_H
#define TENSORLIB_TENSOR_MATMUL_TUNING_H

/* Private experiment controls. Defaults stay portable; MR/NR match the kernel. */
#ifndef TENSORLIB_MATMUL_CONFIG_MC
#define TENSORLIB_MATMUL_CONFIG_MC 64
#endif
#ifndef TENSORLIB_MATMUL_CONFIG_NC
#define TENSORLIB_MATMUL_CONFIG_NC 64
#endif
#ifndef TENSORLIB_MATMUL_CONFIG_KC
#define TENSORLIB_MATMUL_CONFIG_KC 128
#endif

#if TENSORLIB_MATMUL_CONFIG_MC < 4 || TENSORLIB_MATMUL_CONFIG_MC > 512 || \
    TENSORLIB_MATMUL_CONFIG_MC % 4 != 0
#error "Matmul MC must be a multiple of 4 in [4,512]"
#endif
#if TENSORLIB_MATMUL_CONFIG_NC < 16 || TENSORLIB_MATMUL_CONFIG_NC > 512 || \
    TENSORLIB_MATMUL_CONFIG_NC % 16 != 0
#error "Matmul NC must be a multiple of 16 in [16,512]"
#endif
#if TENSORLIB_MATMUL_CONFIG_KC < 1 || TENSORLIB_MATMUL_CONFIG_KC > 512
#error "Matmul KC must be in [1,512]"
#endif

#endif
