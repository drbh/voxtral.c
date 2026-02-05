/*
 * voxtral_metal.m - Metal GPU acceleration for Voxtral inference
 *
 * MPS-accelerated matrix multiplication with bf16->f16 weight caching
 * and activation buffer pooling. Ported from flux-2-4b.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "voxtral_metal.h"
#include "voxtral_shaders_source.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

extern int vox_verbose;

/* ========================================================================
 * Global Metal State
 * ======================================================================== */

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static int g_initialized = 0;

/* Compute shader pipelines */
static id<MTLLibrary> g_shader_library = nil;
static id<MTLComputePipelineState> g_rms_norm_pipeline = nil;
static id<MTLComputePipelineState> g_silu_pipeline = nil;
static id<MTLComputePipelineState> g_gelu_pipeline = nil;
static id<MTLComputePipelineState> g_add_inplace_pipeline = nil;
static id<MTLComputePipelineState> g_mul_inplace_pipeline = nil;
static id<MTLComputePipelineState> g_causal_softmax_pipeline = nil;
static int g_shaders_initialized = 0;

/* ========================================================================
 * BF16 -> F16 Conversion
 * MPS only supports mixed f32/f16 matmul, not f32/bf16.
 * We convert bf16 weights to f16 once and cache the result.
 * ======================================================================== */

static inline uint16_t bf16_to_f16(uint16_t bf16) {
    uint32_t sign = (bf16 >> 15) & 0x1;
    int32_t exp = (bf16 >> 7) & 0xFF;
    uint32_t mant = bf16 & 0x7F;

    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp == 0xFF) return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));

    int32_t new_exp = exp - 127 + 15;
    if (new_exp <= 0) return (uint16_t)(sign << 15);
    if (new_exp >= 31) return (uint16_t)((sign << 15) | 0x7C00);

    uint32_t new_mant = mant << 3;
    return (uint16_t)((sign << 15) | (new_exp << 10) | new_mant);
}

/* ========================================================================
 * F16 Weight Cache (bf16 converted to f16, cached by CPU pointer)
 * ======================================================================== */

#define F16_WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;
    id<MTLBuffer> gpu_buffer;
    size_t num_elements;
} f16_cache_entry_t;

static f16_cache_entry_t g_f16_cache[F16_WEIGHT_CACHE_SIZE];
static int g_f16_cache_count = 0;
static pthread_mutex_t g_f16_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> get_cached_bf16_as_f16_buffer(const uint16_t *weights, size_t num_elements) {
    pthread_mutex_lock(&g_f16_cache_mutex);

    for (int i = 0; i < g_f16_cache_count; i++) {
        if (g_f16_cache[i].cpu_ptr == weights) {
            id<MTLBuffer> buf = g_f16_cache[i].gpu_buffer;
            pthread_mutex_unlock(&g_f16_cache_mutex);
            return buf;
        }
    }

    /* Convert bf16 -> f16 */
    uint16_t *f16_data = (uint16_t *)malloc(num_elements * sizeof(uint16_t));
    if (!f16_data) {
        pthread_mutex_unlock(&g_f16_cache_mutex);
        return nil;
    }
    for (size_t i = 0; i < num_elements; i++) {
        f16_data[i] = bf16_to_f16(weights[i]);
    }

    size_t size = num_elements * sizeof(uint16_t);
    id<MTLBuffer> buf = [g_device newBufferWithBytes:f16_data
                                              length:size
                                             options:MTLResourceStorageModeShared];
    free(f16_data);

    if (buf && g_f16_cache_count < F16_WEIGHT_CACHE_SIZE) {
        g_f16_cache[g_f16_cache_count].cpu_ptr = weights;
        g_f16_cache[g_f16_cache_count].gpu_buffer = buf;
        g_f16_cache[g_f16_cache_count].num_elements = num_elements;
        g_f16_cache_count++;
    }

    pthread_mutex_unlock(&g_f16_cache_mutex);
    return buf;
}

static void clear_f16_cache(void) {
    pthread_mutex_lock(&g_f16_cache_mutex);
    for (int i = 0; i < g_f16_cache_count; i++) {
        g_f16_cache[i].gpu_buffer = nil;
        g_f16_cache[i].cpu_ptr = NULL;
    }
    g_f16_cache_count = 0;
    pthread_mutex_unlock(&g_f16_cache_mutex);
}

/* ========================================================================
 * F32 Weight Cache (for bias and norm weight buffers)
 * ======================================================================== */

#define WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;
    id<MTLBuffer> gpu_buffer;
    size_t size;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];
static int g_weight_cache_count = 0;
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> get_cached_weight_buffer(const float *weights, size_t size) {
    pthread_mutex_lock(&g_cache_mutex);

    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr == weights && g_weight_cache[i].size == size) {
            id<MTLBuffer> buf = g_weight_cache[i].gpu_buffer;
            pthread_mutex_unlock(&g_cache_mutex);
            return buf;
        }
    }

    if (g_weight_cache_count >= WEIGHT_CACHE_SIZE) {
        pthread_mutex_unlock(&g_cache_mutex);
        return [g_device newBufferWithBytes:weights
                                     length:size
                                    options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> buf = [g_device newBufferWithBytes:weights
                                              length:size
                                             options:MTLResourceStorageModeShared];
    if (buf) {
        g_weight_cache[g_weight_cache_count].cpu_ptr = weights;
        g_weight_cache[g_weight_cache_count].gpu_buffer = buf;
        g_weight_cache[g_weight_cache_count].size = size;
        g_weight_cache_count++;
    }

    pthread_mutex_unlock(&g_cache_mutex);
    return buf;
}

static void clear_weight_cache(void) {
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < g_weight_cache_count; i++) {
        g_weight_cache[i].gpu_buffer = nil;
        g_weight_cache[i].cpu_ptr = NULL;
    }
    g_weight_cache_count = 0;
    pthread_mutex_unlock(&g_cache_mutex);
}

/* ========================================================================
 * Activation Buffer Pool
 * ======================================================================== */

#define ACTIVATION_POOL_SIZE 64

typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
    int in_use;
} pool_buffer_t;

static pool_buffer_t g_activation_pool[ACTIVATION_POOL_SIZE];
static int g_pool_count = 0;
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> pool_get_buffer(size_t size) {
    pthread_mutex_lock(&g_pool_mutex);

    for (int i = 0; i < g_pool_count; i++) {
        if (!g_activation_pool[i].in_use && g_activation_pool[i].size >= size) {
            g_activation_pool[i].in_use = 1;
            id<MTLBuffer> buf = g_activation_pool[i].buffer;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    if (g_pool_count < ACTIVATION_POOL_SIZE) {
        size_t alloc_size = size;
        if (alloc_size < 1024 * 1024) {
            alloc_size = ((alloc_size + 65535) / 65536) * 65536;
        } else {
            alloc_size = ((alloc_size + 1048575) / 1048576) * 1048576;
        }

        id<MTLBuffer> buf = [g_device newBufferWithLength:alloc_size
                                                  options:MTLResourceStorageModeShared];
        if (buf) {
            g_activation_pool[g_pool_count].buffer = buf;
            g_activation_pool[g_pool_count].size = alloc_size;
            g_activation_pool[g_pool_count].in_use = 1;
            g_pool_count++;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    pthread_mutex_unlock(&g_pool_mutex);
    return [g_device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

static void pool_release_buffer(id<MTLBuffer> buffer) {
    if (!buffer) return;
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        if (g_activation_pool[i].buffer == buffer) {
            g_activation_pool[i].in_use = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_pool_mutex);
}

static void clear_activation_pool(void) {
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        g_activation_pool[i].buffer = nil;
        g_activation_pool[i].in_use = 0;
        g_activation_pool[i].size = 0;
    }
    g_pool_count = 0;
    pthread_mutex_unlock(&g_pool_mutex);
}

/* ========================================================================
 * MPS Matmul Operator Cache
 * Reuse MPSMatrixMultiplication objects across calls with same shape/config.
 * ======================================================================== */

static NSMutableDictionary *g_matmul_op_cache = nil;
static pthread_mutex_t g_matmul_op_mutex = PTHREAD_MUTEX_INITIALIZER;

static MPSMatrixMultiplication *get_cached_matmul_op(BOOL transposeLeft, BOOL transposeRight,
                                                      int resultRows, int resultColumns,
                                                      int interiorColumns,
                                                      double alpha, double beta) {
    pthread_mutex_lock(&g_matmul_op_mutex);
    if (!g_matmul_op_cache) g_matmul_op_cache = [NSMutableDictionary new];

    NSString *key = [NSString stringWithFormat:@"%d:%d:%d:%d:%d:%.9g:%.9g",
                                               (int)transposeLeft, (int)transposeRight,
                                               resultRows, resultColumns, interiorColumns,
                                               alpha, beta];
    MPSMatrixMultiplication *mm = [g_matmul_op_cache objectForKey:key];
    if (!mm) {
        mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transposeLeft
              transposeRight:transposeRight
                  resultRows:resultRows
               resultColumns:resultColumns
             interiorColumns:interiorColumns
                       alpha:alpha
                        beta:beta];
        if (mm) [g_matmul_op_cache setObject:mm forKey:key];
    }

    pthread_mutex_unlock(&g_matmul_op_mutex);
    return mm;
}

static void clear_matmul_op_cache(void) {
    pthread_mutex_lock(&g_matmul_op_mutex);
    g_matmul_op_cache = nil;
    pthread_mutex_unlock(&g_matmul_op_mutex);
}

/* ========================================================================
 * Shader Compilation
 * ======================================================================== */

static int init_shaders(void) {
    if (g_shaders_initialized) return 1;
    if (!g_initialized) return 0;

    @autoreleasepool {
        NSError *error = nil;

        NSString *shaderSource = [[NSString alloc]
            initWithBytes:voxtral_shaders_metal
                   length:voxtral_shaders_metal_len
                 encoding:NSUTF8StringEncoding];

        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        g_shader_library = [g_device newLibraryWithSource:shaderSource
                                                  options:options
                                                    error:&error];
        if (!g_shader_library) {
            fprintf(stderr, "Metal shaders: compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        /* Create pipelines for each kernel */
        id<MTLFunction> func;

        func = [g_shader_library newFunctionWithName:@"rms_norm"];
        if (func) g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"silu"];
        if (func) g_silu_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"gelu"];
        if (func) g_gelu_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"add_inplace"];
        if (func) g_add_inplace_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"mul_inplace"];
        if (func) g_mul_inplace_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"causal_softmax"];
        if (func) g_causal_softmax_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        g_shaders_initialized = 1;

        if (vox_verbose >= 2) {
            fprintf(stderr, "Metal: compute shaders compiled (%s%s%s%s%s%s)\n",
                    g_rms_norm_pipeline ? "rms_norm " : "",
                    g_silu_pipeline ? "silu " : "",
                    g_gelu_pipeline ? "gelu " : "",
                    g_add_inplace_pipeline ? "add " : "",
                    g_mul_inplace_pipeline ? "mul " : "",
                    g_causal_softmax_pipeline ? "causal_softmax " : "");
        }
    }

    return 1;
}

/* ========================================================================
 * Metal Initialization
 * ======================================================================== */

int vox_metal_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return 0;

        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            if (![g_device supportsFamily:MTLGPUFamilyApple6]) {
                g_device = nil;
                return 0;
            }
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            g_device = nil;
            return 0;
        }

        g_initialized = 1;
        if (vox_verbose >= 2)
            fprintf(stderr, "Metal: GPU acceleration enabled (%s)\n",
                    [[g_device name] UTF8String]);

        init_shaders();
    }

    return 1;
}

int vox_metal_available(void) {
    return g_initialized;
}

void vox_metal_shutdown(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        clear_f16_cache();
        clear_weight_cache();
        clear_activation_pool();
        clear_matmul_op_cache();

        g_rms_norm_pipeline = nil;
        g_silu_pipeline = nil;
        g_gelu_pipeline = nil;
        g_add_inplace_pipeline = nil;
        g_mul_inplace_pipeline = nil;
        g_causal_softmax_pipeline = nil;
        g_shader_library = nil;
        g_shaders_initialized = 0;

        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

/* ========================================================================
 * MPS Matrix Multiplication (bf16 weights)
 *
 * C[M,N] = A[M,K] @ B_bf16[N,K]^T
 * A is f32, B_bf16 is bf16 (converted to f16 and cached), C is f32.
 * ======================================================================== */

void vox_metal_sgemm_bf16(int M, int N, int K,
                           const float *A,
                           const uint16_t *B_bf16,
                           float *C) {
    if (!g_initialized) return;

    @autoreleasepool {
        size_t sizeA = (size_t)M * K * sizeof(float);
        size_t numB = (size_t)N * K;
        size_t sizeC = (size_t)M * N * sizeof(float);

        /* Get cached f16 weight buffer */
        id<MTLBuffer> bufferB = get_cached_bf16_as_f16_buffer(B_bf16, numB);

        /* Activation buffers from pool */
        id<MTLBuffer> bufferA = pool_get_buffer(sizeA);
        if (bufferA) memcpy([bufferA contents], A, sizeA);

        id<MTLBuffer> bufferC = pool_get_buffer(sizeC);

        if (!bufferA || !bufferB || !bufferC) {
            if (bufferA) pool_release_buffer(bufferA);
            if (bufferC) pool_release_buffer(bufferC);
            return;
        }

        /* Matrix descriptors:
         * A: [M, K] f32, row-major
         * B: [N, K] f16, row-major (MPS transposes it)
         * C: [M, N] f32, row-major */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:N columns:K
                            rowBytes:K * sizeof(uint16_t)
                            dataType:MPSDataTypeFloat16];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:N * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        /* C = A @ B^T */
        MPSMatrixMultiplication *matmul =
            get_cached_matmul_op(NO, YES, M, N, K, 1.0, 0.0);
        if (!matmul) {
            pool_release_buffer(bufferA);
            pool_release_buffer(bufferC);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(C, [bufferC contents], sizeC);

        pool_release_buffer(bufferA);
        pool_release_buffer(bufferC);
    }
}

/* ========================================================================
 * MPS Matrix Multiplication (f32 weights)
 *
 * C[M,N] = A[M,K] @ B[N,K]^T
 * ======================================================================== */

void vox_metal_sgemm(int M, int N, int K,
                     const float *A,
                     const float *B,
                     float *C) {
    if (!g_initialized) return;

    @autoreleasepool {
        size_t sizeA = (size_t)M * K * sizeof(float);
        size_t sizeB = (size_t)N * K * sizeof(float);
        size_t sizeC = (size_t)M * N * sizeof(float);

        id<MTLBuffer> bufferB = get_cached_weight_buffer(B, sizeB);

        id<MTLBuffer> bufferA = pool_get_buffer(sizeA);
        if (bufferA) memcpy([bufferA contents], A, sizeA);

        id<MTLBuffer> bufferC = pool_get_buffer(sizeC);

        if (!bufferA || !bufferB || !bufferC) {
            if (bufferA) pool_release_buffer(bufferA);
            if (bufferC) pool_release_buffer(bufferC);
            return;
        }

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:N columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:N * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        MPSMatrixMultiplication *matmul =
            get_cached_matmul_op(NO, YES, M, N, K, 1.0, 0.0);
        if (!matmul) {
            pool_release_buffer(bufferA);
            pool_release_buffer(bufferC);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(C, [bufferC contents], sizeC);

        pool_release_buffer(bufferA);
        pool_release_buffer(bufferC);
    }
}

/* ========================================================================
 * Fused QKV: 3 matmuls in one command buffer, shared input
 *
 * q[M,Nq] = input[M,K] @ wq[Nq,K]^T
 * k[M,Nk] = input[M,K] @ wk[Nk,K]^T
 * v[M,Nv] = input[M,K] @ wv[Nv,K]^T
 * ======================================================================== */

void vox_metal_fused_qkv_bf16(int M, int K,
                                const float *input,
                                const uint16_t *wq_bf16, int Nq,
                                const uint16_t *wk_bf16, int Nk,
                                const uint16_t *wv_bf16, int Nv,
                                float *q_out, float *k_out, float *v_out) {
    if (!g_initialized) return;

    @autoreleasepool {
        size_t sizeInput = (size_t)M * K * sizeof(float);
        size_t sizeQ = (size_t)M * Nq * sizeof(float);
        size_t sizeK = (size_t)M * Nk * sizeof(float);
        size_t sizeV = (size_t)M * Nv * sizeof(float);

        /* Cached f16 weight buffers */
        id<MTLBuffer> bufWq = get_cached_bf16_as_f16_buffer(wq_bf16, (size_t)Nq * K);
        id<MTLBuffer> bufWk = get_cached_bf16_as_f16_buffer(wk_bf16, (size_t)Nk * K);
        id<MTLBuffer> bufWv = get_cached_bf16_as_f16_buffer(wv_bf16, (size_t)Nv * K);

        /* One input copy */
        id<MTLBuffer> bufInput = pool_get_buffer(sizeInput);
        if (bufInput) memcpy([bufInput contents], input, sizeInput);

        /* Output buffers */
        id<MTLBuffer> bufQ = pool_get_buffer(sizeQ);
        id<MTLBuffer> bufK = pool_get_buffer(sizeK);
        id<MTLBuffer> bufV = pool_get_buffer(sizeV);

        if (!bufInput || !bufWq || !bufWk || !bufWv || !bufQ || !bufK || !bufV) {
            pool_release_buffer(bufInput);
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            return;
        }

        /* Shared input descriptor */
        MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:bufInput descriptor:descInput];

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Q = Input @ Wq^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nq columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nq
                                rowBytes:Nq * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWq descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufQ descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nq, K, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufQ);
                pool_release_buffer(bufK);
                pool_release_buffer(bufV);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        /* K = Input @ Wk^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nk columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nk
                                rowBytes:Nk * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWk descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufK descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nk, K, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufQ);
                pool_release_buffer(bufK);
                pool_release_buffer(bufV);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        /* V = Input @ Wv^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nv columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nv
                                rowBytes:Nv * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWv descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufV descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nv, K, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufQ);
                pool_release_buffer(bufK);
                pool_release_buffer(bufV);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(q_out, [bufQ contents], sizeQ);
        memcpy(k_out, [bufK contents], sizeK);
        memcpy(v_out, [bufV contents], sizeV);

        pool_release_buffer(bufInput);
        pool_release_buffer(bufQ);
        pool_release_buffer(bufK);
        pool_release_buffer(bufV);
    }
}

/* ========================================================================
 * Fused SwiGLU FFN: w1+w3+silu+mul+w2 in one command buffer
 *
 * gate = silu(input @ w1^T)
 * up = input @ w3^T
 * output = (gate * up) @ w2^T
 *
 * All intermediate buffers stay on GPU. Only input is copied in,
 * only output is copied out.
 * ======================================================================== */

void vox_metal_fused_ffn_bf16(int M, int dim, int hidden,
                               const float *input,
                               const uint16_t *w1_bf16,
                               const uint16_t *w3_bf16,
                               const uint16_t *w2_bf16,
                               float *output) {
    if (!g_initialized || !g_shaders_initialized) return;

    @autoreleasepool {
        size_t sizeInput = (size_t)M * dim * sizeof(float);
        size_t sizeHidden = (size_t)M * hidden * sizeof(float);
        size_t sizeOutput = (size_t)M * dim * sizeof(float);

        /* Cached f16 weight buffers */
        id<MTLBuffer> bufW1 = get_cached_bf16_as_f16_buffer(w1_bf16, (size_t)hidden * dim);
        id<MTLBuffer> bufW3 = get_cached_bf16_as_f16_buffer(w3_bf16, (size_t)hidden * dim);
        id<MTLBuffer> bufW2 = get_cached_bf16_as_f16_buffer(w2_bf16, (size_t)dim * hidden);

        /* Activation buffers */
        id<MTLBuffer> bufInput = pool_get_buffer(sizeInput);
        if (bufInput) memcpy([bufInput contents], input, sizeInput);

        id<MTLBuffer> bufGate = pool_get_buffer(sizeHidden);
        id<MTLBuffer> bufUp = pool_get_buffer(sizeHidden);
        id<MTLBuffer> bufOutput = pool_get_buffer(sizeOutput);

        if (!bufInput || !bufW1 || !bufW3 || !bufW2 ||
            !bufGate || !bufUp || !bufOutput) {
            pool_release_buffer(bufInput);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufUp);
            pool_release_buffer(bufOutput);
            return;
        }

        /* Shared input matrix descriptor (for w1 and w3) */
        MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:dim
                            rowBytes:dim * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:bufInput descriptor:descInput];

        /* Hidden output descriptor (shared shape for gate and up) */
        MPSMatrixDescriptor *descHidden = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:hidden
                            rowBytes:hidden * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Step 1: gate = input @ w1^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:hidden columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW1 descriptor:descW];
            MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descHidden];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, hidden, dim, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufGate);
                pool_release_buffer(bufUp);
                pool_release_buffer(bufOutput);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matGate];
        }

        /* Step 2: up = input @ w3^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:hidden columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW3 descriptor:descW];
            MPSMatrix *matUp = [[MPSMatrix alloc] initWithBuffer:bufUp descriptor:descHidden];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, hidden, dim, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufGate);
                pool_release_buffer(bufUp);
                pool_release_buffer(bufOutput);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matUp];
        }

        /* Step 3: silu(gate) - GPU compute shader */
        {
            int n = M * hidden;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_silu_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBytes:&n length:sizeof(int) atIndex:1];
            NSUInteger tgSize = MIN((NSUInteger)n, g_silu_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 4: gate *= up - GPU compute shader */
        {
            int n = M * hidden;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_mul_inplace_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBuffer:bufUp offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            NSUInteger tgSize = MIN((NSUInteger)n, g_mul_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 5: output = gate @ w2^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:dim columns:hidden
                                rowBytes:hidden * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descHidden];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW2 descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufOutput descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, dim, hidden, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufGate);
                pool_release_buffer(bufUp);
                pool_release_buffer(bufOutput);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matGate rightMatrix:matW resultMatrix:matOut];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(output, [bufOutput contents], sizeOutput);

        pool_release_buffer(bufInput);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufUp);
        pool_release_buffer(bufOutput);
    }
}

/* ========================================================================
 * GPU Batched Attention
 *
 * All heads processed in one command buffer:
 *   1. QK^T matmul per head (strided views, alpha=scale)
 *   2. Causal masked softmax (compute shader)
 *   3. scores * V matmul per head (strided views)
 *
 * Q:   [seq_q, n_heads * head_dim]
 * K:   [seq_k, n_kv_heads * head_dim]
 * V:   [seq_k, n_kv_heads * head_dim]
 * out: [seq_q, n_heads * head_dim]
 * ======================================================================== */

void vox_metal_batched_attention(float *out,
                                  const float *Q, const float *K, const float *V,
                                  int seq_q, int seq_k,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, float scale,
                                  int window_size, int q_offset) {
    if (!g_initialized || !g_causal_softmax_pipeline) return;

    @autoreleasepool {
        int gqa_ratio = n_heads / n_kv_heads;
        size_t q_total = (size_t)seq_q * n_heads * head_dim;
        size_t k_total = (size_t)seq_k * n_kv_heads * head_dim;
        size_t scores_total = (size_t)n_heads * seq_q * seq_k;
        size_t out_total = q_total;

        /* Copy Q, K, V to GPU */
        id<MTLBuffer> bufQ = pool_get_buffer(q_total * sizeof(float));
        id<MTLBuffer> bufK = pool_get_buffer(k_total * sizeof(float));
        id<MTLBuffer> bufV = pool_get_buffer(k_total * sizeof(float));
        id<MTLBuffer> bufScores = pool_get_buffer(scores_total * sizeof(float));
        id<MTLBuffer> bufOut = pool_get_buffer(out_total * sizeof(float));

        if (!bufQ || !bufK || !bufV || !bufScores || !bufOut) {
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            pool_release_buffer(bufScores);
            pool_release_buffer(bufOut);
            return;
        }

        memcpy([bufQ contents], Q, q_total * sizeof(float));
        memcpy([bufK contents], K, k_total * sizeof(float));
        memcpy([bufV contents], V, k_total * sizeof(float));

        /* Row strides for packed [seq, heads * head_dim] layout */
        size_t q_row_bytes = (size_t)n_heads * head_dim * sizeof(float);
        size_t kv_row_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);
        size_t scores_row_bytes = (size_t)seq_k * sizeof(float);
        size_t out_row_bytes = q_row_bytes;

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        MPSMatrixMultiplication *mm_qk =
            get_cached_matmul_op(NO, YES, seq_q, seq_k, head_dim, (double)scale, 0.0);
        MPSMatrixMultiplication *mm_sv =
            get_cached_matmul_op(NO, NO, seq_q, head_dim, seq_k, 1.0, 0.0);
        if (!mm_qk || !mm_sv) {
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            pool_release_buffer(bufScores);
            pool_release_buffer(bufOut);
            return;
        }

        /* --- Step 1: QK^T per head --- */
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa_ratio;

            /* Q_h: strided view into packed Q */
            MPSMatrixDescriptor *descQh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:q_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matQh = [[MPSMatrix alloc]
                initWithBuffer:bufQ
                        offset:(size_t)h * head_dim * sizeof(float)
                    descriptor:descQh];

            /* K_h: strided view into packed K */
            MPSMatrixDescriptor *descKh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:kv_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matKh = [[MPSMatrix alloc]
                initWithBuffer:bufK
                        offset:(size_t)kv_h * head_dim * sizeof(float)
                    descriptor:descKh];

            /* scores_h: contiguous [seq_q, seq_k] */
            MPSMatrixDescriptor *descSh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:scores_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matSh = [[MPSMatrix alloc]
                initWithBuffer:bufScores
                        offset:(size_t)h * seq_q * seq_k * sizeof(float)
                    descriptor:descSh];

            /* scores_h = scale * Q_h @ K_h^T */
            [mm_qk encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matQh rightMatrix:matKh resultMatrix:matSh];
        }

        /* --- Step 2: Causal masked softmax --- */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_causal_softmax_pipeline];
            [enc setBuffer:bufScores offset:0 atIndex:0];
            [enc setBytes:&seq_q length:sizeof(int) atIndex:1];
            [enc setBytes:&seq_k length:sizeof(int) atIndex:2];
            [enc setBytes:&window_size length:sizeof(int) atIndex:3];
            [enc setBytes:&q_offset length:sizeof(int) atIndex:4];
            /* 1D grid: one threadgroup per (head * seq_q + qi) */
            NSUInteger total_groups = (NSUInteger)n_heads * (NSUInteger)seq_q;
            [enc dispatchThreadgroups:MTLSizeMake(total_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        /* --- Step 3: scores * V per head --- */
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa_ratio;

            /* scores_h: contiguous [seq_q, seq_k] */
            MPSMatrixDescriptor *descSh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:scores_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matSh = [[MPSMatrix alloc]
                initWithBuffer:bufScores
                        offset:(size_t)h * seq_q * seq_k * sizeof(float)
                    descriptor:descSh];

            /* V_h: strided view into packed V */
            MPSMatrixDescriptor *descVh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:kv_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matVh = [[MPSMatrix alloc]
                initWithBuffer:bufV
                        offset:(size_t)kv_h * head_dim * sizeof(float)
                    descriptor:descVh];

            /* out_h: strided view into packed output */
            MPSMatrixDescriptor *descOh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:out_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matOh = [[MPSMatrix alloc]
                initWithBuffer:bufOut
                        offset:(size_t)h * head_dim * sizeof(float)
                    descriptor:descOh];

            /* out_h = scores_h @ V_h */
            [mm_sv encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matSh rightMatrix:matVh resultMatrix:matOh];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(out, [bufOut contents], out_total * sizeof(float));

        pool_release_buffer(bufQ);
        pool_release_buffer(bufK);
        pool_release_buffer(bufV);
        pool_release_buffer(bufScores);
        pool_release_buffer(bufOut);
    }
}

/* ========================================================================
 * Utility
 * ======================================================================== */

void vox_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    if (!g_initialized || !bf16_weights || num_elements == 0) return;
    (void)get_cached_bf16_as_f16_buffer(bf16_weights, num_elements);
}

size_t vox_metal_memory_used(void) {
    if (!g_initialized) return 0;
    size_t total = 0;
    pthread_mutex_lock(&g_f16_cache_mutex);
    for (int i = 0; i < g_f16_cache_count; i++)
        total += g_f16_cache[i].num_elements * sizeof(uint16_t);
    pthread_mutex_unlock(&g_f16_cache_mutex);
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < g_weight_cache_count; i++)
        total += g_weight_cache[i].size;
    pthread_mutex_unlock(&g_cache_mutex);
    return total;
}
