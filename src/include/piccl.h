#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include <cuda_runtime_api.h>   /* only header we need from CUDA */
#include <nccl.h>
#include <nvToolsExt.h>

typedef enum {
    RING,
    K_RING,
    RECURSIVE_DOUBLING,
    RECURSIVE_MULTIPLY,
    PERMUTED_RECURSIVE_DOUBLING,
    PERMUTED_RECURSIVE_MULTIPLY,
    DEFAULT_NCCL,
} piccl_algo_t;

/* --- Symbol resolution helpers ----------------------------------------- */
#define DECLARE_REAL(sym) static __typeof(&(sym)) real_##sym __attribute__((unused)) = NULL
#define RESOLVE_REAL(sym) do { if (!real_##sym) real_##sym = (__typeof(&(sym)))dlsym(RTLD_NEXT, #sym); } while(0)

DECLARE_REAL(ncclAllGather);        /* original, if present */
DECLARE_REAL(ncclBroadcast);
DECLARE_REAL(ncclAllReduce);
DECLARE_REAL(ncclReduce);
DECLARE_REAL(ncclReduceScatter);
DECLARE_REAL(ncclCommInitRank);
DECLARE_REAL(ncclCommDestroy);

/* --- Environemnt variables ------------------------------------------- */
#define N_COLLS 5
#define N_ALGOS 4

extern piccl_algo_t ALL_GATHER_ALGO;
extern piccl_algo_t ALL_REDUCE_ALGO;
extern piccl_algo_t REDUCE_ALGO;
extern piccl_algo_t REDUCE_SCATTER_ALGO;
extern piccl_algo_t BROADCAST_ALGO;

extern int ALL_GATHER_K;
extern int ALL_REDUCE_K;
extern int REDUCE_K;
extern int REDUCE_SCATTER_K;
extern int BROADCAST_K;

extern unsigned long globalBlockSize;
extern unsigned long globalMaxGridSize;
extern unsigned long globalMinGridSize;

/* --- GPU Device Buffers ------------------------------------------------*/
#define DATA_BUFFER_SIZE (8ULL * 1024ULL * 1024ULL * 1024ULL) // 4GB
#define INDEX_BUFFER_SIZE (1024ULL * 8ULL) // 8KB
extern void* reductionBuffer;
extern void* permutationBuffer;
extern void* indexBuffer;
extern void* cpuIndexBuffer;


/* --- Helper functions --------------------------------------------------*/
inline void ncclDataTypeSize(ncclDataType_t datatype, size_t* size) {
    switch (datatype) {
        case ncclChar:
            *size = sizeof(char);
            break;
        case ncclUint8:
            *size = sizeof(uint8_t);
            break;
        case ncclInt:
            *size = sizeof(int32_t);
            break;
        case ncclUint32:
            *size = sizeof(uint32_t);
            break;
        case ncclInt64:
            *size = sizeof(int64_t);
            break;
        case ncclUint64:
            *size = sizeof(uint64_t);
            break;
        case ncclFloat:
            *size = sizeof(float);
            break;
        case ncclDouble:
            *size = sizeof(double);
            break;
        case ncclFloat16:
            *size = 2;
            break;
        default:
            *size = 0;
            fprintf(stderr, "Unsupported data type\n");
    }
}