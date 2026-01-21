#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include <piccl.h>

#ifdef __cplusplus
extern "C" {
#endif

void picclPermuteInplace(void* permuteBuff, void* indices, size_t totalSize, size_t count, 
    ncclDataType_t datatype, cudaStream_t stream);

void picclPermute(void* permuteBuff, const void* inBuff, void* indices, size_t totalSize, size_t count, 
    ncclDataType_t datatype, cudaStream_t stream);

void launchComputePermutationKernel(void *d_perm, int r, int p, int b,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif