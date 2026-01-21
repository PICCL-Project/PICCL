#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#ifdef __cplusplus
extern "C" {
#endif

void picclReduce(void* redBuff, const void* op1, const void* op2, size_t count, size_t sections,
    ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream);

#ifdef __cplusplus
}
#endif