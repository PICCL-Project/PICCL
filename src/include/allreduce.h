#pragma once

#include <piccl.h>

ncclResult_t allreduce_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t allreduce_k_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream, int k);

ncclResult_t allreduce_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, 
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t allreduce_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, 
    ncclComm_t comm, cudaStream_t stream, int k);

ncclResult_t allreduce_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t allreduce_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream, int k);