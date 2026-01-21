#pragma once

#include <piccl.h>

ncclResult_t broadcast_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t broadcast_k_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, int k);

ncclResult_t broadcast_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t broadcast_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, int k);