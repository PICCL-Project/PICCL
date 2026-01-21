#pragma once

#include <piccl.h>

ncclResult_t allgather_ring(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t allgather_k_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, int k);

ncclResult_t allgather_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t allgather_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, int k);

ncclResult_t allgather_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t allgather_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, int k);