#include <reduce_scatter.h>

inline ncclResult_t reduce_scatter_ring(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) 
{
    printf("TODO: REDUCE_SCATTER RING\n");
    return ncclSuccess;
}

inline ncclResult_t reduce_scatter_k_ring(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int k)
{
    printf("TODO: REDUCE_SCATTER K RING\n");
    return ncclSuccess;
}

inline ncclResult_t reduce_scatter_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    printf("TODO: REDUCE_SCATTER PERMUTED RECURSIVE DOUBLING\n");
    return ncclSuccess;
}

inline ncclResult_t reduce_scatter_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int k)
{
    printf("TODO: REDUCE_SCATTER PERMUTED RECURSIVE MULTIPLY\n");
    return ncclSuccess;
}