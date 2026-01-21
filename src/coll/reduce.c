#include <reduce.h>
#include <reduce_kernel.h>

inline ncclResult_t reduce_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    int sendRank = (rank + 1) % nRanks;
    int recvRank = (rank - 1 + nRanks) % nRanks;

    void* tmpbuff = reductionBuffer;

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    if (rank == root) {
        // In-place opreations
        if (sendbuff != recvbuff) {
            // Copy data from sendbuff to recvbuff using cudaMemcpy 
            cudaMemcpy(recvbuff, sendbuff, count * datatypeSize, 
                cudaMemcpyDeviceToDevice);
        }

        ncclRecv(tmpbuff, count, datatype, recvRank, comm, stream);
        picclReduce(recvbuff, recvbuff, tmpbuff, count, 1, datatype, op, stream);


    } else {
        if (rank != (root + 1) % nRanks){
            ncclRecv(tmpbuff, count, datatype, recvRank, comm, stream);
            picclReduce(tmpbuff, tmpbuff, sendbuff, count, 1, datatype, op, stream);
            // Send data to the next rank
            ncclSend(tmpbuff, count, datatype, sendRank, comm, stream);
        } else {
            // If the rank is the next rank of the root, just send the data
            ncclSend(sendbuff, count, datatype, sendRank, comm, stream);
        }
    
    }

    return ncclSuccess;
}

inline ncclResult_t reduce_k_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int k)
{
    printf("TODO: REDUCE K RING\n");
    return ncclSuccess;
}

inline ncclResult_t reduce_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    printf("TODO: REDUCE PERMUTED RECURSIVE DOUBLING\n");
    return ncclSuccess;
}

inline ncclResult_t reduce_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int k)
{
    printf("TODO: REDUCE PERMUTED RECURSIVE MULTIPLY\n");
    return ncclSuccess;
}