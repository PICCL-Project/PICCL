#include <broadcast.h>

inline ncclResult_t broadcast_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    int sendRank = (rank + 1) % nRanks;
    int recvRank = (rank - 1 + nRanks) % nRanks;

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (!((char*) sendbuff == (char*) recvbuff)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
            sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
    }

    // Receive data from the previous rank
    if (rank != root){
        ncclRecv(recvbuff, count, datatype, recvRank, comm, stream);
    }

    // Send data to the next rank
    if (rank != (root - 1 + nRanks) % nRanks){
        ncclSend(recvbuff, count, datatype, sendRank, comm, stream);
    }

    return ncclSuccess;
}

inline ncclResult_t broadcast_k_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, int k)
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    int relRankRoot = root % k;
    int relRank = rank % k;
    int groupOffset = rank / k * k;

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (!((char*) sendbuff == (char*) recvbuff)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
            sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
    }

    // Intergroup communication
    if (relRank == relRankRoot) {
        int sendRank = (rank + k) % nRanks;
        int recvRank = (rank - k + nRanks) % nRanks;

        if (rank != root) {
            // Receive data from the previous rank
            ncclRecv(recvbuff, count, datatype, recvRank, comm, stream);
        } 

        ncclGroupStart();
        if (sendRank != root) {
            // Send data to the next rank
            ncclSend(recvbuff, count, datatype, sendRank, comm, stream);
        }
        if (k > 1) {
            int interSendRank = groupOffset + (relRank + 1) % k;
            ncclSend(recvbuff, count, datatype, interSendRank, comm, stream);
        }
        ncclGroupEnd();
    
    } else { // Intra-group communication
        int sendRank = groupOffset + (relRank + 1) % k;
        int recvRank = groupOffset + (relRank - 1 + k) % k;

        ncclRecv(recvbuff, count, datatype, recvRank, comm, stream);

        if (sendRank % k != relRankRoot) {
            ncclSend(recvbuff, count, datatype, sendRank, comm, stream);
        }        
    }

    return ncclSuccess;
}

inline ncclResult_t broadcast_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    printf("TODO: BROADCAST PERMUTED RECURSIVE DOUBLING\n");
    return ncclSuccess;
}

inline ncclResult_t broadcast_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream, int k)
{
    printf("TODO: BROADCAST PERMUTED RECURSIVE MULTIPLY\n");
    return ncclSuccess;
}