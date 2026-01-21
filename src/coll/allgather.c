#include <allgather.h>

inline ncclResult_t allgather_ring(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
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
    if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
            sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
    }

    int sendOffset = rank;
    int recvOffset = (rank - 1 + nRanks) % nRanks;
    // Perform the ring allgather operation
    for (int i = 0; i < nRanks - 1; ++i) {
        ncclGroupStart();
        
        // Send data to the next rank
        ncclSend((char*) recvbuff + sendOffset * count * datatypeSize, count, datatype, 
            sendRank, comm, stream);

        // Receive data from the previous rank
        ncclRecv((char*) recvbuff + recvOffset * count * datatypeSize, count, datatype, 
            recvRank, comm, stream);

        ncclGroupEnd();

        sendOffset = recvOffset;
        recvOffset = (recvOffset - 1 + nRanks) % nRanks;

    }

    return ncclSuccess;
}


inline ncclResult_t allgather_k_ring(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, int k) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    int groupOffset = rank / k * k;
    int relRank = rank % k;

    int intraSend = groupOffset + (relRank + 1) % k;
    int intraRecv = groupOffset + (relRank - 1 + k) % k;
    int interSend = (rank + k) % nRanks;
    int interRecv = (rank - k + nRanks) % nRanks;

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
            sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
    }

    int sendOffset = rank;
    int recvOffset = interRecv;
    // Perform inter-group communication
    for (int i = 0; i < nRanks / k - 1; ++i) {
        ncclGroupStart();
        
        // Send data to the next rank
        ncclSend((char*) recvbuff + sendOffset * count * datatypeSize, count, datatype, 
            interSend, comm, stream);

        // Receive data from the previous rank
        ncclRecv((char*) recvbuff + recvOffset * count * datatypeSize, count, datatype, 
            interRecv, comm, stream);

        ncclGroupEnd();

        sendOffset = recvOffset;
        recvOffset = (recvOffset - k + nRanks) % nRanks;
    }

    int offset = groupOffset;
    // Perform intra-group communication
    for (int i = 0; i < nRanks / k; ++i) {
        
        int relSendOffset = relRank;
        int relRecvOffset = (relRank - 1 + k) % k;

        for (int j = 0; j < k - 1; ++j) {
            ncclGroupStart();
            
            // Send data to the next rank
            ncclSend((char*) recvbuff + (offset + relSendOffset) * count * datatypeSize, count, datatype, 
                intraSend, comm, stream);

            // Receive data from the previous rank
            ncclRecv((char*) recvbuff + (offset + relRecvOffset) * count * datatypeSize, count, datatype, 
                intraRecv, comm, stream);

            ncclGroupEnd();

            relSendOffset = relRecvOffset;
            relRecvOffset = (relRecvOffset - 1 + k) % k;
        }
        offset = (offset - k + nRanks) % nRanks;
    }


    return ncclSuccess;
}


inline ncclResult_t allgather_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
            sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
    }

    int nElements = count;

    int mask = 0x1;
    int i = 0;

    while (mask < nRanks) {
        int dst = rank ^ mask;

        int dstTreeRoot = dst >> i;
        dstTreeRoot <<= i;

        int myTreeRoot = rank >> i;
        myTreeRoot <<= i;

        ncclGroupStart();

        ncclSend((char*) recvbuff + myTreeRoot * count * datatypeSize, nElements, datatype, dst, comm, stream);
        ncclRecv((char*) recvbuff + dstTreeRoot * count * datatypeSize, nElements, datatype, dst, comm, stream);

        ncclGroupEnd();

        nElements *= 2;
        i++;
        mask <<= 1;
    }
    
    return ncclSuccess;
}


inline ncclResult_t allgather_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, int k) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
            sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
    }

    size_t nElements = count;
    int radixRoot = rank;

    int radixSize = 1;
    int nextRadixSize = k;

    while (radixSize < nRanks) {
        int nextRadixRoot = (rank / nextRadixSize) * nextRadixSize;
        int radixLoc = (int) rank % radixSize;
        size_t sendOffset = radixRoot * count * datatypeSize;

        int dstRoot = nextRadixRoot;
        int dstRootLimit = nextRadixRoot + nextRadixSize;

        ncclGroupStart();
        while (dstRoot < dstRootLimit && dstRoot < nRanks) {
            int dst = dstRoot + radixLoc;

            if (dst != rank && dst < nRanks) {
                size_t recvOffset = dstRoot * count * datatypeSize;
                
                ncclSend((char*) recvbuff + sendOffset, nElements, datatype, dst, comm, stream);
                ncclRecv((char*) recvbuff + recvOffset, nElements, datatype, dst, comm, stream);
            }

            dstRoot += radixSize;
        }
        ncclGroupEnd();

        nElements *= k;

        radixRoot = nextRadixRoot;

        radixSize = nextRadixSize;
        nextRadixSize *= k;
    }

    return ncclSuccess;
}


inline ncclResult_t allgather_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);


    launchComputePermutationKernel(indexBuffer, rank, nRanks, 2, stream);

    // Check for in-place operation
    if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy. We don't need to
        // worry about indices right now since we are permuting the result at the end
        cudaMemcpyAsync(recvbuff, sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpyAsync(recvbuff, (char*) recvbuff + rank * count * datatypeSize, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    size_t offset = count * datatypeSize;
    size_t nElements = count;
    int size = nRanks;
    while (size > 1) {
        int root = rank / size * size;
        int relRank = rank % size;
        int nextSize = size / 2;

        int sendRank  = root + (relRank + nextSize) % size;
        int recvRank = sendRank;

        ncclGroupStart();

        ncclSend(recvbuff, nElements, datatype, sendRank, comm, stream);
        ncclRecv((char*) recvbuff + offset, nElements, datatype, recvRank, comm, stream);

        ncclGroupEnd();
        
        offset += nElements * datatypeSize;
        nElements *= 2;
        size /= 2;
    }

    picclPermuteInplace(recvbuff, indexBuffer, 
        (size_t) (nRanks * count), count, 
        datatype, stream);
    
    
    return ncclSuccess;
}

// Simulatneous send and recv
// inline ncclResult_t allgather_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count, 
//     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
// {
//     int rank, nRanks;
//     ncclCommUserRank(comm, &rank);
//     ncclCommCount(comm, &nRanks);

//     // Calculate the size of the data type
//     size_t datatypeSize;
//     ncclDataTypeSize(datatype, &datatypeSize);


//     launchComputePermutationKernel(indexBuffer, rank, nRanks, 2, stream);    

//     // Check for in-place operation
//     if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
//         // Copy data from sendbuff to recvbuff using cudaMemcpy 
//         cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
//             sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
//     }

//     cudaStreamSynchronize(stream);

//     cudaMemcpy(cpuIndexBuffer, indexBuffer, nRanks * sizeof(int), cudaMemcpyDeviceToHost);

//     int offset = count * datatypeSize;
//     int nElements = count;
//     int size = nRanks;
//     int numTransfers = 1;
//     while (size > 1) {
//         int root = rank / size * size;
//         int relRank = rank % size;
//         int nextSize = size / 2;

//         int sendRank  = root + (relRank + nextSize) % size;
//         int recvRank = sendRank;

//         ncclGroupStart();

//         for (int i = 0; i < numTransfers; i++) {
//             // ncclGroupStart();
//             // printf("rank %d, sendRank %d, recvRank %d, i %d\n", rank, sendRank, recvRank, i);
//             int sendOffset = ((int*) cpuIndexBuffer)[i] * count * datatypeSize;
//             int recvOffset = ((int*) cpuIndexBuffer)[numTransfers + i] * count * datatypeSize;
//             // printf("COMPUTED OFFSET ON RANK Q %d: %d, %d\n", rank, sendOffset, recvOffset);
//             ncclSend((char*) recvbuff + sendOffset, nElements, datatype, sendRank, comm, stream);
//             ncclRecv((char*) recvbuff + recvOffset, nElements, datatype, recvRank, comm, stream);
//             // ncclGroupEnd();
//         }

//         ncclGroupEnd();
        
//         offset += nElements * datatypeSize;
//         size /= 2;
//         numTransfers *= 2;
//     }
    
//     return ncclSuccess;
// }


// inline ncclResult_t allgather_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count, 
//     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
// {
//     int rank, nRanks;
//     ncclCommUserRank(comm, &rank);
//     ncclCommCount(comm, &nRanks);

//     // Calculate the size of the data type
//     size_t datatypeSize;
//     ncclDataTypeSize(datatype, &datatypeSize);

//     launchComputePermutationKernel(indexBuffer, 0, nRanks, 2, stream);
//     cudaMemcpy(cpuIndexBuffer, indexBuffer, nRanks * sizeof(int), cudaMemcpyDeviceToHost);
//     int exchPeer = ((int*) cpuIndexBuffer)[rank];
    
//     const int sendOffset = exchPeer;
//     int recvOffset = exchPeer;

//     ncclGroupStart();

//     // Initial swap between rank and exchPeer
//     if (exchPeer == rank) {
//         if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
//             // Copy data from sendbuff to recvbuff using cudaMemcpy 
//             cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
//                 sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
//         }
//     } else {
//         if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
//             ncclSend(sendbuff, count, datatype, exchPeer, comm, stream);
//             cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
//                 sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
//         } else{
//             ncclSend((char*) recvbuff + rank * count * datatypeSize, count, datatype, exchPeer, comm, stream);
//         }
//         ncclRecv((char*) recvbuff + recvOffset * count * datatypeSize, count, datatype, exchPeer, comm, stream);
//     }

//     ncclGroupEnd();


//     int size = nRanks;
//     int numTransfers = 1;
//     while (size > 1) {
//         int root = rank / size * size;
//         int relRank = rank % size;
//         int nextSize = size / 2;

//         int sendRank  = root + (relRank + nextSize) % size;
//         int recvRank = sendRank;

//         int src_root = ((int*) cpuIndexBuffer)[rank] / numTransfers * numTransfers;
//         int dst_root = ((int*) cpuIndexBuffer)[recvRank] / numTransfers * numTransfers;

//         ncclGroupStart();

//         ncclSend((char*) recvbuff + src_root * count * datatypeSize, numTransfers * count, datatype, sendRank, comm, stream);
//         ncclRecv((char*) recvbuff + dst_root * count * datatypeSize, numTransfers * count, datatype, recvRank, comm, stream);


//         ncclGroupEnd();

//         size /= 2;
//         numTransfers *= 2;
//     }
    
//     return ncclSuccess;
// }

// inline ncclResult_t allgather_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count, 
//     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) 
// {
//     int rank, nRanks;
//     ncclCommUserRank(comm, &rank);
//     ncclCommCount(comm, &nRanks);

//     // Calculate the size of the data type
//     size_t datatypeSize;
//     ncclDataTypeSize(datatype, &datatypeSize);

//     launchComputePermutationKernel(indexBuffer, 0, nRanks, 2, stream);
//     cudaMemcpy(cpuIndexBuffer, indexBuffer, nRanks * sizeof(int), cudaMemcpyDeviceToHost);
//     int exchPeer = ((int*) cpuIndexBuffer)[rank];
    
//     const int sendOffset = exchPeer;
//     int recvOffset = exchPeer;

//     ncclGroupStart();

//     // Initial swap between rank and exchPeer
//     if (exchPeer == rank) {
//         if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
//             // Copy data from sendbuff to recvbuff using cudaMemcpy 
//             cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
//                 sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
//         }
//     } else {
//         if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
//             ncclSend(sendbuff, count, datatype, exchPeer, comm, stream);
//             cudaMemcpy((char*) recvbuff + rank * count * datatypeSize, 
//                 sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice);
//         } else{
//             ncclSend((char*) recvbuff + rank * count * datatypeSize, count, datatype, exchPeer, comm, stream);
//         }
//         ncclRecv((char*) recvbuff + recvOffset * count * datatypeSize, count, datatype, exchPeer, comm, stream);
//     }

//     ncclGroupEnd();


//     int size = nRanks;
//     int numTransfers = 1;
//     while (size > 1) {
//         int root = rank / size * size;
//         int relRank = rank % size;
//         int nextSize = size / 2;

//         int sendRank  = root + (relRank + nextSize) % size;
//         int recvRank = sendRank;

//         int src_root = ((int*) cpuIndexBuffer)[rank] / numTransfers * numTransfers;
//         int dst_root = ((int*) cpuIndexBuffer)[recvRank] / numTransfers * numTransfers;

//         ncclGroupStart();

//         if (sendRank >= src_root && sendRank < src_root + numTransfers) {
//             int leftTransfers = sendRank - src_root;
//             int rightTransfers = numTransfers - leftTransfers - 1;
//             if (leftTransfers > 0) {
//                 ncclSend((char*) recvbuff + src_root * count * datatypeSize, leftTransfers * count, datatype, sendRank, comm, stream);
//             }

//             if (rightTransfers > 0) {
//                 ncclSend((char*) recvbuff + (src_root + leftTransfers + 1) * count * datatypeSize, rightTransfers * count, datatype, sendRank, comm, stream);
//             }
            
//         } else {
//             ncclSend((char*) recvbuff + src_root * count * datatypeSize, numTransfers * count, datatype, sendRank, comm, stream);
//         }

//         if (rank >= dst_root && rank < dst_root + numTransfers) {
//             int leftTransfers = rank - dst_root;
//             int rightTransfers = numTransfers - leftTransfers - 1;

//             if (leftTransfers > 0) {
//                 ncclRecv((char*) recvbuff + dst_root * count * datatypeSize, leftTransfers * count, datatype, recvRank, comm, stream);
//             }

//             if (rightTransfers > 0) {
//                 ncclRecv((char*) recvbuff + (dst_root + leftTransfers + 1) * count * datatypeSize, rightTransfers * count, datatype, recvRank, comm, stream);
//             }
//         } else {
//             ncclRecv((char*) recvbuff + dst_root * count * datatypeSize, numTransfers * count, datatype, recvRank, comm, stream);
//         }


//         ncclGroupEnd();

//         size /= 2;
//         numTransfers *= 2;
//     }
    
//     return ncclSuccess;
// }

inline ncclResult_t allgather_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, int k) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);


    launchComputePermutationKernel(indexBuffer, rank, nRanks, k, stream);

    // Check for in-place operation
    if (!((char*) sendbuff == (char*) recvbuff + rank * count * datatypeSize)) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy. We don't need to
        // worry about indices right now since we are permuting the result at the end
        cudaMemcpyAsync(recvbuff, sendbuff, count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpyAsync(recvbuff, (char*) recvbuff + rank * count * datatypeSize, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    size_t offset = count * datatypeSize;
    size_t nElements = count;
    int size = nRanks;
    while (size > 1) {
        int root = rank / size * size;
        int relRank = rank % size;
        int nextSize = size / k;

        ncclGroupStart();
        for (int i = 1; i < k; i += 1) {
            int exchangeRank = root + (relRank + i * nextSize) % size;
            ncclSend(recvbuff, nElements, datatype, exchangeRank, comm, stream);
            ncclRecv((char*) recvbuff + i * offset, nElements, 
                datatype, exchangeRank, comm, stream);
        }
        ncclGroupEnd();
        
        nElements *= k;
        offset = nElements * datatypeSize;
        size /= k;
    }

    
    picclPermuteInplace(recvbuff, indexBuffer, (size_t) (nRanks * count), count, 
        datatype, stream);
    
    return ncclSuccess;
}