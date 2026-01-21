#include <math.h>

#include <allreduce.h>
#include <reduce_kernel.h>

// Macro to compute mod nRanks using an if statement
#define MOD_NRANKS(x) ((x) >= (nRanks) ? (x) - (nRanks) : (x))

static inline ncclResult_t internal_allgather_ring(void* recvbuff, ncclDataType_t datatype, const int* recvcounts, 
    const int* displs, ncclComm_t comm, cudaStream_t stream) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    int total_count = 0;
    for (int i = 0; i < nRanks; i++)
        total_count += recvcounts[i];

    // ASSUMING IN-PLACE

    int left = (nRanks + rank - 1) % nRanks;
    int right = (rank + 1) % nRanks;

    int torecv = total_count - recvcounts[rank];
    int tosend = total_count - recvcounts[right];

    int max = recvcounts[0];
    for (int i = 1; i < nRanks; i++)
        if (max < recvcounts[i])
            max = recvcounts[i];
    
    int chunk_count = max;
    int soffset, roffset;
    int sidx, ridx;
    sidx = rank;
    ridx = left;
    soffset = 0;
    roffset = 0;

    while (tosend || torecv) {  /* While we have data to send or receive */
        int sendnow, recvnow;
        sendnow = ((recvcounts[sidx] - soffset) >
                   chunk_count) ? chunk_count : (recvcounts[sidx] - soffset);
        recvnow = ((recvcounts[ridx] - roffset) >
                   chunk_count) ? chunk_count : (recvcounts[ridx] - roffset);

        char *sbuf, *rbuf;
        sbuf = (char *) recvbuff + ((displs[sidx] + soffset) * datatypeSize);
        rbuf = (char *) recvbuff + ((displs[ridx] + roffset) * datatypeSize);

        /* Protect against wrap-around of indices */
        if (!tosend)
            sendnow = 0;
        if (!torecv)
            recvnow = 0;

        if (!sendnow && !recvnow) {
            /* Don't do anything. This case is possible if two
                * consecutive processes contribute 0 bytes each. */
        } else if (!sendnow) {  /* If there's no data to send, just do a recv call */
            ncclRecv(rbuf, recvnow, datatype, left, comm, stream);
            torecv -= recvnow;
        } else if (!recvnow) {  /* If there's no data to receive, just do a send call */
            ncclSend(sbuf, sendnow, datatype, right, comm, stream);            
            tosend -= sendnow;
        } else {        /* There's data to be sent and received */

            ncclGroupStart();
            ncclSend(sbuf, sendnow, datatype, right, comm, stream);
            ncclRecv(rbuf, recvnow, datatype, left, comm, stream);
            ncclGroupEnd();
           
            tosend -= sendnow;
            torecv -= recvnow;
        }

        soffset += sendnow;
        roffset += recvnow;
        if (soffset == recvcounts[sidx]) {
            soffset = 0;
            sidx = (sidx + nRanks - 1) % nRanks;
        }
        if (roffset == recvcounts[ridx]) {
            roffset = 0;
            ridx = (ridx + nRanks - 1) % nRanks;
        }

    }

    return ncclSuccess;
}

inline ncclResult_t allreduce_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) 
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    void* tmpbuff = reductionBuffer;

    // Set up index buffers
    int cnts[nRanks];
    int displs[nRanks];

    for (int i = 0; i < nRanks; i++)
        cnts[i] = 0;

    int total_count = 0;
    for (int i = 0; i < nRanks; i++) {
        cnts[i] = (count + nRanks - 1) / nRanks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (int i = 1; i < nRanks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    // Check for in-place operation
    if (sendbuff != recvbuff) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpyAsync(recvbuff, sendbuff, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    int src = (rank - 1 + nRanks) % nRanks;
    int dst = (rank + 1) % nRanks;

    // Reduce-Scatter
    for (int i = 0; i < nRanks - 1; ++i) {
        int recv_rank = (nRanks + rank - 2 - i) % nRanks;
        int send_rank = (nRanks + rank - 1 - i) % nRanks;

        ncclGroupStart();
        ncclSend((char *) recvbuff + displs[send_rank] * datatypeSize, cnts[send_rank], datatype, dst, comm, stream);
        ncclRecv(tmpbuff, cnts[recv_rank], datatype, src, comm, stream);
        ncclGroupEnd();

        picclReduce((char *) recvbuff + displs[recv_rank] * datatypeSize, (char *) recvbuff + displs[recv_rank] * datatypeSize, tmpbuff, cnts[recv_rank], 1, datatype, op, stream);
    }

    // AllGather
    return internal_allgather_ring(recvbuff, datatype, cnts, displs, comm, stream);
}

inline ncclResult_t allreduce_k_ring(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int k)
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);


    void* tmpbuff = reductionBuffer;
    
    // Set up index buffers
    int cnts[nRanks];
    int displs[nRanks];

    for (int i = 0; i < nRanks; i++)
        cnts[i] = 0;

    int total_count = 0;
    for (int i = 0; i < nRanks; i++) {
        cnts[i] = (count + nRanks - 1) / nRanks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else
            total_count += cnts[i];
    }

    displs[0] = 0;
    for (int i = 1; i < nRanks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    /* Defaulting to k=n where n is the communicator size if n is not a multiple of k */
    if (nRanks != nRanks / k * k)
        k = nRanks;

    if (nRanks < k)
        k = nRanks;

    /* Get relative group information */
    int group_num = nRanks / k;          /* Number of groups total to contain all processes */
    int group = rank / k;                   /* Group number in which current process is located */
    int group_rank = rank % k;              /* Relative rank of process within a group */
    int group_start = ((group_num + group - 1) % group_num) * k; /* Start node of first intra-group communication */

    int group_rank_m1 = (k + group_rank - 1) % k;
    int group_rank_m2 = (k + group_rank - 2) % k;
    int group_rank_p1 = (group_rank + 1) % k;

    int intra_left = group_rank_m1 + group * k;  /* Process to receive from for intra-group communication */
    int intra_right = group_rank_p1 + group * k;     /* Process to send to for intra-group communication */
    int inter_left = group_rank_m1 + group_start; /* Process to receive from for inter-group communication */
    int inter_right = group_rank_p1 + ((group + 1) % group_num) * k; /* Process to send to for inter-group communicatin */

    // Check for in-place operation
    if (sendbuff != recvbuff) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpyAsync(recvbuff, sendbuff, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    // printf("rank %d, group %d, intra_left %d, intra_right %d, inter_left %d, inter_right %d\n", rank, group, intra_left, intra_right, inter_left, inter_right);

    int intra_send = group_rank_m1 + group_start;
    int intra_recv = group_rank_m2 + group_start;
    for (int i = 1; i < k; i++) {
        ncclGroupStart();
        ncclSend(((char *) recvbuff + displs[intra_send] * datatypeSize), 
            cnts[intra_send], datatype, intra_right, comm, stream);
        ncclRecv(tmpbuff, cnts[intra_recv], datatype, intra_left, comm, stream);
        ncclGroupEnd();

        /* Perform commutitve reduce operation */
        picclReduce((char *) recvbuff + displs[intra_recv] * datatypeSize, 
            (char *) recvbuff + displs[intra_recv] * datatypeSize, tmpbuff, cnts[intra_recv], 1, datatype, op, stream);

        intra_send = intra_recv;
        intra_recv = (k + intra_recv - 1) % k + group_start;
    }

    for (int i = 1; i < group_num; i++) {
        int next_group_rank = (nRanks + group_start - k) % nRanks;
        int inter_send = group_start + group_rank;
        int inter_recv = group_rank_m1 + next_group_rank;

        /* Inter-group communication */
        ncclGroupStart();
        ncclSend(((char *) recvbuff + displs[inter_send] * datatypeSize),
            cnts[inter_send], datatype, inter_right, comm, stream);
        ncclRecv(tmpbuff, cnts[inter_recv], datatype, inter_left, comm, stream);
        ncclGroupEnd();

        /* Perform commutitve reduce operation */
        picclReduce((char *) recvbuff + displs[inter_recv] * datatypeSize,
            (char *) recvbuff + displs[inter_recv] * datatypeSize, tmpbuff, cnts[inter_recv], 1, datatype, op, stream);
        
        group_start = next_group_rank;

        /* Intra-group communication */
        intra_send = group_rank_m1 + group_start;
        intra_recv = group_rank_m2 + group_start;
        for (int j = 1; j < k; j++) {
            ncclGroupStart();
            ncclSend(((char *) recvbuff + displs[intra_send] * datatypeSize),
                cnts[intra_send], datatype, intra_right, comm, stream);
            ncclRecv(tmpbuff, cnts[intra_recv], datatype, intra_left, comm, stream);
            ncclGroupEnd();
            
            picclReduce((char *) recvbuff + displs[intra_recv] * datatypeSize,
                (char *) recvbuff + displs[intra_recv] * datatypeSize, tmpbuff, cnts[intra_recv], 1, datatype, op, stream);
            
            intra_send = intra_recv;
            intra_recv = (k + intra_recv - 1) % k + group_start;
        }

        inter_send = inter_recv;
        inter_recv = (nRanks + inter_recv - k) % nRanks;
    }

    // AllGather
    return internal_allgather_ring(recvbuff, datatype, cnts, displs, comm, stream);
}

inline ncclResult_t allreduce_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (sendbuff != recvbuff) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpyAsync(recvbuff, sendbuff, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    // get nearest power-of-two less than or equal to comm_size
    int p = (int) (log(nRanks) / log(2));
    int pof2 = (int) pow(2, p);
    int rem = nRanks - pof2;

    void* tmpbuff = reductionBuffer;

    // Initial reduction for non power-of-two ranks
    if (rank >= (nRanks - 2 * rem)) {
        if (rank >= pof2) { // Send
            int dst = rank - rem;
            ncclSend(recvbuff, count, datatype, dst, comm, stream);
        } else { // Recv
            int dst = rank + rem;
            ncclRecv(tmpbuff, count, datatype, dst, comm, stream);
            picclReduce(recvbuff, recvbuff, tmpbuff, count, 1, datatype, op, stream);
        }
    }

    // Recursive doubling
    if (rank < pof2) {
        int mask = 0x1;
        while (mask < pof2) {
            int dst = rank ^ mask;

            ncclGroupStart();
            ncclSend(recvbuff, count, datatype, dst, comm, stream);
            ncclRecv(tmpbuff, count, datatype, dst, comm, stream);
            ncclGroupEnd();

            picclReduce(recvbuff, recvbuff, tmpbuff, count, 1, datatype, op, stream);            
    
            mask <<= 1;
        }
    }

    // Final exchange for non power-of-two ranks
    if (rank >= (nRanks - 2 * rem)) {
        if (rank >= pof2) { // Send
            int dst = rank - rem;
            ncclRecv(recvbuff, count, datatype, dst, comm, stream);
        } else { // Recv
            int dst = rank + rem;
            ncclSend(recvbuff, count, datatype, dst, comm, stream);
        }
    }

    return ncclSuccess;
}

inline ncclResult_t allreduce_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int k)
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (sendbuff != recvbuff) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpyAsync(recvbuff, sendbuff, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    // get nearest power-of-k less than or equal to comm_size
    int p = (int) (log(nRanks) / log(k));
    int pofk = (int) pow(k, p);

    void* tmpbuff = reductionBuffer;
    void* persistbuff = permutationBuffer;
    void* truebuff = recvbuff;

    // Initial exchange for non power-of-k ranks
    if (rank >= pofk) {
        // We are follower
        int leader = rank % pofk;
        ncclSend(recvbuff, count, datatype, leader, comm, stream);
    } else {
        // We are maybe the leader
        for (int i = rank + pofk; i < nRanks; i += pofk) {
            ncclRecv(tmpbuff, count, datatype, i, comm, stream);
            picclReduce(recvbuff, recvbuff, tmpbuff, count, 1, datatype, op, stream);
        }
    }

    if (rank < pofk) {
        int radixSize = 1;
        int nextRadixSize = k;

        while (radixSize < pofk) {
            int nextRadixRoot = (rank / nextRadixSize) * nextRadixSize;
            int radixLoc = (int) rank % radixSize;

            int dstRoot = nextRadixRoot;
            int dstRootLimit = nextRadixRoot + nextRadixSize;

            int iter = 0;
            ncclGroupStart(); 
            while (dstRoot < dstRootLimit && dstRoot < pofk) {
                int dst = dstRoot + radixLoc;

                if (dst != rank && dst < pofk) {
                    ncclSend(recvbuff, count, datatype, dst, comm, stream);
                    ncclRecv((char* ) tmpbuff + iter * count * datatypeSize, count, datatype, dst, comm, stream);
                    iter++;
                }
                
                dstRoot += radixSize;
            }
            ncclGroupEnd();
            
            picclReduce(persistbuff, recvbuff, tmpbuff, count, k-1, datatype, op, stream);

            // Swap buffers
            void* tmp = recvbuff;
            recvbuff = persistbuff;
            persistbuff = tmp;


            radixSize = nextRadixSize;
            nextRadixSize *= k;
        }
    }


    // Final exchange for non power-of-k ranks
    if (rank >= pofk) {
        // We are follower
        int leader = rank % pofk;
        ncclRecv(recvbuff, count, datatype, leader, comm, stream);
    } else {
        // We are maybe the leader
        ncclGroupStart();
        for (int i = rank + pofk; i < nRanks; i += pofk) {
            ncclSend(recvbuff, count, datatype, i, comm, stream);
        }
        ncclGroupEnd();
    }

    if (truebuff != recvbuff) {
        cudaMemcpyAsync(truebuff, recvbuff, count * datatypeSize, 
            cudaMemcpyDeviceToDevice, stream);
    }

    return ncclSuccess;
}

inline ncclResult_t allreduce_permuted_recursive_doubling(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (sendbuff != recvbuff) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpyAsync(recvbuff, sendbuff, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    // get nearest power-of-two less than or equal to comm_size
    int p = (int) (log(nRanks) / log(2));
    int pof2 = (int) pow(2, p);
    int rem = nRanks - pof2;

    void* tmpbuff = reductionBuffer;

    // Initial reduction for non power-of-two ranks
    if (rank >= (nRanks - 2 * rem)) {
        if (rank >= pof2) { // Send
            int dst = rank - rem;
            ncclSend(recvbuff, count, datatype, dst, comm, stream);
        } else { // Recv
            int dst = rank + rem;
            ncclRecv(tmpbuff, count, datatype, dst, comm, stream);
            picclReduce(recvbuff, recvbuff, tmpbuff, count, 1, datatype, op, stream);
        }
    }

    // Recursive doubling (permuted)
    if (rank < pof2) {
        int size = pof2;

        while (size > 1) {
            int root = rank / size * size;
            int relRank = rank % size;
            int nextSize = size / 2;
    
            int dst  = root + (relRank + nextSize) % size;
    
            ncclGroupStart();
            ncclSend(recvbuff, count, datatype, dst, comm, stream);
            ncclRecv(tmpbuff, count, datatype, dst, comm, stream);
            ncclGroupEnd();

            picclReduce(recvbuff, recvbuff, tmpbuff, count, 1, datatype, op, stream);     
            
            size /= 2;
        }
    }

    // Final exchange for non power-of-two ranks
    if (rank >= (nRanks - 2 * rem)) {
        if (rank >= pof2) { // Send
            int dst = rank - rem;
            ncclRecv(recvbuff, count, datatype, dst, comm, stream);
        } else { // Recv
            int dst = rank + rem;
            ncclSend(recvbuff, count, datatype, dst, comm, stream);
        }
    }

    return ncclSuccess;
}

inline ncclResult_t allreduce_permuted_recursive_multiplying(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int k)
{
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    // Calculate the size of the data type
    size_t datatypeSize;
    ncclDataTypeSize(datatype, &datatypeSize);

    // Check for in-place operation
    if (sendbuff != recvbuff) {
        // Copy data from sendbuff to recvbuff using cudaMemcpy 
        cudaMemcpyAsync(recvbuff, sendbuff, 
            count * datatypeSize, cudaMemcpyDeviceToDevice, stream);
    }

    // get nearest power-of-k less than or equal to comm_size
    int p = (int) (log(nRanks) / log(k));
    int pofk = (int) pow(k, p);

    void* tmpbuff = reductionBuffer;
    void* persistbuff = permutationBuffer;
    void* truebuff = recvbuff;

    // Initial exchange for non power-of-k ranks
    if (rank >= pofk) {
        // We are follower
        int leader = rank % pofk;
        ncclSend(recvbuff, count, datatype, leader, comm, stream);
    } else {
        // We are maybe the leader
        for (int i = rank + pofk; i < nRanks; i += pofk) {
            ncclRecv(tmpbuff, count, datatype, i, comm, stream);
            picclReduce(recvbuff, recvbuff, tmpbuff, count, 1, datatype, op, stream);
        }
    }

    if (rank < pofk) {
        int size = pofk;
        while (size > 1) {
            int root = rank / size * size;
            int nextSize = size / k;
            int relRank = rank % nextSize;

            int iter = 0;
            ncclGroupStart();             
            for (int i = 0; i < k; i += 1) {
                int dst = root + i * nextSize + relRank;

                if (dst != rank && dst < pofk) {
                    ncclSend(recvbuff, count, datatype, dst, comm, stream);
                    ncclRecv((char* ) tmpbuff + iter * count * datatypeSize, count, datatype, dst, comm, stream);
                    iter++;
                }
            }
            ncclGroupEnd();
            
            picclReduce(persistbuff, recvbuff, tmpbuff, count, k-1, datatype, op, stream);

            // Swap buffers
            void* tmp = recvbuff;
            recvbuff = persistbuff;
            persistbuff = tmp;

            size /= k;
        }    
    }

    // Final exchange for non power-of-k ranks
    if (rank >= pofk) {
        // We are follower
        int leader = rank % pofk;
        ncclRecv(recvbuff, count, datatype, leader, comm, stream);
    } else {
        // We are maybe the leader
        ncclGroupStart();
        for (int i = rank + pofk; i < nRanks; i += pofk) {
            ncclSend(recvbuff, count, datatype, i, comm, stream);
        }
        ncclGroupEnd();
    }

    if (truebuff != recvbuff) {
        cudaMemcpyAsync(truebuff, recvbuff, count * datatypeSize, 
            cudaMemcpyDeviceToDevice, stream);
    }

    return ncclSuccess;
}
