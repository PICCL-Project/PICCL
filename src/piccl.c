/*
* piccl.c
* ---------------------
* LD_PRELOAD shim which now re‑implements *some* NCCL collectives using the
* **point‑to‑point** primitives `ncclSend` / `ncclRecv`.
*
* Build:
*   make                     # see accompanying Makefile
*
* Run:
*   LD_PRELOAD=./piccl.so <app>
*/

#define _GNU_SOURCE 1

#include <dlfcn.h>

#include <piccl.h>
#include <allgather.h>
#include <allreduce.h>
#include <reduce.h>
#include <reduce_scatter.h>
#include <broadcast.h>

/* ======================================================================== */
/*  NCCL Collectives Wrapper                                                */
/* ======================================================================== */

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream){
   nvtxRangePushA("ncclAllGather");
   ncclResult_t res = ncclSuccess;
   switch(ALL_GATHER_ALGO) {
      case RING:
         res = allgather_ring(sendbuff, recvbuff, count, datatype, comm, stream);
         break;
      case K_RING:
         res = allgather_k_ring(sendbuff, recvbuff, count, datatype, comm, stream, ALL_GATHER_K);
         break;
      case RECURSIVE_DOUBLING:
         res = allgather_recursive_doubling(sendbuff, recvbuff, count, datatype, comm, stream);
         break;
      case RECURSIVE_MULTIPLY:
         res = allgather_recursive_multiplying(sendbuff, recvbuff, count, datatype, comm, stream, ALL_GATHER_K);
         break;
      case PERMUTED_RECURSIVE_DOUBLING:
         res = allgather_permuted_recursive_doubling(sendbuff, recvbuff, count, datatype, comm, stream);
         break;
      case PERMUTED_RECURSIVE_MULTIPLY:
         res = allgather_permuted_recursive_multiplying(sendbuff, recvbuff, count, datatype, comm, stream, ALL_GATHER_K);
         break;
      case DEFAULT_NCCL:
      default:
         nvtxRangePop();
         return real_ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
         break;
   }

   nvtxRangePop();
   return res;
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream){
   ncclResult_t res = ncclSuccess;
   switch(ALL_REDUCE_ALGO) {
      case RING:
         res = allreduce_ring(sendbuff, recvbuff, count, datatype, op, comm, stream);
         break;
      case K_RING:
         res = allreduce_k_ring(sendbuff, recvbuff, count, datatype, op, comm, stream, ALL_REDUCE_K);
         break;
      case RECURSIVE_DOUBLING:
         res = allreduce_recursive_doubling(sendbuff, recvbuff, count, datatype, op, comm, stream);
         break;
      case RECURSIVE_MULTIPLY:
         res = allreduce_recursive_multiplying(sendbuff, recvbuff, count, datatype, op, comm, stream, ALL_REDUCE_K);
         break;
      case PERMUTED_RECURSIVE_DOUBLING:
         res = allreduce_permuted_recursive_doubling(sendbuff, recvbuff, count, datatype, op, comm, stream);
         break;
      case PERMUTED_RECURSIVE_MULTIPLY:
         res = allreduce_permuted_recursive_multiplying(sendbuff, recvbuff, count, datatype, op, comm, stream, ALL_REDUCE_K);
         break;
      case DEFAULT_NCCL:
      default:
         return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
         break;
   }

   return res;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream){
   ncclResult_t res = ncclSuccess;
   switch(REDUCE_ALGO) {
      case RING:
         res = reduce_ring(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
         break;
      case K_RING:
         res = reduce_k_ring(sendbuff, recvbuff, count, datatype, op, root, comm, stream, REDUCE_K);
         break;
      case PERMUTED_RECURSIVE_DOUBLING:
         res = reduce_permuted_recursive_doubling(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
         break;
      case PERMUTED_RECURSIVE_MULTIPLY:
         res = reduce_permuted_recursive_multiplying(sendbuff, recvbuff, count, datatype, op, root, comm, stream, REDUCE_K);
         break;
      case DEFAULT_NCCL:
      default:
         return real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
         break;
   }

   return res;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                              ncclDataType_t datatype, ncclRedOp_t op,
                              ncclComm_t comm, cudaStream_t stream){
   ncclResult_t res = ncclSuccess;

   switch(REDUCE_SCATTER_ALGO) {
      case RING:
         res = reduce_scatter_ring(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
         break;
      case K_RING:
         res = reduce_scatter_k_ring(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, REDUCE_SCATTER_K);
         break;
      case PERMUTED_RECURSIVE_DOUBLING:
         res = reduce_scatter_permuted_recursive_doubling(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
         break;
      case PERMUTED_RECURSIVE_MULTIPLY:
         res = reduce_scatter_permuted_recursive_multiplying(sendbuff, recvbuff, recvcount, datatype, op, comm, stream, REDUCE_SCATTER_K);
         break;
      case DEFAULT_NCCL:
      default:
         return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
         break;
   }

   return res;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
   ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream){
   ncclResult_t res = ncclSuccess;
   
   switch(BROADCAST_ALGO) {
      case RING:
         res = broadcast_ring(sendbuff, recvbuff, count, datatype, root, comm, stream);
         break;
      case K_RING:
         res = broadcast_k_ring(sendbuff, recvbuff, count, datatype, root, comm, stream, BROADCAST_K);
         break;
      case PERMUTED_RECURSIVE_DOUBLING:
         res = broadcast_permuted_recursive_doubling(sendbuff, recvbuff, count, datatype, root, comm, stream);
         break;
      case PERMUTED_RECURSIVE_MULTIPLY:
         res = broadcast_permuted_recursive_multiplying(sendbuff, recvbuff, count, datatype, root, comm, stream, BROADCAST_K);
         break;
      case DEFAULT_NCCL:
      default:
         return real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
         break;
   }

   return res;
}

void* reductionBuffer = NULL;
void* permutationBuffer = NULL;
void* indexBuffer = NULL;
void* cpuIndexBuffer = NULL;


ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
   cudaMalloc(&reductionBuffer, DATA_BUFFER_SIZE);
   cudaMalloc(&permutationBuffer, DATA_BUFFER_SIZE);
   cudaMalloc(&indexBuffer, INDEX_BUFFER_SIZE);

   cpuIndexBuffer = malloc(INDEX_BUFFER_SIZE);
   
   return real_ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
   cudaFree(reductionBuffer);
   cudaFree(permutationBuffer);
   cudaFree(indexBuffer);

   free(cpuIndexBuffer);

   return real_ncclCommDestroy(comm);
}

/* --- Banner ------------------------------------------------------------- */
__attribute__((constructor)) static void piccl_banner(void){ fprintf(stderr,"[PICCL] loaded (PID %d).\n",(int)getpid()); }


/* --- Symbol resolution ----------------------------------------- */
__attribute__((constructor)) static void piccl_real_nccl(void){
   void* handle = dlopen("libnccl.so", RTLD_LAZY | RTLD_GLOBAL);
   if (!handle) {
       fprintf(stderr, "[PICCL] Failed to load libnccl.so: %s\n", dlerror());
       exit(EXIT_FAILURE);
   }

   RESOLVE_REAL(ncclAllGather);
   RESOLVE_REAL(ncclBroadcast);
   RESOLVE_REAL(ncclAllReduce);
   RESOLVE_REAL(ncclReduce);
   RESOLVE_REAL(ncclReduceScatter);
   RESOLVE_REAL(ncclCommInitRank);
   RESOLVE_REAL(ncclCommDestroy);

   fprintf(stderr, "[PICCL] Loaded real NCCL functions.\n");
}

/* --- Environment --------------------------------------------------------*/
piccl_algo_t ALL_GATHER_ALGO;
piccl_algo_t ALL_REDUCE_ALGO;
piccl_algo_t REDUCE_ALGO;
piccl_algo_t REDUCE_SCATTER_ALGO;
piccl_algo_t BROADCAST_ALGO;

int ALL_GATHER_K;
int ALL_REDUCE_K;
int REDUCE_K;
int REDUCE_SCATTER_K;
int BROADCAST_K;

unsigned long globalBlockSize;
unsigned long globalMaxGridSize;
unsigned long globalMinGridSize;

char PICCL_ALGO_ENV[N_COLLS][256] = {
   "PICCL_ALLGATHER_ALGO",
   "PICCL_ALLREDUCE_ALGO",
   "PICCL_REDUCE_ALGO",
   "PICCL_REDUCE_SCATTER_ALGO",
   "PICCL_BROADCAST_ALGO"
};

char PICCL_ALGO_ENV_K[N_COLLS][256] = {
   "PICCL_ALLGATHER_K",
   "PICCL_ALLREDUCE_K",
   "PICCL_REDUCE_K",
   "PICCL_REDUCE_SCATTER_K",
   "PICCL_BROADCAST_K"
};

__attribute__((constructor)) static void piccl_algo_env(void){
   fprintf(stderr, "[PICCL] Environment algo variables:\n");
   for (int i=0; i < N_COLLS; i++) {
      
      char *env = getenv(PICCL_ALGO_ENV[i]);
      piccl_algo_t algo = DEFAULT_NCCL;
      
      if(env) {
         if(strcmp(env,"RING") == 0) algo = RING;
         else if(strcmp(env,"K_RING") == 0) algo = K_RING;
         else if(strcmp(env,"RD") == 0) algo = RECURSIVE_DOUBLING;
         else if(strcmp(env,"RM") == 0) algo = RECURSIVE_MULTIPLY;
         else if(strcmp(env,"PRD") == 0) algo = PERMUTED_RECURSIVE_DOUBLING;
         else if(strcmp(env,"PRM") == 0) algo = PERMUTED_RECURSIVE_MULTIPLY;
         else algo = DEFAULT_NCCL;
      }

      switch(i) {
         case 0: ALL_GATHER_ALGO = algo; break;
         case 1: ALL_REDUCE_ALGO = algo; break;
         case 2: REDUCE_ALGO = algo; break;
         case 3: REDUCE_SCATTER_ALGO = algo; break;
         case 4: BROADCAST_ALGO = algo; break;
      }

      fprintf(stderr, " %s=%s ", PICCL_ALGO_ENV[i], env ? env : "default");
   }
   fprintf(stderr, "\n");
}

__attribute__((constructor)) static void piccl_param_env(void){
   fprintf(stderr, "[PICCL] Environment param variables:\n");
   for (int i=0; i < N_COLLS; i++) {
      
      char *env = getenv(PICCL_ALGO_ENV_K[i]);
      int param = -1;
      if (env) {
         param = atoi(env);
      }

      switch(i) {
         case 0: ALL_GATHER_K = param; break;
         case 1: ALL_REDUCE_K = param; break;
         case 2: REDUCE_K = param; break;
         case 3: REDUCE_SCATTER_K = param; break;
         case 4: BROADCAST_K = param; break;
      }

      fprintf(stderr, " %s=%s ", PICCL_ALGO_ENV_K[i], env ? env : "-1");
   }
   fprintf(stderr, "\n");
}

__attribute__((constructor)) static void piccl_kernel_param_env(void){
   char *env = getenv("PICCL_BLOCK_SIZE");
   if (env) {
      globalBlockSize = atoi(env);
   } else {
      globalBlockSize = 1024UL;
   }
   fprintf(stderr, "[PICCL] Environment kernel block size: %lu\n", globalBlockSize);

   env = getenv("PICCL_MAX_GRID_SIZE");
   if (env) {
      globalMaxGridSize = atoi(env);
   } else {
      globalMaxGridSize = 1024UL;
   }
   fprintf(stderr, "[PICCL] Environment kernel max grid size: %lu\n", globalMaxGridSize);

   env = getenv("PICCL_MIN_GRID_SIZE");
   if (env) {
      globalMinGridSize = atoi(env);
   } else {
      globalMinGridSize = 1UL;
   }
   fprintf(stderr, "[PICCL] Environment kernel min grid size: %lu\n", globalMinGridSize);
}