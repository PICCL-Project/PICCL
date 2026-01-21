# PICCL
PICCL is a LD_PRELOAD collectives library built on top of NCCL.

## Defining the Environment
To build and run PICCL, you will first need an environemnt that has openmpi and cuda. We recommend that you create a file that you can `source` in the `envs` directory. An example of a source file is shown below. 

```
# src_env_delta.sh
module load openmpi+cuda 
module load libfabric/1.15.2.0
module load nccl/2.19.3-1.awsplugin

export NCCL_HOME=<nccl_dir>/build
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export NCCL_CROSS_NIC=1
export MPI_HOME=<mpi_dir>
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$GCC_HOME/lib64:$LD_LIBRARY_PATH
```

## Install Dependencies
The two main dependencies we need are NCCL and OSU benchmarks. There is a handy script in the project root directory `download_libs.sh` to help download and compile them for you. 

Be sure to set the variable `SRC_FILE` and `ARCH` to the correct source file and GPU architecture respectively in the script file. 

```
$ cd $HOME_DIR
$ ./download_libs.sh
```
This will create a new `libs/` directory with the two dependencies. 


## Building PICCL
To build PICCL, in the project root directory, simply run 
```
make
```
in the project root directory. This will produce a file called `piccl.so` in the `src` folder. 

## Running PICCL
To use PICCL, run
```
LD_PRELOAD=<path-to-piccl.so> <srun/mpirun/mpiexec> <application>
```

To use different algorithms, use the following environment variables
```
    export PICCL_ALLGATHER_ALGO=<RING/K_RING/PRD/PRM>
    export PICCL_ALLREDUCE_ALGO=<RING/K_RING/PRD/PRM>
    export PICCL_REDUCE_ALGO=<RING/K_RING/PRD/PRM>
    export PICCL_REDUCE_SCATTER_ALGO=<RING/K_RING/PRD/PRM>
    export PICCL_BROADCAST_ALGO=<RING/K_RING/PRD/PRM>
```

To set different paramters, use the following environment variables
 ```
    export PICCL_ALLGATHER_K=<K>
    export PICCL_ALLREDUCE_K=<K>
    export PICCL_REDUCE_K=<K>
    export PICCL_REDUCE_SCATTER_K=<K>
    export PICCL_BROADCAST_K=<K>
```

To set different CUDA kernel block and grid sizes, use the following environment variables
```
    export PICCL_BLOCK_SIZE=<val>
    export PICCL_MIN_GRID_SIZE=<val>
    export PICCL_MAX_GRID_SIZE=<val>
```