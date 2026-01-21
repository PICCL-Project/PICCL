#include <stdint.h>
#include <stdio.h>

#include <reduce_kernel.h>

// Template for applying the reduction operation
template <typename T, ncclRedOp_t Op>
__device__ __forceinline__ T applyReduceOp(T a, T b, size_t count) {
    if constexpr (Op == ncclSum) return a + b;
    if constexpr (Op == ncclProd) return a * b;
    if constexpr (Op == ncclMax) return max(a, b);
    if constexpr (Op == ncclMin) return min(a, b);
    else return a; // Default to no-op
}

// Template-based reduction kernel
template <typename T, ncclRedOp_t Op>
__global__ void reduceKernel(T* redBuff, const T* op1, const T* op2, size_t count, size_t sections) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < count; i += blockDim.x * gridDim.x) {
        redBuff[i] = op1[i];
    }

    for (size_t sec = 0; sec < sections; ++sec){
        for (size_t i = idx; i < count; i += blockDim.x * gridDim.x) {
            T tmp = applyReduceOp<T, Op>(redBuff[i], op2[sec * count + i], count);
            redBuff[i] = tmp;
        }
    }
}

// Wrapper function to launch the kernel
template <typename T, ncclRedOp_t Op>
void launchReduceKernel(void* redBuff, const void* op1, const void* op2, size_t count, size_t sections, cudaStream_t stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 256;

    reduceKernel<T, Op><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        static_cast<T*>(redBuff), static_cast<const T*>(op1), static_cast<const T*>(op2), count, sections);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Function to launch the reduction kernel
void picclReduce(void* redBuff, const void* op1, const void* op2, size_t count, 
    size_t sections, ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream) {
    switch (datatype) {
        case ncclChar:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<char, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<char, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<char, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<char, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<char, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclChar: %d\n", op);
                    break;
            }
            break;
        case ncclUint8:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<uint8_t, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<uint8_t, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<uint8_t, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<uint8_t, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<uint8_t, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclUint8: %d\n", op);
                    break;
            }
            break;
        case ncclInt:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<int, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<int, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<int, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<int, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<int, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclInt: %d\n", op);
                    break;
            }
            break;
        case ncclUint32:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<uint32_t, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<uint32_t, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<uint32_t, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<uint32_t, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<uint32_t, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclUint32: %d\n", op);
                    break;
            }
            break;
        case ncclInt64:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<int64_t, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<int64_t, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<int64_t, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<int64_t, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<int64_t, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclInt64: %d\n", op);
                    break;
            }
            break;
        case ncclUint64:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<uint64_t, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<uint64_t, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<uint64_t, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<uint64_t, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<uint64_t, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclUint64: %d\n", op);
                    break;
            }
            break;
        case ncclFloat:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<float, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<float, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<float, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<float, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<float, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclFloat: %d\n", op);
                    break;
            }
            break;
        case ncclDouble:
            switch (op) {
                case ncclSum:
                    launchReduceKernel<double, ncclSum>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclProd:
                    launchReduceKernel<double, ncclProd>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMax:
                    launchReduceKernel<double, ncclMax>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclMin:
                    launchReduceKernel<double, ncclMin>(redBuff, op1, op2, count, sections, stream);
                    break;
                case ncclAvg:
                    launchReduceKernel<double, ncclAvg>(redBuff, op1, op2, count, sections, stream);
                    break;
                default:
                    // Handle unsupported operations
                    fprintf(stderr, "Unsupported operation for ncclDouble: %d\n", op);
                    break;
            }
            break;

        default:
            // Handle unsupported data types
            break;
    }
}
