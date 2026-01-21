#include <stdint.h>
#include <stdio.h>
#include <algorithm>

#include <permutation_kernel.h>


// Perform the permutation in-place using cycle decomposition.
// Not that this algorithms only works for at most 1024 nodes.
template <typename T>
__global__ void permutationInplaceKernel(T* permuteBuff, void* indices, size_t totalSize, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int* idxs = (int*) indices;

    for (size_t i = idx; i < totalSize; i += blockDim.x * gridDim.x) {
        const size_t group = i / count;
        size_t relative_idx = i % count;

        // Discover cycle
        size_t smallest_idx = group;
        size_t curr_idx = group;
        size_t next_idx = idxs[curr_idx];
        while (next_idx != group) {            
            
            if (next_idx < smallest_idx) {
                smallest_idx = next_idx;
            }

            curr_idx = next_idx;
            next_idx = idxs[curr_idx];
        }
    
        // Do nothig if it's not the smallest index (root)
        if (smallest_idx != group) continue;

        // Otherwise we want to traverse the cycle moving corresponding elements
        curr_idx = group;
        next_idx = idxs[curr_idx];
        while (next_idx != group) {

            // Move element
            T tmp = permuteBuff[next_idx * count + relative_idx];
            permuteBuff[next_idx * count + relative_idx] = permuteBuff[group * count + relative_idx];
            permuteBuff[group * count + relative_idx] = tmp;

            // Update index
            curr_idx = next_idx;
            next_idx = idxs[curr_idx];
        }
        
    }
}

template <typename T>
__global__ void permutationKernel(T* permuteBuff, const T* inBuff, void* indices, size_t totalSize, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int* idxs = (int*) indices;

    for (int i=idx; i < totalSize; i += blockDim.x * gridDim.x) {
        int group = i / count;
        int relative_idx = i % count;
    
        int groupIdx = idxs[group];

        permuteBuff[groupIdx * count + relative_idx] = inBuff[i];
    }

}

template <typename T>
void launchInplacePermutationKernel(void* permuteBuff, void* indices, size_t totalSize, size_t count,
    cudaStream_t stream) {
    // Launch the kernel with a grid size of (count + blockSize - 1) / blockSize
    unsigned long blockSize = globalBlockSize;
    unsigned long gridSize = std::min(std::max(static_cast<unsigned long>(count) / blockSize, globalMinGridSize), globalMaxGridSize);
    permutationInplaceKernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<T*>(permuteBuff), indices, totalSize, count);
}

template <typename T>
void launchPermutationKernel(void* permuteBuff, const void* inBuff, void* indices, size_t totalSize, size_t count,
    cudaStream_t stream) {
    // Launch the kernel with a grid size of (count + blockSize - 1) / blockSize
    unsigned long blockSize = globalBlockSize;
    unsigned long gridSize = std::min(std::max(static_cast<unsigned long>(count) / blockSize, globalMinGridSize), globalMaxGridSize);
    permutationKernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<T*>(permuteBuff), static_cast<const T*>(inBuff), indices, totalSize, count);
}

void picclPermuteInplace(void* permuteBuff, void* indices, size_t totalSize, size_t count, 
    ncclDataType_t datatype, cudaStream_t stream) {
    // Determine the data type of the input buffer
    switch (datatype) {
        case ncclChar:
            launchInplacePermutationKernel<char>(permuteBuff, indices, totalSize, count, stream);
            break;
        case ncclUint8:
            launchInplacePermutationKernel<uint8_t>(permuteBuff, indices, totalSize, count, stream);
            break;
        case ncclInt:
            launchInplacePermutationKernel<int>(permuteBuff, indices, totalSize, count, stream);
            break;
        case ncclUint32:
            launchInplacePermutationKernel<uint32_t>(permuteBuff, indices, totalSize, count, stream);
            break;
        case ncclInt64:
            launchInplacePermutationKernel<int64_t>(permuteBuff, indices, totalSize, count, stream);
            break;
        case ncclUint64:
            launchInplacePermutationKernel<uint64_t>(permuteBuff, indices, totalSize, count, stream);
            break;
        case ncclFloat:
            launchInplacePermutationKernel<float>(permuteBuff, indices, totalSize, count, stream);
            break;
        case ncclDouble:
            launchInplacePermutationKernel<double>(permuteBuff, indices, totalSize, count, stream);
            break;
        // case ncclBfloat16:
        //     launchInplacePermutationKernel<__nv_bfloat16>(permuteBuff, indices, totalSize, count, stream);
        //     break;
        default:
            // Handle unsupported data types
            break;
    }
}

void picclPermute(void* permuteBuff, const void* inBuff, void* indices, size_t totalSize, size_t count, 
    ncclDataType_t datatype, cudaStream_t stream) {
    // Determine the data type of the input buffer
    switch (datatype) {
        case ncclChar:
            launchPermutationKernel<char>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        case ncclUint8:
            launchPermutationKernel<uint8_t>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        case ncclInt:
            launchPermutationKernel<int>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        case ncclUint32:
            launchPermutationKernel<uint32_t>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        case ncclInt64:
            launchPermutationKernel<int64_t>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        case ncclUint64:
            launchPermutationKernel<uint64_t>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        case ncclFloat:
            launchPermutationKernel<float>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        case ncclDouble:
            launchPermutationKernel<double>(permuteBuff, inBuff, indices, totalSize, count, stream);
            break;
        // case ncclBfloat16:
        //     launchPermutationKernel<__nv_bfloat16>(permuteBuff, inBuff, indices, totalSize, count, stream);
        //     break;
        default:
            // Handle unsupported data types
            break;
    }
}

__device__ __forceinline__ int compute_d(int p, int b) {
    int d = 0;
    while (p > 1) {
        p /= b;
        ++d;
    }
    return d;
}

__device__ __forceinline__ int digitReverse(int x, int d, int b) {
    int y = 0;
    for (int i = 0; i < d; ++i) {
        int digit = x % b;
        y = y * b + digit;
        x /= b;
    }
    return y;
}

__device__ __forceinline__ int digitwiseAdd(int a, int c, int d, int b) {
    int res = 0, mul = 1;
    for (int i = 0; i < d; ++i) {
        int ai = a % b;
        int ci = c % b;
        int ri = (ai + ci) % b;   // digit‑wise modulo‑b add
        res += ri * mul;
        a /= b;  c /= b;  mul *= b;
    }
    return res;
}


__global__ void computePermutationKernel(void *perm, int r, int p, int b) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= p) return;

    int* perm_arr = (int*) perm;

    int d = compute_d(p, b);
    int rev = digitReverse(q, d, b);
    perm_arr[q] = digitwiseAdd(r, rev, d, b);
}

void launchComputePermutationKernel(void *d_perm, int r, int p, int b,
    cudaStream_t stream = nullptr) {
    constexpr int TPB = 256;
    int blocks = (p + TPB - 1) / TPB;
    computePermutationKernel<<<blocks, TPB, 0, stream>>>(d_perm, r, p, b);
}