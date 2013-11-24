#include "gpu.h"

namespace fastgm {
   namespace reduce_kernel {
      const int BLOCK_SIZE = 512;

      template <typename T>
      __global__ void reduce(device_array<T> in, device_array<T> out) {
         __shared__ T s_data[BLOCK_SIZE];

         // Load into shared memory
         int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < in.size()) {
            s_data[tid] = in[i];
            __syncthreads();

            // Do reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
               if (tid < s) {
                  s_data[tid] += s_data[tid + s];
               }
               __syncthreads();
            }

            if (tid == 0) out[blockIdx.x] = s_data[0];
         } else {
            s_data[tid] = 0;
         }
      }
   }
}
