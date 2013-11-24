#pragma once

#include "reduce.cu"

namespace fastgm {
   template <typename T>
      T sum_reduce(device_array<T> array) {
         using namespace reduce_kernel;

         device_array<T> out(DIVUP(array.size(), BLOCK_SIZE));
         reduce<<<DIVUP(array.size(), BLOCK_SIZE), BLOCK_SIZE>>>(array, out);

         host_array<T> result = out.to_host();
         printf("%f ", result[0]);
         float sum = 0;
         for (int i = 0; i < result.size(); i++) sum += result[i];

         out.release();
         return sum;
      }
}
