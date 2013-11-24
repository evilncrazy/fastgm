#pragma once

#include "device_ptr.h"

namespace fastgm {
   /**
    * Allocate memory on the device, enough to hold 'length' number of
    * objects of type T
    *
    * @param length The number of objects of type T that this memory block
    *    needs to be able to hold. Defaults to 1.
    */
   template <class T>
      __host__ device_ptr<T> device_malloc(size_t length = 1) {
         // Create a pointer to be allocated with device memory
         T *data;
         CUDA_SAFE_CALL(cudaMalloc((void **)&data, length * sizeof(T)));
         return device_ptr<T>(data);
      }
}
