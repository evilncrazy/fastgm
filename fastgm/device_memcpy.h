#pragma once

#include "device_ptr.h"

namespace fastgm {
   /**
    * Copy a device pointer to another device pointer
    *
    * @param to Destination of the copy
    * @param from Source of the copy
    * @param length Number of elements to copy
    */
   template <class T>
      __host__ void device_memcpy(device_ptr<T> &to, device_ptr<T> from, size_t length) {
         CUDA_SAFE_CALL(cudaMemcpy(to.get(), from.get(), length * sizeof(T), cudaMemcpyDeviceToDevice));
      }

   /**
    * Copy a device pointer to a raw host pointer
    *
    * @param to Destination of the copy
    * @param from Source of the copy
    * @param length Number of elements to copy
    */
   template <class T>
      __host__ void device_memcpy(T *to, device_ptr<T> from, size_t length) {
         CUDA_SAFE_CALL(cudaMemcpy(to, from.get(), length * sizeof(T), cudaMemcpyDeviceToHost));
      }

   /**
    * Copy a raw host pointer to a device pointer
    *
    * @param to Destination of the copy
    * @param from Source of the copy
    * @param length Number of elements to copy
    */
   template <class T>
      __host__ void device_memcpy(device_ptr<T> to, T *from, size_t length) {
         CUDA_SAFE_CALL(cudaMemcpy(to.get(), from, length * sizeof(T), cudaMemcpyHostToDevice));
      }
}
