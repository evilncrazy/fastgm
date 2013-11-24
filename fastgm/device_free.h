#pragma once

#include "device_ptr.h"

namespace fastgm {
   template <class T>
      /**
       * Frees memory associated with a device pointer
       */
      void device_free(device_ptr<T> ptr) {
         CUDA_SAFE_CALL(cudaFree(ptr.get()));
      }
}
