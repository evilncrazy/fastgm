#pragma once

#include "device_ptr.h"

namespace fastgm {
   template <class T>
      /**
       * Sets the memory associated with a ptr to a particular value
       */
      void device_free(device_ptr<T> &ptr) {
         CUDA_SAFE_CALL(ptr.get());
      }
}
