#pragma once

#include "gpu.h"

namespace fastgm {
   /**
    * Represents a scalar value stored on the device. Memory has to manually deallocated
    * by calling the release() function.
    */
   template <class T>
      class device_scalar {
         public:
            /**
             * Create a device scalar value
             */
            __host__ device_scalar(T value) : v_(device_malloc<T>(1)) {
               device_memcpy(v_, &value, 1);
            }

            /**
             * Transfer the value back into host memory.
             *
             * @param ptr A host pointer to where to copy the value
             */
            __host__ T *to_host() const {
               T *raw = (T *)malloc(sizeof(T));
               device_memcpy(raw, v_, 1);
               T value = *raw;
               free(raw);
               return value;
            }

            /**
             * Return a device reference to the value
             */
            __device__ &value() { return *(v_->get()); }

            /**
             * Release the memory used by this array. This invalidates any other
             * device arrays that point to the same memory.
             */
            __host__ void release() {
               device_free(v_);
            }

         private:
            device_ptr<T> v_; // Device pointer to the value
      };
}
