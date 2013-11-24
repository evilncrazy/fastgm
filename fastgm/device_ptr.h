#pragma once

namespace fastgm {
   template <typename T>
      class device_ptr {
         public:
            /**
             * Construct a null device pointer
             */
            __host_device__ device_ptr() : d_ptr_(0) { }

            /**
             * Copy a raw pointer that is pointing to device memory
             *
             * @param raw The raw pointer to copy from
             */
            __host_device__ explicit device_ptr(T *raw) : d_ptr_(raw) { }

            /**
             * Returns the raw pointer stored in this device pointer
             */
            __host_device__ T *get() const { return d_ptr_; }
         private:
            T *d_ptr_; // A raw pointer to memory on the device
      };
}

