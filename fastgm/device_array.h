#pragma once

#include "host_array.h"
#include "gpu.h"

namespace fastgm {
   /**
    * Represents an array stored on the device. Memory has to manually deallocated
    * by calling the release() function.
    */
   template <class T>
      class device_array {
         public:
            /**
             * Construct an empty array stored in device memory
             */
            __host__ device_array() : size_(0), v_(NULL) { }

            /**
             * Create a device array with a specific capacity
             *
             * @param size Capacity
             */
            __host__ explicit device_array(int size, bool zeroed = false) : size_(size) {
               v_ = device_malloc<T>(size);
               if (zeroed) reset();
            }

            /**
             * Create a device array with a specific capacity
             * and an initial value
             *
             * @param size Capacity
             * @param val The initial value to fill the array with
             */
            __host__ device_array(int size, T val) : size_(size) {
               // First allocate the array on the host and initialize all its
               // elements to val
               T *host = static_cast<T>(malloc(size * sizeof(T)));
               for (int i = 0; i < size; i++) host[i] = val;

               // Copy it over to device memory
               v_ = device_malloc<T>(size);
               device_memcpy(v_, host, size);

               free(host);
            }

            /**
             * Create a device array from a host array, transfering all the
             * contents in the host array to device memory.
             *
             * @param other The host array to copy from
             */
            __host__ device_array(const host_array<T> &other) : size_(other.size()) {
               v_ = device_malloc<T>(other.size());
               device_memcpy(v_, other.raw(), other.size());
            }

            /**
             * Create a device array from another device array by pointing to
             * the same memory. No new device memory is allocated.
             *
             * @param other The device array to copy from
             */
            __host_device__ device_array(const device_array<T> &other) : v_(other.ptr()), size_(other.size()) { }

            /**
             * Create a device array from a raw device pointer and a specific length.
             * No new device memory is allocated.
             *
             * @param ptr A pointer to the start of the array
             * @param size The length of the array
             */
            __host_device__ device_array(const device_ptr<T> &ptr, int size) : v_(ptr), size_(size) { }

            /**
             * Create a device array from a section of another device array.
             * No new device memory is allocated.
             *
             * @param other The device array to copy from
             * @param offset The start index of the subarray
             * @param length The length of the subarray
             */
            __host_device__ device_array(const device_array<T> &other, int offset, int length) :
               v_(other.ptr().get() + offset), size_(length) { }

            /**
             * Reset all the elements of the array to 0
             */
            __host__ void reset() {
               if (v_.get())
                  CUDA_SAFE_CALL(cudaMemset(v_.get(), 0, size_ * sizeof(T)));
            }

            /**
             * Transfer all the elements in the array back into host memory.
             *
             * @param dest The host array to store the data in
             */
            __host__ host_array<T> to_host() const {
               host_array<T> dest(size_);
               device_memcpy(dest.raw(), v_, size_);
               return dest;
            }

            /**
             * Release the memory used by this array. This invalidates any other
             * device arrays that point to the same memory.
             */
            __host__ void release() {
               device_free(v_);
            }

            /**
             * Get a specific element from the array
             * @param i The index of the element to get
             */
            __device__ T &operator[] (const int i) { return v_.get()[i]; }
            __device__ T &operator[] (const int i) const { return v_.get()[i]; }

            /**
             * Get the size of the array
             */
            __host_device__ int size() const { return size_; }
            
            /**
             * Return the device pointer pointing to the start of the array
             */
            __host_device__ device_ptr<T> ptr() const { return v_; }

         private:
            device_ptr<T> v_; // Device pointer to array 
            int size_;
      };
}
