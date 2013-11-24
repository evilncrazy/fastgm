#pragma once

#include "gpu.h"
#include "host_matrix.h"

namespace fastgm {
   /**
    * Represents a 2D matrix stored on device memory. Memory has to be manually
    * deallocated using the release() function.
    */
   template <class T> 
      class device_matrix {
         public:
            typedef T value_type;

            /**
             * Create a device matrix with specified dimensions
             */
            device_matrix(int rows, int cols, bool zeroed = false) : rows_(rows), cols_(cols) {
               T *data;
               CUDA_SAFE_CALL(cudaMallocPitch((void **)&data, &pitch_,
                     cols * sizeof(T), rows));
               data_ = device_ptr<T>(data);

               if (zeroed) reset();
            }

            /**
             * Create a device matrix from a host matrix
             */
            device_matrix(const host_matrix<T> &matrix) : rows_(matrix.rows()), cols_(matrix.cols()) {
               T *data;
               CUDA_SAFE_CALL(cudaMallocPitch((void **)&data, &pitch_,
                     matrix.cols() * sizeof(T), matrix.rows()));
               CUDA_SAFE_CALL(cudaMemcpy2D(data, pitch_,	
                     matrix.raw(), matrix.cols() * sizeof(T), matrix.cols() * sizeof(T),
                     matrix.rows(), cudaMemcpyHostToDevice));
               data_ = device_ptr<T>(data);
            }

            /**
             * Return the number of rows in the matrix
             */
            __host_device__ int rows() const {
               return rows_;
            }

            /**
             * Return the number of cols in the matrix
             */
            __host_device__ int cols() const {
               return cols_;
            }
            
            /**
             * Return the number of bytes per row. Includes any extra padding
             * bytes.
             */
            __host_device__ size_t pitch() const {
               return pitch_;
            }

            /**
             * Access an element of the matrix
             */
            __device__ T &operator ()(int r, int c) {
               return *((T *)((char *)data_.get() + r * pitch_) + c);
            }

            /**
             * Return a device array representing a particular row of the matrix
             */
            __host_device__ device_array<T> row(int r) const {
               return device_array<T>(device_ptr<T>((T *)((char *)data_.get() + r * pitch_)), cols_);
            }

            /**
             * Reset all the elements of the matrix to 0
             */
            __host__ void reset() {
               if (data_.get())
                  CUDA_SAFE_CALL(cudaMemset2D(data_.get(), pitch_, 0, cols_ * sizeof(T), rows_));
            }

            /**
             * Copy a device matrix back to host matrix
             */
            host_matrix<T> to_host() {
               host_matrix<T> matrix(rows_, cols_);
               CUDA_SAFE_CALL(cudaMemcpy2D(matrix.raw(), cols_ * sizeof(T),
                     data_.get(), pitch_, cols_ * sizeof(T),
                     rows_, cudaMemcpyDeviceToHost));
               return matrix;
            }
            
            /**
             * Free any memory associated with this matrix. Invalidates any
             * other pointers to this matrix.
             */
            void release() {
               if (data_.get()) {
                  CUDA_SAFE_CALL(cudaFree(data_.get()));
                  data_ = device_ptr<T>(); // INvalidate the pointer
               }
            }

         private:
            int rows_, cols_;
            device_ptr<T> data_;
            size_t pitch_;
      };
}
