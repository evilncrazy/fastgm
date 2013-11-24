#pragma once

namespace fastgm {
   /**
    * Represents a 2D matrix stored on host memory
    */
   template <class T>
      class host_matrix {
         public:
            typedef T value_type;

            /**
             * Create a host matrix with specified dimensions
             */
            host_matrix(int rows, int cols) : rows_(rows), cols_(cols) {
               data_ = new T[rows * cols];
            }

            /**
             * Create a host matrix with specified dimensions and initialize
             * all the elements to a specific value
             *
             * @param val The initial value of all the elements
             */
            host_matrix(int rows, int cols, int val) : rows_(rows), cols_(cols) {
               data_ = new T[rows * cols];
               for (int i = 0; i < rows * cols; i++) data_[i] = val;
            }

            /**
             * Return the number of rows in the matrix
             */
            int rows() const {
               return rows_;
            }

            /**
             * Return the number of cols in the matrix
             */
            int cols() const {
               return cols_;
            }

            /**
             * Access an element of the matrix
             */
            T &operator ()(int r, int c) {
               return data_[r * cols_ + c];
            }
            
            T *raw() const { return data_; }

         private:
            int rows_, cols_;
            T *data_;
      };
}
