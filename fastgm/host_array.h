#pragma once

namespace fastgm {
   /**
    * Represents an array stored on the host.
    */
   template <class T>
      class host_array {
         public:
            /**
             * Create an empty host array
             */
            host_array() : v_(0), size_(0) { }

            /**
             * Create a host array with a specific capacity
             *
             * @param size Capacity
             */
            explicit host_array(int size) : size_(size) {
               v_ = new T[size];
            }

            /**
             * Create a host array with a specific capacity
             * and an initial value
             *
             * @param size Capacity
             * @param val Initial value
             */
            host_array(int size, T val) : size_(size) {
               v_ = new T[size];
               for (int i = 0; i < size; i++) (*this)[i] = val;
            }

            /**
             * Copy a host array
             */
            host_array(const host_array<T> &other) : size_(other.size()) {
               v_ = new T[size_];
               for (int i = 0; i < size_; i++) (*this)[i] = other[i];
            }

            /**
             * Copy a subsequence of a host array 
             *
             * @param offset The start of the subsequence
             * @param length the length of the subsequence
             */
            host_array(const host_array<T> &other, int offset, int length) : size_(length) {
               v_ = new T[size_];
               for (int i = 0; i < size_; i++) (*this)[i] = other[offset + i];
            }

            /**
             * Deallocate memory associated with this host array
             */
            ~host_array() {
               if (v_) delete[] v_;
            }

            /**
             * Get a specific element from the array
             *
             * @param i The index of the element to get
             */
            T &operator[] (const int i) { return v_[i]; }
            const T &operator[] (const int i) const { return v_[i]; }

            /**
             * Get the size of the array
             */
            int size() const { return size_; }

            host_array<T> &operator=(const host_array<T> &other) {
               if (this == &other) return *this;
               if (size_ != other.size()) {
                  // Realloc memory so that we have the same size as the other array
                  size_ = other.size();

                  if (v_) delete[] v_;
                  v_ = new T[size_];
               }

               // Copy over the array
               for (int i = 0; i < size_; i++) (*this)[i] = other[i];
               return *this;
            }
            
            /**
             * Return a raw pointer to the start of the array
             */
            T *raw() const { return v_; }

         private:
            T *v_; // Raw pointer to array 
            int size_;
      };
}
