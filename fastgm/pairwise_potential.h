#pragma once

#include "potential.h"
#include "device_matrix.h"
#include "edge.h"

namespace fastgm {
   /**
    * Represents a parameterized pairwise potential function. Each parameter is
    * associated with a feature function. The feature functions have access
    * to data of type FeatureData for each node.
    */
   template <typename T, class FeatureData> 
      class pairwise_param {
         public:
            /**
             * Construct a parameterized pairwise potential.
             *
             * @param num_params The number of parameters (and hence features)
             *    used in this potential.
             */
            __host__ pairwise_param(int num_params) :
               num_params_(num_params) { }

            /**
             * Set the feature data.
             */
            __host__ void set_data(const device_array<FeatureData> &data) {
               data_ = data;
            }

            /**
             * Return the number of parameters in this potential function.
             */
            __host_device__ int num_params() const { return num_params_; }

         protected:
            /**
             * Get the all the feature data as an array.
             */
            const device_array<FeatureData> &data() const { return data_; }

            /**
             * Get the feature data of a particular node
             */
            __device__ const FeatureData& data(int x) const {
               return data_[x];
            }
         private:
            int num_params_;
            device_array<FeatureData> data_;
      };

   /**
    * Represents a pairwise potential function which returns a constant value.
    * It is parameterized on that value.
    */
   template <typename T>
      class const_pairwise_param : public pairwise_param<T, no_features> {
         public:
            /**
             * Constructs a pairwise potential function which returns a constant value
             *
             * @param val The value to return. Defaults to 0.
             */
            __host__ const_pairwise_param(T val = 0) : pairwise_param<T, no_features>(1) {
               this->set_params(device_array<T>(1, val));
            }

            __device__ void compute_features(device_array<T> f,
                  const grid_edge &, int, int) const {
               f[0] = 1;
            }
      };

   /**
    * Represents a pairwise potential with explicitly specified values
    */
   template <typename T>
      class explicit_pairwise {
         public:
            /**
             * Construct a pairwise potential with explicity specified values.
             *
             * @param num_labels Number of labels each node can be in
             * @param pot An array containing the potential value for each label
             */
            __host__ explicit explicit_pairwise(int num_labels, device_array<T> pot) :
               pot_(pot) { }
            
            __device__ T operator() (const grid_edge &, int p, int q) const {
               return pot_(p, q);
            }
         private:
            device_matrix<T> pot_;
      };
      
   /**
    * Represents a Potts pairwise potential
    */
   template <typename T, class F = no_features>
      class potts : public pairwise_param<T, F> {
         public:
            __host__ explicit potts(T val) : pairwise_param<T, F>(1),
               val_(val) { }
            
            __device__ void compute_features(device_array<T> f,
                  const grid_edge &, int p, int q) const {
               f[0] = (p != q) * val_;
            }
         private:
            T val_;
      };
      
   /**
    * Represents a linear potential
    */
   template <typename T, class F = no_features>
      class pairwise_linear : public pairwise_linear<T, F> {
         public:
            __host__ explicit pairwise_linear(T s) : pairwise_param<T, F>(1),
               val_(val) { }
            
            __device__ void compute_features(device_array<T> f,
                  const grid_edge &, int p, int q) const {
               f[0] = val_ * abs(p - q);
            }
         private:
            T val_;
      };
}
