#pragma once

#include "potential.h"
#include "device_array.h"

namespace fastgm {
   /**
    * Represents a parameterized unary potential function. Each parameter is
    * associated with a feature function. The feature functions have access
    * to data of type FeatureData for each node.
    */
   template <class T, class FeatureData>
      class unary_param {
         public:
            /**
             * Construct a parameterized unary potential.
             *
             * @param num_params The number of parameters (and hence features)
             *    used in this potential.
             */
            __host__ explicit unary_param(int num_params) :
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
            __host_device__ const device_array<FeatureData> &data() const {
               return data_;
            }

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
    * Represents a unary potential function which returns a constant value.
    * It is parameterized on that value.
    */
   template <typename T>
      class const_unary_param : public unary_param<T, no_features> {
         public:
            /**
             * Constructs a unary potential function which returns a constant value
             *
             * @param val The value to return. Defaults to 0.
             */
            __host__ explicit const_unary_param(T val = 0) : unary_param<T, no_features>(1) {
               this->set_params(device_array<T>(1, val));
            }

            __device__ void compute_features(device_array<T> f, int, int) const {
               f[0] = 1;
            }
      };

   /**
    * Represents a unary potential with explicitly specified values.
    */
   template <class T>
      class explicit_unary {
         public:
            /**
             * Construct a unary potential with explicity specified values.
             *
             * @param num_labels Number of labels each node can be in
             * @param pot An array containing the potential value for each label
             */
            __host__ explicit explicit_unary(int num_labels, device_array<T> pot) :
               pot_(pot) { }

            __device__ T operator() (int, int l) const {
               return pot_[l];
            }

         private:
            device_array<T> pot_;
      };
}
