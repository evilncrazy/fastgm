#pragma once

#include "../device_array.h"

#include "gradient_descent.cu"

namespace fastgm {
   /**
    * A very simple gradient descent function.
    *
    * @param w Array of weights. Modified with new weights.
    * @param grad Array of gradients with same size as w. Modified with new gradients.
    * @param grad_func A function or functor that represents the function being optimized.
    *    It should take in an array of weights and array of gradients as arguments,
    *    and writes the gradients of the function into the gradient parameter.
    * @param t The step size
    */
   template <class Func>
      void gradient_descent(device_array<float> &w, device_array<float> &grad, Func &grad_func, float t) {
         using namespace gradient_descent_kernel;

         for (int i = 0; i < NUM_ITERATIONS; i++) {
            grad_func(w, grad);
            update_weights<<<DIVUP(w.size(), BLOCK_SIZE), BLOCK_SIZE>>>(w, grad, t);
         }
      }
}
