#pragma once

#include "gpu.h"

namespace fastgm {
   namespace regularise_kernel {
      __global__ void update_gradient(device_array<float> grad, device_array<float> w, device_scalar<T> result) {
         for (int i = 0; i < w.size(); i++) {
            result.value() += w[i] * w[i];
            grad[i] += 2 * lambda * w[i];
         }
         
         result.value() *= lambda;
      }
   }
}
