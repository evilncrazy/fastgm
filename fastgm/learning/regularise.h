#pragma once

#include "device_array.h"

namespace fastgm {
   float l2norm(const device_array<float> &w, device_array<float> &grad, float lambda) {
      // Since the number of parameters is usually small, it's more efficient
      // to just launch a single kernel to add it sequentially than to do any
      // reduction operations
      device_scalar<T> result(0);
      update_gradient<<<1, 1>>>(grad, w, result);
      float value = result.to_host();
      result.release();

      return value;
   }
}
