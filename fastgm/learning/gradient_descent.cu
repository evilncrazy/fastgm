namespace fastgm {
   namespace gradient_descent_kernel {
      const int NUM_ITERATIONS = 10;

      __global__ void update_weights(device_array<float> w, device_array<float> grad, float t) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < w.size()) w[i] -= t * grad[i];
      }
   }
}
