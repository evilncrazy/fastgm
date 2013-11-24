namespace fastgm {
   namespace inference_result_kernel {
      const int BLOCK_SIZE = 512;
      
      /**
       * Normalize each row of a matrix so that every row sums to 1
       */
      __global__ void normalize_rows(device_matrix<float> m) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;

         if (i < m.rows()) {
            float sum = 0;
            for (int j = 0; j < m.cols(); j++)
               sum += m(i, j);

            for (int j = 0; j < m.cols(); j++)
               m(i, j) /= sum;
         }
      }
   }
}
