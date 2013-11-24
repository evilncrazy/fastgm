#pragma once

namespace fastgm {
   namespace bp_decoding_kernel {
      const int BLOCK_SIZE = 512;
      const int BLOCK_WIDTH = 64;
      const int BLOCK_HEIGHT = 4;

      __device__ int msg_from(int i, int d) {
         return 4 * i + d;
      }

      template <class Grid>
         __global__ void collect_msgs(int iter, Grid g, device_matrix<float> uf, device_array<float> w, device_matrix<float> msgs, device_matrix<float> from) {
            int y = blockIdx.y * blockDim.y + threadIdx.y; 
            int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((y + iter) & 1);

            if (x > 0 && y > 0 && x < g.width() - 1 && y < g.height() - 1) {
               int i = y * g.width() + x;

               // Collect messages from the four neighbours
               for (int p = 0; p < g.num_labels(); p++) {
                  from(i, p) =
                     g(uf.row(i), w, i, p) +
                     msgs(msg_from(i + g.width(), DirUp), p) + 
                     msgs(msg_from(i - 1, DirRight), p) +
                     msgs(msg_from(i - g.width(), DirDown), p) +
                     msgs(msg_from(i + 1, DirLeft), p);
               }
            }
         }

      template <class Grid>
         __global__ void send_msgs_decode(int iter, Grid g, Direction dir, int opp_delta, Direction opp_dir, device_matrix<float> pf, device_array<float> w, device_matrix<float> from, device_matrix<float> msgs) {
            int y = blockIdx.y * blockDim.y + threadIdx.y; 
            int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((y + iter) & 1);

            if (x > 0 && y > 0 && x < g.width() - 1 && y < g.height() - 1) {
               int i = y * g.width() + x;

               float sum = 0;
               for (int p = 0; p < g.num_labels(); p++) {
                  // Find the label which maximizes the sum of messages and pairwise potential
                  float maxpot = -FLT_MAX;
                  for (int q = 0; q < g.num_labels(); q++) {
                     maxpot = max(maxpot, g(pf.row(i), w, g.edge_from(i, dir), p, q) +
                        from(i, q) - msg_from(i + opp_delta, opp_dir));
                  }

                  sum += msgs(msg_from(i, dir), p) = maxpot;
               }

               // Normalize messages
               sum /= g.num_labels();
               for (int p = 0; p < g.num_labels(); p++) {
                  msgs(msg_from(i, dir), p) -= sum;
               }
            }
         }

      template <class Grid>
         __global__ void get_labeling(Grid g, device_matrix<float> from, device_array<int> result) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < g.size()) {
               float max = -FLT_MAX; int max_label = 0;
               for (int p = 0; p < g.num_labels(); p++) {
                  float r = from(i, p);
                  if (r > max) {
                     max = r;
                     max_label = p;
                  }
               }

               result[i] = max_label;
            }
         }
   }
}
