#pragma once

#include "../gpu.h"

namespace fastgm {
   namespace bp_inference_kernel {
      const int BLOCK_SIZE = 256;
      const int BLOCK_WIDTH = 64;
      const int BLOCK_HEIGHT = 4;
      
      __device__ int msg_from(int i, int d) {
         return 4 * i + d;
      }

      template <class Grid>
         __global__ void collect_msgs(int iter, Grid g, device_matrix<float> uf, device_array<float> w, device_matrix<float> msgs, device_matrix<float> from) {
            int y = blockIdx.y * blockDim.y + threadIdx.y; 
            int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((y + iter + 1) & 1);

            if (x > 0 && y > 0 && x < g.width() - 1 && y < g.height() - 1) {
               int i = y * g.width() + x;

               // Collect messages from the four neighbours
               for (int p = 0; p < g.num_labels(); p++) {
                  from(i, p) = expf(g(uf.row(i), w, i, p)) *
                     msgs(msg_from(i + g.width(), DirUp), p) *
                     msgs(msg_from(i - 1, DirRight), p) *
                     msgs(msg_from(i - g.width(), DirDown), p) *
                     msgs(msg_from(i + 1, DirLeft), p);
               }
            }
         }

      template <class Grid>
         __global__ void send_msg(int iter, Grid g, Direction dir, int opp_delta, Direction opp_dir, device_matrix<float> pf, device_array<float> w, device_matrix<float> from, device_matrix<float> msgs) {
            int y = blockIdx.y * blockDim.y + threadIdx.y; 
            int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((y + iter) & 1);

            if (x > 0 && y > 0 && x < g.width() - 1 && y < g.height() - 1) {
               int i = y * g.width() + x;

               float sum = 0;
               for (int p = 0; p < g.num_labels(); p++) {
                  // Update the messages that this node sends to its neighbours
                  msgs(msg_from(i, dir), p) = 0;
                  for (int q = 0; q < g.num_labels(); q++) {
                     msgs(msg_from(i, dir), p) += 
                        expf(g(pf.row(i), w, g.edge_from(i, dir), p, q)) * from(i, q) /
                           msgs(msg_from(i + opp_delta, opp_dir), q);
                  }
               }

               // Normalize messages
               sum /= g.num_labels();
               for (int p = 0; p < g.num_labels(); p++) {
                  msgs(msg_from(i, dir), p) -= sum;
               }
            }
         }
         
      template <class Grid>
         __global__ void node_marginals(Grid g, device_matrix<float> from, device_matrix<float> result) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < g.size()) {
               for (int j = 0; j < g.num_labels(); j++) {
                  result(i, j) = from(i, j);
               }
            }
         }

      template <class Grid>
         __global__ void edge_marginals(Grid g, device_matrix<float> pf, device_array<float> w, device_matrix<float> node_marginal, device_matrix<float> result) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i < g.num_edges()) {
               grid_edge edge = g.edge(i);

               for (int p = 0; p < g.num_labels(); p++) {
                  for (int q = 0; q <= p; q++) {
                     result(i, p * g.num_labels() + q) =
                     result(i, q * g.num_labels() + p) =
                        node_marginal(edge.first(), p) *
                        node_marginal(edge.second(), q) *
                        expf(g(pf.row(i), w, edge, p, q));
                  }
               }
            }
         }
   }
}
