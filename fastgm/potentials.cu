#pragma once

#include "gpu.h"

namespace fastgm {
   namespace potentials_kernel {
      const int BLOCK_SIZE = 512;

      template <class Grid>
         __global__ void node_pot(Grid g, device_matrix<float> uf, device_array<float> w, device_array<int> labeling, device_array<float> out) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < g.size())
               out[i] = g(uf.row(i), w, i, labeling[i]);
         }

      template <class Grid>
         __global__ void edge_pot(Grid g, device_matrix<float> pf, device_array<float> w, device_array<int> labeling, device_array<float> out) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < g.num_edges()) {
               grid_edge edge = g.edge(i);
               out[i] = g(pf.row(i), w, edge, labeling[edge.first()], labeling[edge.second()]);
            }
         }
   }
}
