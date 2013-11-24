#pragma once

#include "host_array.h"
#include "potentials.cu"
#include "reduce.h"

namespace fastgm {
   template <class Grid>
      float log_labeling_potential(const Grid &g, device_array<float> w, device_array<int> labeling) {
         using namespace potentials_kernel;
         
         // Launch kernels to compute the unary and pairwise potentials in parallel
         device_array<float> upot(g.size());
         device_array<float> ppot(g.num_edges());

         device_matrix<float> uf(g.size(), g.num_labels());
         device_matrix<float> pf(g.num_edges(), g.num_labels());

         // Launch kernels to compute unary and pairwise potentials
         node_pot<<<DIVUP(g.size(), BLOCK_SIZE), BLOCK_SIZE>>>(g, uf, w, labeling, upot);
         edge_pot<<<DIVUP(g.num_edges(), BLOCK_SIZE), BLOCK_SIZE>>>(g, pf, w, labeling, ppot);

         // Add up all the potentials
         float result = sum_reduce(upot) + sum_reduce(ppot);

         uf.release(); pf.release();
         upot.release(); ppot.release();
         
         return result;
      }

   template <class Grid>
      float labeling_potential(const Grid &g, const host_array<int> &labeling) {
         return exp(log_labeling_potential(g, labeling));
      }
}
