#pragma once

#include "../grid.h"
#include "inference_result.h"

#include "bp.cu"

namespace fastgm {
      class loopy_bp_inference {
         public:
            loopy_bp_inference(int max_iters = 10) : max_iters_(max_iters) { }

            template <class Grid>
            void operator() (const Grid &g, const device_array<float> w, inference_result &result) {
               using namespace bp_inference_kernel;
               
               dim3 grid_size(DIVUP(g.width() / 2 + 1, BLOCK_WIDTH), DIVUP(g.height(), BLOCK_HEIGHT), 1);
               dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT, 1);

               device_matrix<float> uf(g.size(), g.unary().num_params());
               device_matrix<float> pf(g.size(), g.pairwise().num_params());

               device_matrix<float> from(g.size(), 4, true);
               device_matrix<float> msgs(4 * g.size(), g.num_labels(), true);
               for (int t = 0; t < max_iters_; t++) {
                  // Collect messages from neighbours
                  collect_msgs<<<grid_size, block_size>>>(t, g, uf, w, msgs, from);

                  // Launch bp kernel for each direction
                  send_msg<<<grid_size, block_size>>>(t, g, DirUp, -g.width(), DirDown, pf, w, from, msgs);
                  send_msg<<<grid_size, block_size>>>(t, g, DirLeft, -1, DirRight, pf, w, from, msgs);
                  send_msg<<<grid_size, block_size>>>(t, g, DirDown, g.width(), DirUp, pf, w, from, msgs);
                  send_msg<<<grid_size, block_size>>>(t, g, DirRight, 1, DirLeft, pf, w, from, msgs);
               }

               // Find node and edge marginals
               collect_msgs<<<grid_size, block_size>>>(max_iters_, g, uf, w, msgs, from);
               node_marginals<<<DIVUP(g.size(), BLOCK_SIZE), BLOCK_SIZE>>>(g, from, result.node_marginals());
               edge_marginals<<<DIVUP(g.num_edges(), BLOCK_SIZE), BLOCK_SIZE>>>(g, pf, w, result.node_marginals(), result.edge_marginals());

               result.normalize();
               
               CUDA_CHECK_ERRORS;
               uf.release(); pf.release();
               from.release(); msgs.release();
            }
         private:
            int max_iters_;
      };
}
