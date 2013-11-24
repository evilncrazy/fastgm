#pragma once

#include "../grid.h"

namespace fastgm {
   template <typename T>
      class icm_decoding {
         public:
            icm_decoding(int max_iters = 10) : max_iters_(max_iters) { }

            template <class Grid>
            host_array<int> operator() (const Grid &g) {
               host_array<int> result(g.size());

               // Initial configuration maximizes the unary potentials
               for (int i = 0; i < g.size(); i++) {
                  T maxEnergy = std::numeric_limits<T>::lowest();
                  for (int p = 0; p < g.num_labels(); p++) {
                     T energy = g.unary()(i, p);
                     if (energy > maxEnergy) {
                        result[i] = p;
                        maxEnergy = energy;
                     }
                  }
               }

               for (int t = 0; t < max_iters_; t++) {
                  for (int y = 1; y < g.height() - 1; y++) {
                     for (int x = 1; x < g.width() - 1; x++) {
                        int i = y * g.width() + x;

                        // Calculate local energy
                        T maxEnergy = std::numeric_limits<T>::lowest();
                        for (int p = 0; p < g.num_labels(); p++) {
                           T energy =
                              g.unary()(i, p) +
                              g.pairwise()(g.edge_from(i, DirUp), p, result[i - g.width()]) +
                              g.pairwise()(g.edge_from(i, DirRight), p, result[i + 1]) +
                              g.pairwise()(g.edge_from(i, DirDown), p, result[i + g.width()]) +
                              g.pairwise()(g.edge_from(i, DirLeft), p, result[i - 1]);
                           
                           if (energy > maxEnergy) {
                              result[i] = p;
                              maxEnergy = energy;
                           }
                        }
                     }
                  }
               }

               return result;
            }
         private:
            int max_iters_;
      };
}
