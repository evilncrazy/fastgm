#pragma once

#include <cfloat>

namespace fastgm {
   namespace likelihood_kernel {
      const int BLOCK_SIZE = 256;
      const int BLOCK_WIDTH = 64;
      const int BLOCK_HEIGHT = 4;

      __device__ float safe_log(float x) {
         return x < 1e-9 ? -FLT_MAX : log(x);
      }

      /**
       * Compute the local probabilities for each node, storing them in the pot matrix
       */
      template <class Grid>
      __global__ void pseudo_prob(Grid g, device_matrix<float> uf, device_matrix<float> pf, device_array<float> w, device_array<int> labeling, device_matrix<float> pot) {
         int x = blockIdx.x * blockDim.x + threadIdx.x;
         int y = blockIdx.y * blockDim.y + threadIdx.y;

         if (x >= 0 && y >= 0 && x < g.width() && y < g.height()) {
            int i = y * g.width() + x;

            // Compute probability of each local configuration as given in this labeling
            float sum = 0;
            for (int j = 0; j < g.num_labels(); j++) {
               sum += pot(i, j) = expf(g(uf.row(i), w, i, j) +
                   (i < g.width() ? 0 : g(pf.row(i), w, g.edge_between(i - g.width(), i), j, labeling[i - g.width()])) +
                   (i % g.width() == 0 ? 0 : g(pf.row(i), w, g.edge_between(i - 1, i), j, labeling[i - 1])) +
                   (x % g.width() == g.width() - 1 ? 0 : g(pf.row(i), w, g.edge_between(i, i + 1), j, labeling[i + 1])) +
                   (i >= g.width() * (g.height() - 1) ? 0 : g(pf.row(i), w, g.edge_between(i, i + g.width()), j, labeling[i + g.width()])));
            }

            // Normalize the potentials
            for (int j = 0; j < g.num_labels(); j++) {
               pot(i, j) /= sum;
            }
         }
      }
      
      /**
       * Compute the unary probabilities for each node, storing them in the pot matrix
       */
      template <class Grid>
      __global__ void unary_prob(const Grid g, device_matrix<float> uf, device_array<float> w, device_matrix<float> pot) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;

         if (i < g.size()) {
            // Compute probability of each node at each label
            float sum = 0;
            for (int j = 0; j < g.num_labels(); j++) {
               sum += pot(i, j) = expf(g(uf.row(i), w, i, j));
            }

            // Normalize the potentials
            for (int j = 0; j < g.num_labels(); j++) {
               pot(i, j) /= sum;
            }
         }
      }
      
      /**
       * Compute the pairwise probabilities for each edge, storing them in the pot matrix
       */
      template <class Grid>
      __global__ void pairwise_prob(Grid g, device_matrix<float> pf, device_array<float> w, device_matrix<float> pot) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;

         if (i < g.num_edges()) {
            grid_edge edge = g.edge(i);
            
            // Compute probability of each edge at each label
            float sum = 0;
            for (int p = 0; p < g.num_labels(); p++) {
               for (int q = 0; q < g.num_labels(); q++) {
                  sum += pot(i, p * g.num_labels() + q) /*= pot(i, q * g.num_labels() + p)*/ = expf(g(pf.row(i), w, edge, p, q));
               }
            }

            // Normalize the potentials
            for (int p = 0; p < g.num_labels(); p++) {
               for (int q = 0; q < g.num_labels(); q++) {
                  pot(i, p * g.num_labels() + q) /= sum;
               }
            }
         }
      }

      /**
       * Compute the gradients for unary parameters, separately for each node
       * to prevent data races.
       */
      template <class Grid>
         __global__ void compute_node_gradient(Grid g, device_matrix<float> grad, device_matrix<float> pot, device_matrix<float> uf, device_array<int> labeling) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < g.size()) {
               for (int f = 0; f < uf.cols(); f++)
                  grad(f, i) = 0;
               
               for (int x = 0; x < g.num_labels(); x++) {
                  g.unary().compute_features(uf.row(i), i, x);

                  float l = pot(i, x) - (x == labeling[i]);
                  for (int f = 0; f < uf.cols(); f++) {
                     grad(f, i) += uf(i, f) * l;
                  }
               }
            }
         }

      /**
       * Compute the gradients for pairwise parameters for pseudolikelihood,
       * separately for each edge to prevent data races.
       */
      template <class Grid>
         __global__ void compute_edge_gradient_pseudo(Grid g, device_matrix<float> grad, device_matrix<float> pot, device_matrix<float> pf, device_array<int> labeling) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < g.num_edges()) {
               grid_edge edge = g.edge(i);

               for (int f = 0; f < pf.cols(); f++)
                  grad(f, i) = 0;

               for (int p = 0; p < g.num_labels(); p++) {
                  g.pairwise().compute_features(pf.row(i), edge, p, labeling[edge.second()]);
                  float l = pot(edge.first(), p) - (p == labeling[edge.first()]);
                  for (int f = 0; f < pf.cols(); f++) {
                     grad(f, i) += pf(i, f) * l;
                  }

                  g.pairwise().compute_features(pf.row(i), edge, p, labeling[edge.first()]);
                  l = pot(edge.second(), p) - (p == labeling[edge.second()]);
                  for (int f = 0; f < pf.cols(); f++) {
                     grad(f, i) += pf(i, f) * l;
                  }
               }
            }
         }
         
      template <class Grid>
         __global__ void compute_edge_gradient(Grid g, device_matrix<float> grad, device_matrix<float> pot, device_matrix<float> pf, device_array<int> labeling) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < g.num_edges()) {
               grid_edge edge = g.edge(i);

               for (int f = 0; f < pf.cols(); f++)
                  grad(f, i) = 0;

               for (int p = 0; p < g.num_labels(); p++) {
                  for (int q = 0; q <= p; q++) {
                     g.pairwise().compute_features(pf.row(i), edge, p, q);
                     
                     float l = pot(i, p * g.num_labels() + q) -
                               ((p == labeling[edge.first()] && q == labeling[edge.second()]) ||
                               (p == labeling[edge.second()] && q == labeling[edge.first()]));
                     for (int f = 0; f < pf.cols(); f++) {
                        grad(f, i) += pf(i, f) * l;
                     }
                  }
               }
            }
         }

      /**
       * Compute the sum of the columns in each row of a matrix, storing them in
       * a vector.
       */
      __global__ void sum_cols(device_matrix<float> m, int offset, device_array<float> result) {
         // TODO: could be implemented as a reduce operation
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < m.rows()) {
            for (int c = 0; c < m.cols(); c++) {
               result[i + offset] += m(i, c);
            }
         }
      }
   }
}
