#pragma once

#include "inference_result.cu"

namespace fastgm {
   /**
    * Represents the result of running inference on a grid
    */
      class inference_result {
         public:
            inference_result(int size, int num_edges, int num_labels) :
               num_labels_(num_labels),
               node_m_(size, num_labels, true),
               edge_m_(num_edges, num_labels * num_labels, true),
               log_z_(0) {
            }

            ~inference_result() {
               node_m_.release(); edge_m_.release();
            }

            /**
             * Get or set the marginal probability of a node given a particular label
             *
             * @param i The index of the node
             * @param x The label of the node
             */
            __device__ float &node_marginal(int i, int x) {
               return node_m_(i, x);
            }

            /**
             * Get or set the marginal probability of a edge given particular labels
             * for its endpoints
             *
             * @param p The label of one of the endpoints
             * @param q The label of the other endpoint
             */
            __device__ float &edge_marginal(const grid_edge &edge, int p, int q) {
               return edge_m_(edge.index(), p * num_labels_ + q);
            }
            
            /**
             * Get or set the log of the value of the partition function z
             */
            __host__ float log_z() const {
               return log_z_;
            }

            /**
             * Normalizes the node and edge marginal potentials using the
             * partition function z
             */
            __host__ void normalize() {
               using namespace inference_result_kernel;
               normalize_rows<<<DIVUP(node_m_.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(node_m_);
               normalize_rows<<<DIVUP(edge_m_.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(edge_m_);
            }

            /**
             * Returns a matrix of node marginals. Each row represents a node,
             * and each column is a label.
             */
            __host__ device_matrix<float> node_marginals() {
               return node_m_;
            }

            /**
             * Returns a matrix of edge marginals. Each row represents an edge,
             * and the label pair (p, q) is at column p * number of labels + q.
             * That is, the labels are stored in row major order (with p as row)
             */
            __host__ device_matrix<float> edge_marginals() {
               return edge_m_;
            }
         private:
            int num_labels_;
            device_matrix<float> node_m_;
            device_matrix<float> edge_m_;
            float log_z_;
      };
}
