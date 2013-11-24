#pragma once 

#include "../device_array.h"
#include "../potentials.h"
#include "../inference/inference_result.h"

#include "likelihood.cu"

namespace fastgm {
   /**
    * Compute the log likelihood
    *
    * @param w Array of weights for all the parameters
    * @param g The grid
    * @param labelings A matrix containing a training instance in each row.
    *    Each training instance contains a valid label for each node.
    * @param features A matrix containing feature data associated with each
    *    training instance and each node.
    * @param inf An inference algorithm that computes node and edge marginals
    * @param grad Output array that contains the computed gradients for each
    *    of the corresponding parameters
    */
   template <class Grid, class FeatureData, class InferenceMethod>
      float crf_nll(const device_array<float> &w,
            Grid g, const device_matrix<int> &labelings,
            const device_matrix<FeatureData> &features,
            InferenceMethod &inf, device_array<float> &grad) {
         using namespace likelihood_kernel;
         float nll = 0;

         // Reset gradient vector to 0
         grad.reset();
         
         // Compute the required grid and block sizes
         dim3 grid_size(DIVUP(g.width(), BLOCK_WIDTH), DIVUP(g.height(), BLOCK_HEIGHT), 1);
         dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT, 1);

         // Stores the result of inference
         inference_result result(g.size(), g.num_edges(), g.num_labels());
         
         // Temporary feature vectors
         device_matrix<float> uf(g.size(), g.unary().num_params());
         device_matrix<float> pf(g.num_edges(), g.pairwise().num_params());

         // Each row contains gradients calculated by each node. The actual gradient
         // is computed by summing along each row.
         device_matrix<float> ugrad(g.unary().num_params(), g.size());
         device_matrix<float> pgrad(g.pairwise().num_params(), g.num_edges());

         // For each training data
         for (int t = 0; t < labelings.rows(); t++) {
            device_array<int> labeling(labelings.row(t));

            // If the training data also has features, then we need to update
            // the underlying CRF with these features
            if (t < features.rows()) g.set_data(features.row(t));
            
            // Run inference on it to get the partition function z
            inf(g, w, result);
            
            // Update likelihood
            nll = nll - log_labeling_potential(g, w, labeling) + result.log_z();
            
            // Compute gradients
            compute_node_gradient<<<DIVUP(g.size(), BLOCK_SIZE), BLOCK_SIZE>>>(g, ugrad, result.node_marginals(), uf, labeling);
            compute_edge_gradient<<<DIVUP(g.num_edges(), BLOCK_SIZE), BLOCK_SIZE>>>(g, pgrad, result.edge_marginals(), pf, labeling);
            
            sum_cols<<<DIVUP(ugrad.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(ugrad, 0, grad);
            sum_cols<<<DIVUP(pgrad.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(pgrad, g.unary().num_params(), grad);
         }

         // Clean up resources
         CUDA_CHECK_ERRORS;
         
         uf.release(); ugrad.release(); pf.release(); pgrad.release();
         return nll;
      }

   /**
    * Compute the pseudo-negative log likelihood
    *
    * @param w Array of weights for all the parameters
    * @param g The grid
    * @param labelings A matrix containing a training instance in each row.
    *    Each training instance contains a valid label for each node.
    * @param features A matrix containing feature data associated with each
    *    training instance and each node.
    * @param grad Output array that contains the computed gradients for each
    *    of the corresponding parameters
    */
   template <class Grid, class FeatureData>
      float crf_pseudo_nll(const device_array<float> &w,
            Grid g, const device_matrix<int> &labelings,
            const device_matrix<FeatureData> &features,
            device_array<float> &grad) {
         using namespace likelihood_kernel;
         float nll = 0;

         // Reset gradient vector to 0
         grad.reset();
         
         // Compute the required grid and block sizes
         dim3 grid_size(DIVUP(g.width(), BLOCK_WIDTH), DIVUP(g.height(), BLOCK_HEIGHT), 1);
         dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
         
         // Matrix of local probabilities at each node for each label
         device_matrix<float> pot(g.size(), g.num_labels(), true);
         
         // Temporary feature vectors
         device_matrix<float> uf(g.size(), g.unary().num_params());
         device_matrix<float> pf(g.num_edges(), g.pairwise().num_params());

         // Each row contains gradients calculated by each node. The actual gradient
         // is computed by summing along each row.
         device_matrix<float> ugrad(g.unary().num_params(), g.size());
         device_matrix<float> pgrad(g.pairwise().num_params(), g.num_edges());

         // For each training data
         for (int t = 0; t < labelings.rows(); t++) {
            device_array<int> labeling(labelings.row(t));

            // If the training data also has features, then we need to update
            // the underlying CRF with these features
            if (t < features.rows()) g.set_data(features.row(t));

            // Compute the local pseudo probabilities and update gradients
            pseudo_prob<<<grid_size, block_size>>>(g, uf, pf, w, labeling, pot);

            // TODO: nll = nll - safe_log(pot[i][labeling[i]]);

            compute_node_gradient<<<DIVUP(g.size(), BLOCK_SIZE), BLOCK_SIZE>>>(g, ugrad, pot, uf, labeling);
            compute_edge_gradient_pseudo<<<DIVUP(g.num_edges(), BLOCK_SIZE), BLOCK_SIZE>>>(g, pgrad, pot, pf, labeling);
            
            sum_cols<<<DIVUP(ugrad.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(ugrad, 0, grad);
            sum_cols<<<DIVUP(pgrad.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(pgrad, g.unary().num_params(), grad);
         }
         
         // Clean up resources
         CUDA_CHECK_ERRORS;
         
         pot.release();
         uf.release(); ugrad.release(); pf.release(); pgrad.release();

         return nll;
      }

   template <class Grid, class FeatureData>
      float crf_piecewise_nll(const device_array<float> &w,
            Grid g, const device_matrix<int> &labelings,
            const device_matrix<FeatureData> &features,
            device_array<float> &grad) {
         using namespace likelihood_kernel;
         float nll = 0;

         // Reset gradient vector to 0
         grad.reset();
         
         // Compute the required grid and block sizes
         dim3 grid_size(DIVUP(g.width(), BLOCK_WIDTH), DIVUP(g.height(), BLOCK_HEIGHT), 1);
         dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
         
         // Matrix of unary and pairwise probabilities
         device_matrix<float> upot(g.size(), g.num_labels());
         device_matrix<float> ppot(g.num_edges(), g.num_labels() * g.num_labels());
         
         // Temporary feature vectors
         device_matrix<float> uf(g.size(), g.unary().num_params());
         device_matrix<float> pf(g.num_edges(), g.pairwise().num_params());

         // Each row contains gradients calculated by each node. The actual gradient
         // is computed by summing along each row.
         device_matrix<float> ugrad(g.unary().num_params(), g.size());
         device_matrix<float> pgrad(g.pairwise().num_params(), g.num_edges());

         // For each training data
         for (int t = 0; t < labelings.rows(); t++) {
            device_array<int> labeling(labelings.row(t));

            // If the training data also has features, then we need to update
            // the underlying CRF with these features
            if (t < features.rows()) g.set_data(features.row(t));
            
            // Calculate unary probabilities
            unary_prob<<<DIVUP(upot.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(g, uf, w, upot);
            
            // Calculate pairwise probabilities
            pairwise_prob<<<DIVUP(ppot.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(g, pf, w, ppot);

            // Compute gradients
            compute_node_gradient<<<DIVUP(g.size(), BLOCK_SIZE), BLOCK_SIZE>>>(g, ugrad, upot, uf, labeling);
            compute_edge_gradient<<<DIVUP(g.num_edges(), BLOCK_SIZE), BLOCK_SIZE>>>(g, pgrad, ppot, pf, labeling);
            
            sum_cols<<<DIVUP(ugrad.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(ugrad, 0, grad);
            sum_cols<<<DIVUP(pgrad.rows(), BLOCK_SIZE), BLOCK_SIZE>>>(pgrad, g.unary().num_params(), grad);
         }

         // Clean up resources
         CUDA_CHECK_ERRORS;
         
         upot.release(); ppot.release();
         uf.release(); ugrad.release(); pf.release(); pgrad.release();

         return nll;
      }
}
