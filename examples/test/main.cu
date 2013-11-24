#include "grid.h"
#include "host_matrix.h"
#include "host_array.h"

#include "learning/likelihood.h"

#include <cstdio>

using namespace fastgm;

// Create a custom unary potential with 3 parameters
class binary_feature : public unary_param<float, float> {
   public:
      binary_feature() : unary_param<float, float>(3) { }
    
      __device__ void compute_features(device_array<float> f, int i, int x) const {
        // First term is the bias term, which is always 1
        f[0] = 1;
        
        // We use either the second or third parameter, depending on the value
        // of the label x
        if (x == 0) {
           f[1] = data()[i];
           f[2] = 0;
        } else {
           f[1] = 0;
           f[2] = data()[i];
        }
      }
};

typedef grid< binary_feature, potts<float, float> > grid_feature_t;

int main(int argc, char *argv[]) {
   grid_feature_t g(2, 2, 2, binary_feature(), fastgm::potts<float, float>(5));

   host_matrix<int> labelings(2, g.size());
   host_matrix<float> features(2, g.size());

   // Set up the labeling and features of the first training data
   labelings(0, 0) = 1; labelings(0, 1) = 1; labelings(0, 2) = 0; labelings(0, 3) = 1;
   features(0, 0) = 3; features(0, 1) = 5; features(0, 2) = 2; features(0, 3) = 4;

   // Set up the labeling and features of the second training data
   labelings(1, 0) = 1; labelings(1, 1) = 0; labelings(1, 2) = 0; labelings(1, 3) = 0;
   features(1, 0) = 5; features(1, 1) = 1; features(1, 2) = 1; features(1, 3) = 1;

   // Set up the parameters
   host_array<float> w(g.num_params(), 0);
   w[0] = 1; w[1] = 3; w[2] = 2; w[3] = 1;

   // Copy to GPU
   device_matrix<float> dfeatures(features);
   device_matrix<int> dlabelings(labelings);
   device_array<float> dw(w);
   device_array<float> dgrad(w.size(), true);

   // Set up the parameters
   crf_piecewise_nll(dw, g, dlabelings, dfeatures, dgrad);

   // Print out the gradients
   host_array<float> grad = dgrad.to_host();
   for (int i = 0; i < grad.size(); i++) {
      printf("%f ", grad[i]);
   }

   return 0;
}
