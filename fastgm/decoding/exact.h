#pragma once

#include "../grid.h"
#include "../potentials.h"

namespace fastgm {
   template <class T>
      class exact_decoding {
         public:
            template <class Grid>
            host_array<int> operator() (const Grid &g) {
               host_array<int> result(g.size(), 0);
               
               // Store the current labeling combination
               host_array<int> labeling(g.size(), 0);

               // Go through each possible combination of labels
               T max_pot = labeling_potential<T>(g, labeling);
               while (true) {
                  // Calculate the potential of this combination
                  T pot = labeling_potential<T>(g, labeling);
                  if (pot > max_pot) {
                     max_pot = pot;
                     result = labeling;
                  }

                  int i = 0;
                  for (; i < labeling.size(); i++) {
                     // Increment this node's label. If it goes over the number
                     // of possible labels, then we need to reset it to zero and
                     // increment the next label
                     if (++labeling[i] < g.num_labels()) break;
                     labeling[i] = 0;
                  }

                  // If the last element in the labeling is reset by to zero
                  // it means that we've gone through all the possible combinations
                  if (i == labeling.size() && labeling[i - 1] == 0) break;
               }

               return result;
            }
      };
}
