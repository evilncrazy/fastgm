#pragma once

#include "edge.h"
#include "unary_potential.h"
#include "pairwise_potential.h"

namespace fastgm {
   template <class U, class P>
      class grid {
         public:
            typedef grid<U, P> grid_type;
            typedef U unary_type;
            typedef P pairwise_type;

            /**
             * Construct a grid-based four-connected Markov random field,
             * with unary potentials at each node, and pairwise potentials
             * between each pair of adjacent nodes
             *
             * @param width The width of the grid
             * @param height The height of the grid
             * @param nlabels The size of the label space for each node
             * @param pairwise The pairwise potential function, defaults to a
             *    constant function that evaluates to one 
             */
            __host__ grid(int width, int height, int nlabels, U unary, P pairwise) :
               width_(width), height_(height), nlabels_(nlabels),
               unary_(unary), pairwise_(pairwise) {
               // Generate list of edges and store them on the device
               host_array<grid_edge> edges(num_edges());
               for (int i = 0; i < size(); i++) {
                  for (int dir = 0; dir < 4; dir++) {
                    grid_edge edge = edge_from(i, static_cast<Direction>(dir));
                    if (edge.valid()) edges[edge.index()] = edge;
                  }
               }
               
               edges_ = device_array<grid_edge>(edges);
            }
            
            /**
             * Get the width of the grid
             */
            __host_device__ int width() const { return width_; }

            /**
             * Get the height of the grid
             */
            __host_device__ int height() const { return height_; }

            /**
             * Get the total number of nodes in the grid
             */
            __host_device__ int size() const { return width() * height(); }

            /**
             * Get the size of the label space
             */
            __host_device__ int num_labels() const { return nlabels_; }

            /**
             * Get the number of edges in the grid
             */
            __host_device__ int num_edges() const {
               return (width_ - 1) * height_ + (height_ - 1) * width_;
            }

            /**
             * Return a reference to the unary potential
             */
            __host_device__ U &unary() { return unary_; }
            __host_device__ const U &unary() const { return unary_; }
            
            /**
             * Return a reference to the pairwise potential
             */
            __host_device__ P &pairwise() { return pairwise_; }
            __host_device__ const P &pairwise() const { return pairwise_; }

            /**
             * Return the number of parameters used by this grid
             */
            int num_params() const {
               return unary_.num_params() + pairwise_.num_params();
            }
            
            /**
             * Evaluate the unary potential function.
             *
             * @param f A device array to temporarily hold the computed feature values.
             *    Must be exclusively used by a single thread to avoid data races.
             * @param w An array of weights for all the parameters
             *    (including pairwise weights)
             * @param i The node to evaluate the potential function on
             * @param x The label given to the node
             */
            __device__ float operator() (device_array<float> f, device_array<float> w, int i, int x) const {
               float result = 0;
               unary_.compute_features(f, i, x);
               for (int i = 0; i < f.size(); i++) {
                  result += w[i] * f[i];
               }
               return result;
            }
            
            /**
             * Evaluate the pairwise potential function.
             *
             * @param f A device array to temporarily hold the computed feature values.
             *    Must be exclusively used by a single thread to avoid data races.
             * @param w An array of weights for all the parameters
             *    (including unary weights)
             * @param edge The edge to evaluate the potential function on
             * @param p The label given to one of the endpoints
             * @param q The label given to the other endpoint
             */
            __device__ float operator() (device_array<float> f, device_array<float> w, grid_edge edge, int p, int q) const {
               float result = 0;
               pairwise_.compute_features(f, edge, p, q);
               for (int i = 0; i < f.size(); i++) {
                  result += w[i + unary_.num_params()] * f[i];
               }
               return result;
            }
            
            /**
             * Return the index of the edge between two nodes. No checking is
             * done on whether there actually is an edge between the nodes.
             */
            __host_device__ int edge_index(int i, int j) const {
               if (j - i == 1)
                 return (i / width()) * (width() - 1) + i % width();
               else
                 return i + (width() - 1) * height();
            }
            
            /**
             * Return a grid_edge object representing the edge between two nodes.
             * No checking is done on whether there actually is an edge between the nodes.
             */
            __host_device__ grid_edge edge_between(int i, int j) const {
               return grid_edge(i, j, edge_index(i, j));
            }

            /**
             * Return a grid_edge object representing the edge from a node in
             * a particular direction. Returns an invalid edge object if there's
             * not such edge.
             */
            __host_device__ grid_edge edge_from(int i, Direction d) const {
               if (d == DirUp && i >= width())
                  return edge_between(i - width(), i);
               if (d == DirDown && i < width() * (height() - 1))
                  return edge_between(i, i + width());
               if (d == DirLeft && i % width() != 0)
                  return edge_between(i - 1, i);
               if (d == DirRight && i % width() != width() - 1)
                  return edge_between(i, i + 1);
               return grid_edge(0, 0, -1);
            }
            
            /**
             * Set the feature data of the unary and pairwise potentials
             */
            template <class F>
            __host__ void set_data(const device_array<F> &data) {
            	unary().set_data(data);
            	pairwise().set_data(data);
            }
            
            /**
             * Return the edge object for the ith edge
             */
            __device__ grid_edge edge(int i) const {
            	return edges_[i];
            }

         private:
            int width_, height_; // Dimensions of the grid
            int nlabels_; // Number of labels for each node

            U unary_; P pairwise_; // Potential functions
            device_array<grid_edge> edges_; // Contains list of all the edges
      };
}
