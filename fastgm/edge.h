#pragma once

#include "gpu.h"

namespace fastgm {
   /**
    * Represents the four orthogonal directions
    */
   enum Direction {
      DirRight, DirDown, DirLeft, DirUp
   };

   /**
    * Represents an edge in the grid.
    */
   class grid_edge {
      public:
         /**
          * Construct a default, invalid edge.
          */
         __host_device__ grid_edge() : first_(0), second_(0), index_(-1) { }
         
         /**
          * Construct an edge instance between two nodes. The parameters must be
          * ordered so that there's only one correct representation of an edge.
          *
          * @param first The index of the first endpoint (must be smaller than index
          *    of second)
          * @param second The index of the second endpoint (must be greater than index
          *    of first)
          * @param index The index of this edge
          */
         __host_device__ grid_edge(int first, int second, int index) :
            first_(first), second_(second), index_(index) { }

         /**
          * Returns the index of the first endpoint.
          */
         __host_device__ int first() const { return first_; }

         /**
          * Returns the index of the seond endpoint.
          */
         __host_device__ int second() const { return second_; }

         /**
          * Returns the index of this edge
          */
         __host_device__ int index() const { return index_; }
         
         /**
          * Returns false if this edge is not a valid edge
          */
         bool valid() const { return index_ >= 0; }
      private:
         int first_, second_, index_;
   };
}
