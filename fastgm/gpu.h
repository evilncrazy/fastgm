#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

// Shorthand for code running on both host and device
#define __host_device__ __host__ __device__

// Attempt to call a CUDA API function. If it fails, then the program exits with an error message
#define CUDA_SAFE_CALL(call) { cudaAssert(call, __FILE__, __LINE__); }

// Checks whether there were any CUDA errors, printing an error message and exiting if there is
#define CUDA_CHECK_ERRORS { cudaAssert(cudaGetLastError(), __FILE__, __LINE__); }

// Ceil of x divided by y
#define DIVUP(x, y) (int)ceil((float)(x) / (y))

/**
 * Stops execution of program if there was a CUDA error. Called by the 
 * CUDA_SAFE_CALL macro to stop the program when there's a CUDA API error.
 */
__host__ void cudaAssert(const cudaError err, const char *file, const int line) {
   if (cudaSuccess != err) {
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
            file, line, cudaGetErrorString(err) );
      exit(1);
   }
}

#include "device_malloc.h"
#include "device_free.h"
#include "device_memcpy.h"
