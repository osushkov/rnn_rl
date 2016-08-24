
#include "TransposeKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

__global__
void transposeKernel(CuMatrix src, CuMatrix dst, const unsigned spitch) {

  extern __shared__ float buf[];

  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < src.cols && y < src.rows) {
    buf[threadIdx.x + threadIdx.y * spitch] = *Elem(src, y, x);
  }

  __syncthreads();

  x = blockIdx.y * blockDim.y + threadIdx.x;  // transpose block offset
  y = blockIdx.x * blockDim.x + threadIdx.y;

  if (x < dst.cols && y < dst.rows) {
    *Elem(dst, y, x) = buf[threadIdx.y + threadIdx.x * spitch];
  }
}

void TransposeKernel::Apply(CuMatrix src, CuMatrix dst, cudaStream_t stream) {
  assert(dst.cols >= src.rows);
  assert(dst.rows >= src.cols);

  int bpgX = (src.cols + TPB_X - 1) / TPB_X;
  int bpgY = (src.rows + TPB_Y - 1) / TPB_Y;

  unsigned spitch = TPB_X + 1;
  size_t sharedMemSize = spitch * TPB_Y * sizeof(float);

  transposeKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      src, dst, spitch);
}
