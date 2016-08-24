#include "MatrixScaleKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

__global__
void matrixScaleKernel(CuMatrix target, float scale) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < target.rows && col < target.cols) {
    float* elem = Elem(target, row, col);
    *elem = *elem * scale;
  }
}

void MatrixScaleKernel::Apply(CuMatrix target, float scale, cudaStream_t stream) {
  int bpgX = (target.cols + TPB_X - 1) / TPB_X;
  int bpgY = (target.rows + TPB_Y - 1) / TPB_Y;

  matrixScaleKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), 0, stream>>>(target, scale);
}
