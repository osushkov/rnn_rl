#include "MatrixFillKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

__global__
void matrixFillKernel(CuMatrix target, float value) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < target.rows && col < target.cols) {
    *Elem(target, row, col) = value;
  }
}

void MatrixFillKernel::Apply(const CuMatrix &target, float value, cudaStream_t stream) {
  int bpgX = (target.cols + TPB_X - 1) / TPB_X;
  int bpgY = (target.rows + TPB_Y - 1) / TPB_Y;

  matrixFillKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), 0, stream>>>(target, value);
}
