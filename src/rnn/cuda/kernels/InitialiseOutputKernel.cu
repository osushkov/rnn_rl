#include "InitialiseOutputKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

__global__
void initialiseOutputKernel(CuMatrix activation) {
  const unsigned row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < activation.rows) {
    *Elem(activation, row, activation.cols - 1) = 1.0f;
  }
}

void InitialiseOutputKernel::Apply(CuMatrix activation, cudaStream_t stream) {
  int bpgX = (activation.rows + TPB_X - 1) / TPB_X;
  initialiseOutputKernel<<<bpgX, TPB_X, 0, stream>>>(activation);
}
