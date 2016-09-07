
#include "ErrorMeasureKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

using namespace rnn;
using namespace rnn::cuda;

__global__
void errorMeasureKernel(ConnectionActivation nnOut, TargetOutput target, CuMatrix deltaMask,
                        LayerBatchDeltas out) {

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= out.batchSize || col >= out.delta.cols) {
    return;
  }

  float mask = *Elem(deltaMask, row, col);
  float derivative = 1.0f;//*Elem(nnOut.derivative, row, col);
  *Elem(out.delta, row, col) =
      mask * derivative * (*Elem(nnOut.activation, row, col) - *Elem(target.value, row, col));
  // printf("%d delta: %f ** %f\n", col, *Elem(out.delta, row, col), *Elem(target.value, row, col));
}

void ErrorMeasureKernel::Apply(ConnectionActivation networkOutput, TargetOutput targetOutput,
                               CuMatrix deltaMask, LayerBatchDeltas out, cudaStream_t stream) {

  assert(networkOutput.activation.cols == targetOutput.value.cols + 1);
  assert(out.delta.cols == targetOutput.value.cols);
  assert(deltaMask.rows == out.delta.rows);
  assert(deltaMask.cols == out.delta.cols);

  int bpgX = (out.delta.cols + TPB_X - 1) / TPB_X;
  int bpgY = (out.batchSize + TPB_Y - 1) / TPB_Y;

  errorMeasureKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), 0, stream>>>(
      networkOutput, targetOutput, deltaMask, out);
      // getchar();
}
