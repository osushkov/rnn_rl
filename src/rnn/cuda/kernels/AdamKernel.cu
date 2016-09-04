
#include "AdamKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

__global__
void updateMomentumAndRMS(CuMatrix gradient, CuMatrix momentum, CuMatrix rms,
                          const float beta1, const float beta2) {

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= gradient.rows || col >= gradient.cols) {
    return;
  }

  float g = *Elem(gradient, row, col);
  float m = *Elem(momentum, row, col);
  float r = *Elem(rms, row, col);

  *Elem(momentum, row, col) = m * beta1 + g * (1.0f - beta1);
  *Elem(rms, row, col) = r * beta2 + g * g * (1.0f - beta2);
}

__global__
void updateWeightsWithAdam(CuMatrix weights, CuMatrix momentum, CuMatrix rms,
                           const float beta1, const float beta2,
                           const float lr, const float epsilon) {

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= rms.rows || col >= rms.cols) {
    return;
  }

  float mc = *Elem(momentum, row, col) / (1.0f - beta1);
  float rc = *Elem(rms, row, col) / (1.0f - beta2);

  *Elem(weights, row, col) -= lr * mc / sqrtf(rc + epsilon);
}

void AdamKernel::UpdateMomentumAndRMS(const CuMatrix &gradient, const CuMatrix &momentum,
                                      const CuMatrix &rms, float beta1, float beta2,
                                      cudaStream_t stream) {

  assert(gradient.rows == momentum.rows);
  assert(gradient.cols == momentum.cols);
  assert(gradient.rows == rms.rows);
  assert(gradient.cols == rms.cols);

  int bpgX = (gradient.cols + TPB_X - 1) / TPB_X;
  int bpgY = (gradient.rows + TPB_Y - 1) / TPB_Y;

  updateMomentumAndRMS<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), 0, stream>>>(
      gradient, momentum, rms, beta1, beta2);
}

void AdamKernel::UpdateWeightsWithAdam(const CuMatrix &weights, const CuMatrix &momentum,
                                       const CuMatrix &rms, float beta1, float beta2, float lr,
                                       float epsilon, cudaStream_t stream) {

  assert(weights.rows == momentum.rows);
  assert(weights.cols == momentum.cols);
  assert(weights.rows == rms.rows);
  assert(weights.cols == rms.cols);

  int bpgX = (weights.cols + TPB_X - 1) / TPB_X;
  int bpgY = (weights.rows + TPB_Y - 1) / TPB_Y;

  updateWeightsWithAdam<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), 0, stream>>>(
      weights, momentum, rms, beta1, beta2, lr, epsilon);
}
