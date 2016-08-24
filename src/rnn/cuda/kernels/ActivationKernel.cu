#include "ActivationKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

__device__ float activationValue(float in, const LayerActivation activation) {
  switch(activation) {
  case LayerActivation::TANH:
    return tanhf(in);
  case LayerActivation::LOGISTIC:
    return 1.0f / (1.0f + expf(-in));
  case LayerActivation::RELU:
    return fmaxf(0.0f, in);
  case LayerActivation::LEAKY_RELU:
    return fmaxf(0.01f * in, in);
  case LayerActivation::ELU:
    return in > 0.0f ? in : (expf(in) - 1.0f);
  case LayerActivation::LINEAR:
  case LayerActivation::SOFTMAX:
    return in;
  }
  assert(false); // should never get here.
  return in;
}

__device__ float activationDerivative(float in, float out, const LayerActivation activation) {
  switch(activation) {
  case LayerActivation::TANH:
    return 1.0f - out * out;
  case LayerActivation::LOGISTIC:
    return out * (1.0f - out);
  case LayerActivation::RELU:
    return in > 0.0f ? 1.0f : 0.0f;
  case LayerActivation::LEAKY_RELU:
    return in > 0.0f ? 1.0f : 0.01f;
  case LayerActivation::ELU:
    return in > 0.0f ? 1.0f : (out + 1.0f);
  case LayerActivation::LINEAR:
  case LayerActivation::SOFTMAX:
    return 1.0f;
  }
  assert(false); // should never get here.
  return 1.0f;
}

__global__
void activationKernel(ConnectionActivation layer, LayerActivation activation) {
  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < layer.batchSize && col < layer.activation.cols - 1) {
    float *aElem = Elem(layer.activation, row, col);
    float *dElem = Elem(layer.derivative, row, col);

    float in = *aElem;
    float av = activationValue(in, activation);

    *aElem = av;
    *dElem = activationDerivative(in, av, activation);
  }
}

void ActivationKernel::Apply(const ConnectionActivation &layer, const LayerActivation &activation,
                             cudaStream_t stream) {

  assert(layer.activation.rows == layer.derivative.rows);
  assert(layer.activation.cols == layer.derivative.cols);
  assert(layer.batchSize <= layer.activation.rows);

  // -1 is here since we dont need to compute the bias term for the output vector.
  int bpgX = (layer.activation.cols - 1 + TPB_X - 1) / TPB_X;
  int bpgY = (layer.batchSize + TPB_Y - 1) / TPB_Y;

  activationKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), 0, stream>>>(layer, activation);
}
