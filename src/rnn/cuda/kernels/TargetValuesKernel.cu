
#include "TargetValuesKernel.hpp"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

__global__
void targetValuesKernel(CuMatrix nextTargetActivation, CuMatrix batchRewards,
                        float discountFactor, CuMatrix outTargetValue) {

  const unsigned batchIndex = blockIdx.x;

  float maxVal = *Elem(nextTargetActivation, batchIndex, 0);
  for (unsigned i = 1; i < nextTargetActivation.cols - 1; i++) {
    maxVal = fmaxf(maxVal, *Elem(nextTargetActivation, batchIndex, i)));
  }

  float target = *Elem(batchRewards, batchIndex, 0) + discountFactor * maxVal;

  // TODO: this is unnecessary as we only need to set the target on one output connection,
  // corresponding to the action actually performed.
  for (unsigned i = 0; i < outTargetValue.cols; i++) {
    *Elem(outTargetValue, batchIndex, i) = target;
  }
}

void TargetValuesKernel::Apply(CuMatrix nextTargetActivation, CuMatrix batchRewards,
                               float discountFactor, CuMatrix outTargetValue, cudaStream_t stream) {

  assert(nextTargetActivation.activation.cols == outTargetValue.cols);
  assert(nextTargetActivation.activation.rows == outTargetValue.rows);
  assert(batchRewards.cols == 1);
  assert(batchRewards.rows == outTargetValue.rows);
  assert(discountFactor > 0.0f && discountFactor <= 1.0f);

  int tpb = 1;
  int bpg = outTargetValue.rows;

  targetValuesKernel<<<bpg, tpb, 0, stream>>>(
      nextTargetActivation, batchRewards, discountFactor, outTargetValue);
}
