#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace AdamKernel {

void UpdateMomentumAndRMS(const CuMatrix &gradient, const CuMatrix &momentum, const CuMatrix &rms,
                          float beta1, float beta2, cudaStream_t stream);

void UpdateWeightsWithAdam(const CuMatrix &weights, const CuMatrix &momentum, const CuMatrix &rms,
                           float beta1, float beta2, float lr, float epsilon,
                           cudaStream_t stream);
}
}
}
