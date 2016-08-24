#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace WeightedIncrementKernel {

void Apply(CuMatrix layerWeights, ConnectionActivation input, CuMatrix output, cudaStream_t stream);
}
}
}
