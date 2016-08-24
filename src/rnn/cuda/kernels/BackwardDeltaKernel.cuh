#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace BackwardDeltaKernel {

void Apply(LayerBatchDeltas nextDelta, CuMatrix transposedWeights, ConnectionActivation connection,
           LayerBatchDeltas outDelta, cudaStream_t stream);
}
}
}
