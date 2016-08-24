#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace GradientIncrementKernel {

void Apply(LayerBatchDeltas layerDeltas, ConnectionActivation connection, CuMatrix outGradient,
           cudaStream_t stream);
}
}
}
