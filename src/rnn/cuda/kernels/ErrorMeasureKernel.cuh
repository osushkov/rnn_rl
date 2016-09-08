#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace ErrorMeasureKernel {

void Apply(ConnectionActivation networkOutput, CuMatrix targetOutput, CuMatrix deltaMask,
           LayerBatchDeltas out, cudaStream_t stream);
}
}
}
