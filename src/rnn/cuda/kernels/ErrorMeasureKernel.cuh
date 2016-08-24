#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace ErrorMeasureKernel {

void Apply(ConnectionActivation networkOutput, TargetOutput targetOutput, LayerBatchDeltas out,
           cudaStream_t stream);
}
}
}
