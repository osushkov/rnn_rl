#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace TargetValuesKernel {

void Apply(CuMatrix nextTargetActivation, CuMatrix nextActionMask, CuMatrix batchRewards,
           float discountFactor, bool useOnlyReward, CuMatrix outTargetValue, cudaStream_t stream);
}
}
}
