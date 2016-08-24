#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace SoftmaxKernel {

void Apply(ConnectionActivation lastLayer, cudaStream_t stream);
}
}
}
