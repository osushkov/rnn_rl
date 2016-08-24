#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace MatrixScaleKernel {

void Apply(CuMatrix target, float scale, cudaStream_t stream);
}
}
}
