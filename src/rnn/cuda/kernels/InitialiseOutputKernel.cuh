#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace InitialiseOutputKernel {

void Apply(CuMatrix activation, cudaStream_t stream);
}
}
}
