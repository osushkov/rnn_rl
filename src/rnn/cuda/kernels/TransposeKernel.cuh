#pragma once

#include "../Types.hpp"
#include <cuda_runtime.h>

namespace rnn {
namespace cuda {
namespace TransposeKernel {

void Apply(CuMatrix src, CuMatrix dst, cudaStream_t stream);
}
}
}
