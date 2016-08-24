#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CheckError(ans)                                                                            \
  { rnn::cuda::util::OutputError((ans), __FILE__, __LINE__); }

namespace rnn {
namespace cuda {
namespace util {

inline void OutputError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "GPU error: " << cudaGetErrorString(code) << " "
        << file << "(" << line << ")" << std::endl;
    exit(code);
  }
}

}
}
}
