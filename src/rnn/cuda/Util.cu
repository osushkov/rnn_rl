
#include "Util.hpp"
#include "Util.cuh"
#include "../../math/MatrixView.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

using namespace rnn;
using namespace rnn::cuda;

static unsigned totalBytesAllocated = 0;

void util::CudaSynchronize(void) {
  cudaDeviceSynchronize();
}

void *util::AllocPinned(size_t bufSize) {
  void* result = nullptr;

  cudaError_t err = cudaHostAlloc(&result, bufSize, cudaHostAllocDefault);
  CheckError(err);
  assert(result != nullptr);

  return result;
}

void util::FreePinned(void *buf) {
  assert(buf != nullptr);
  cudaError_t err = cudaFreeHost(buf);
  CheckError(err);
}

CuMatrix util::AllocMatrix(unsigned rows, unsigned cols) {
  CuMatrix result;

  result.data = nullptr;
  result.rows = rows;
  result.cols = cols;

  size_t width = cols * sizeof(float);
  size_t height = rows;

  cudaError_t err = cudaMallocPitch(&(result.data), &(result.pitch), width, height);
  CheckError(err);
  assert(result.data != nullptr);

  totalBytesAllocated += result.pitch * height;
  //std::cout << "total matrix mem allocated: " << totalBytesAllocated << std::endl;

  return result;
}

void util::FreeMatrix(CuMatrix &m) {
  assert(m.data != nullptr);
  cudaError_t err = cudaFree(m.data);
  CheckError(err);
  m.data = nullptr;
}

static void printMatrixView(math::MatrixView view) {
  for (unsigned r = 0; r < view.rows; r++) {
    for(unsigned c = 0; c < view.cols; c++) {
      std::cout << view.data[c + r * view.cols] << " ";
    }
    std::cout << std::endl;
  }
}

void util::PrintMatrix(const CuMatrix &matrix) {
  math::MatrixView view;
  view.rows = matrix.rows;
  view.cols = matrix.cols;
  view.data = new float[view.rows * view.cols];

  cudaError_t err = cudaMemcpy2D(
      view.data, view.cols * sizeof(float),
      matrix.data, matrix.pitch,
      view.cols * sizeof(float), view.rows,
      cudaMemcpyDeviceToHost);

  CheckError(err);

  printMatrixView(view);
  delete[] view.data;
}
