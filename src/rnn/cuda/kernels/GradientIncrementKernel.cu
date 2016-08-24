#include "GradientIncrementKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

__global__
void gradientIncrementKernel(LayerBatchDeltas layerDeltas, ConnectionActivation connection,
                             CuMatrix outGradient, unsigned spitch) {

  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  // buffer for holding the layer weight matrix chunk
  float *ldChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *connChunk = (float *) &buf[spitch * blockDim.y];

  const int dCol = blockDim.y * blockIdx.y + threadIdx.x;
  const int cCol = col;

  const int numChunks = (layerDeltas.batchSize + blockDim.y - 1) / blockDim.y;
  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.y;

  float sum = 0.0f;
  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.y) {
    const int dRow = chunkOffset + threadIdx.y;
    if (dRow < layerDeltas.batchSize && dCol < layerDeltas.delta.cols) {
      ldChunk[chunkIndex] = *Elem(layerDeltas.delta, dRow, dCol);
    }

    const int cRow = dRow;
    if (cRow < connection.batchSize && cCol < connection.activation.cols) {
      connChunk[chunkIndex] = *Elem(connection.activation, cRow, cCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, layerDeltas.batchSize - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += ldChunk[threadIdx.y + j * spitch] * connChunk[threadIdx.x + j * spitch];
    }

    __syncthreads();
  }

  if (row < outGradient.rows && col < outGradient.cols) {
    *Elem(outGradient, row, col) += sum;
  }
}

void GradientIncrementKernel::Apply(LayerBatchDeltas layerDeltas, ConnectionActivation connection,
                                    CuMatrix outGradient, cudaStream_t stream) {

  assert(layerDeltas.batchSize == connection.batchSize);
  assert(layerDeltas.delta.cols == outGradient.rows);
  assert(connection.activation.cols == outGradient.cols);

  int bpgX = (outGradient.cols + TPB_X - 1) / TPB_X;
  int bpgY = (outGradient.rows + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  gradientIncrementKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerDeltas, connection, outGradient, spitch);
}
