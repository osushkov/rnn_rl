
#include "BackwardDeltaKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>
#include <cassert>

using namespace rnn;
using namespace rnn::cuda;

// computes outDelta = tw * nextDelta (elemwisemul) layerOutput.derivatives
__global__
void backwardDeltaKernel(LayerBatchDeltas nextDelta, CuMatrix tw, ConnectionActivation connection,
                         LayerBatchDeltas outDelta, unsigned spitch) {

  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  const int numChunks = (tw.cols + blockDim.x - 1) / blockDim.x;

  // buffer for holding the layer weight matrix chunk
  float *twChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *ndChunk = (float *) &buf[spitch * blockDim.y];

  const int twRow = blockDim.x * blockIdx.x + threadIdx.y;
  const int ndRow = row;

  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.x;

  float sum = 0.0f;
  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.x) {
    const int twCol = chunkOffset + threadIdx.x;
    if (twRow < tw.rows && twCol < tw.cols) {
      twChunk[chunkIndex] = *Elem(tw, twRow, twCol);
    }

    const int ndCol = twCol;
    if (ndRow < nextDelta.batchSize && ndCol < nextDelta.delta.cols) {
      ndChunk[chunkIndex] = *Elem(nextDelta.delta, ndRow, ndCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, tw.cols - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += twChunk[j + threadIdx.x * spitch] * ndChunk[j + threadIdx.y * spitch];
    }
    __syncthreads();
  }

  if (row < outDelta.batchSize && col < outDelta.delta.cols) {
    float od = *Elem(connection.derivative, row, col);
    *Elem(outDelta.delta, row, col) += sum * od;
  }
}

void BackwardDeltaKernel::Apply(LayerBatchDeltas nextDelta, CuMatrix transposedWeights,
                                ConnectionActivation connection, LayerBatchDeltas outDelta,
                                cudaStream_t stream) {

  assert(nextDelta.delta.cols == transposedWeights.cols);
  assert(outDelta.delta.cols == transposedWeights.rows - 1);
  assert(outDelta.delta.cols == connection.activation.cols - 1);
  assert(nextDelta.batchSize == connection.batchSize);
  assert(nextDelta.batchSize == outDelta.batchSize);

  int bpgX = (outDelta.delta.cols + TPB_X - 1) / TPB_X;
  int bpgY = (outDelta.batchSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = (TPB_X + 1);
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  backwardDeltaKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      nextDelta, transposedWeights, connection, outDelta, spitch);
}
