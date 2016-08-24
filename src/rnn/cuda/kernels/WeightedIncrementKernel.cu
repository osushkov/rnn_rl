#include "WeightedIncrementKernel.cuh"
#include "Constants.hpp"
#include "../Types.cuh"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

__global__
void weightedIncrementKernel(CuMatrix lw, ConnectionActivation input, CuMatrix output,
                             const unsigned spitch) {

  extern __shared__ float buf[]; // shared memory buffer

  const unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned col = blockDim.x * blockIdx.x + threadIdx.x;

  const int numChunks = (lw.cols + blockDim.x - 1) / blockDim.x;

  // buffer for holding the layer weight matrix chunk
  float *lwChunk = (float *) buf;

  // buffer for holding the prev outputs matrix chunk
  float *inChunk = (float *) &buf[spitch * blockDim.y];

  const int lwRow = blockDim.x * blockIdx.x + threadIdx.y;
  const int inRow = row;

  const int chunkIndex = threadIdx.x + threadIdx.y * spitch;
  const int lim = numChunks * blockDim.x;

  float sum = 0.0f;
  for (int chunkOffset = 0; chunkOffset < lim; chunkOffset += blockDim.x) {
    const int lwCol = chunkOffset + threadIdx.x;
    if (lwRow < lw.rows && lwCol < lw.cols) {
      lwChunk[chunkIndex] = *Elem(lw, lwRow, lwCol);
    }

    const int inCol = lwCol;
    if (inRow < input.batchSize && inCol < input.activation.cols) {
      inChunk[chunkIndex] = *Elem(input.activation, inRow, inCol);
    }
    __syncthreads();

    int chunkLim = min(blockDim.x, lw.cols - chunkOffset);
    for (int j = 0; j < chunkLim; j++) {
      sum += lwChunk[j + threadIdx.x * spitch] * inChunk[j + threadIdx.y * spitch];
    }
    __syncthreads();
  }

  if (row < input.batchSize && col < output.cols - 1) {
    float *outElem = Elem(output, row, col);
    *outElem += sum;
  }
}

void WeightedIncrementKernel::Apply(CuMatrix layerWeights, ConnectionActivation input,
                                    CuMatrix output, cudaStream_t stream) {
  assert(layerWeights.cols == input.activation.cols);
  assert(layerWeights.rows == output.cols - 1);
  assert(input.batchSize <= output.rows);

  // -1 is here since we dont need to compute the bias term for the output vector.
  int bpgX = (output.cols - 1 + TPB_X - 1) / TPB_X;
  int bpgY = (input.batchSize + TPB_Y - 1) / TPB_Y;

  unsigned spitch = TPB_X + 1;
  size_t sharedMemSize = 2 * spitch * TPB_Y * sizeof(float);

  weightedIncrementKernel<<<dim3(bpgX, bpgY, 1), dim3(TPB_X, TPB_Y, 1), sharedMemSize, stream>>>(
      layerWeights, input, output, spitch);
}
