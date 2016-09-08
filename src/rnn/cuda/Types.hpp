
#pragma once

#include <cassert>
#include <cstddef>

namespace rnn {
namespace cuda {

struct CuMatrix {
  unsigned rows;
  unsigned cols;

  // Data pointers allocated with cudaMallocPitch. Logical size is (rows * cols).
  // Elements are stored in row major format (inner loop traverses across a given row).
  float *data;

  // The pitch of the rows of the weights matrix in bytes.
  size_t pitch;
};

struct ConnectionActivation {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.

  CuMatrix activation;
  CuMatrix derivative;

  ConnectionActivation() = default;
  ConnectionActivation(unsigned batchSize, CuMatrix activation, CuMatrix derivative)
      : batchSize(batchSize), activation(activation), derivative(derivative) {
    assert(batchSize > 0);
    assert(batchSize <= activation.rows);
    assert(activation.rows == derivative.rows && activation.cols == derivative.cols);
  }
};

struct LayerBatchDeltas {
  unsigned batchSize; // equal to the number of rows in the matrix actually used.
  CuMatrix delta;

  LayerBatchDeltas() = default;
  LayerBatchDeltas(unsigned batchSize, CuMatrix delta) : batchSize(batchSize), delta(delta) {
    assert(batchSize > 0);
    assert(batchSize <= delta.rows);
  }
};
}
}
