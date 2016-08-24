#pragma once

#include "MatrixView.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdlib>

typedef Eigen::VectorXf EVector;

// TODO: row major is required for cuda (the way Ive implemented it), but is slower(?) for cpu
// computation.
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EMatrix;

namespace math {

static inline MatrixView GetMatrixView(EMatrix &m) {
  MatrixView result;
  result.rows = m.rows();
  result.cols = m.cols();
  result.data = m.data();
  return result;
}

// Returns a uniformly distributed random number between 0 and 1.
static inline float UnitRand(void) { return rand() / (float)RAND_MAX; }

static inline float RandInterval(float s, float e) { return s + (e - s) * UnitRand(); }

static inline float GaussianSample(float mean, float sd) {
  // Taken from GSL Library Gaussian random distribution.
  float x, y, r2;

  do {
    // choose x,y in uniform square (-1,-1) to (+1,+1)
    x = RandInterval(-1.0f, 1.0f);
    y = RandInterval(-1.0f, 1.0f);

    // see if it is in the unit circle
    r2 = x * x + y * y;
  } while (r2 > 1.0f || r2 <= 0.0001f);

  // Box-Muller transform
  return mean + sd * y * sqrtf(-2.0f * logf(r2) / r2);
}

static inline EVector SoftmaxActivations(const EVector &in) {
  assert(in.rows() > 0);
  EVector result(in.rows());

  float maxVal = in(0);
  for (int r = 0; r < in.rows(); r++) {
    maxVal = fmax(maxVal, in(r));
  }

  float sum = 0.0f;
  for (int i = 0; i < in.rows(); i++) {
    result(i) = expf(in(i)-maxVal);
    sum += result(i);
  }

  for (int i = 0; i < result.rows(); i++) {
    result(i) /= sum;
  }

  return result;
}
}
