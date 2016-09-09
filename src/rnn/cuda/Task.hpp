#pragma once

#include "../../math/MatrixView.hpp"
#include "../LayerDef.hpp"
#include "Types.hpp"

namespace rnn {
namespace cuda {

enum class TaskType {
  LAYER_ACTIVATION,
  ERROR_MEASURE,
  PROPAGATE_DELTA,
  GRADIENT_INCREMENT,
  FILL_MATRIX,
  SCALE_MATRIX,
  TRANSPOSE_MATRIX,
  FORWARD_INCREMENT,
  TARGET_QVALUES,
  ADAM_UPDATE,
  ADAM_INCREMENT,
  COPY_MATRIX_D2H,
  COPY_MATRIX_H2D,
  COPY_MATRIX_D2D,
};

struct LayerActivationData {
  ConnectionActivation layer;
  LayerActivation activation;

  LayerActivationData() = default;
  LayerActivationData(ConnectionActivation layer, LayerActivation activation)
      : layer(layer), activation(activation) {}
};

struct ErrorMeasureData {
  ConnectionActivation networkOutput;
  TargetOutput targetOutput;
  CuMatrix deltaMask;
  LayerBatchDeltas outputLayer;

  ErrorMeasureData() = default;
  ErrorMeasureData(ConnectionActivation networkOutput, TargetOutput targetOutput,
                   CuMatrix deltaMask, LayerBatchDeltas outputLayer)
      : networkOutput(networkOutput), targetOutput(targetOutput), deltaMask(deltaMask),
        outputLayer(outputLayer) {}
};

struct PropagateDeltaData {
  LayerBatchDeltas nextDelta;
  CuMatrix transposedWeights;
  ConnectionActivation connection;
  LayerBatchDeltas outDelta;

  PropagateDeltaData() = default;
  PropagateDeltaData(LayerBatchDeltas nextDelta, CuMatrix transposedWeights,
                     ConnectionActivation connection, LayerBatchDeltas outDelta)
      : nextDelta(nextDelta), transposedWeights(transposedWeights), connection(connection),
        outDelta(outDelta) {}
};

struct GradientIncrementData {
  LayerBatchDeltas layerDeltas;
  ConnectionActivation connection;
  CuMatrix outGradient;

  GradientIncrementData() = default;
  GradientIncrementData(LayerBatchDeltas layerDeltas, ConnectionActivation connection,
                        CuMatrix outGradient)
      : layerDeltas(layerDeltas), connection(connection), outGradient(outGradient) {}
};

struct FillMatrixData {
  CuMatrix target;
  float value;

  FillMatrixData() = default;
  FillMatrixData(CuMatrix target, float value) : target(target), value(value) {
    assert(target.data != nullptr);
  }
};

struct ScaleMatrixData {
  CuMatrix target;
  float scale;

  ScaleMatrixData() = default;
  ScaleMatrixData(CuMatrix target, float scale) : target(target), scale(scale) {
    assert(target.data != nullptr);
  }
};

struct TransposeMatrixData {
  CuMatrix src;
  CuMatrix dst;

  TransposeMatrixData() = default;
  TransposeMatrixData(CuMatrix src, CuMatrix dst) : src(src), dst(dst) {
    assert(dst.cols >= src.rows);
    assert(dst.rows >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct ForwardIncrementData {
  CuMatrix layerWeights;
  ConnectionActivation input;
  CuMatrix output;

  ForwardIncrementData() = default;
  ForwardIncrementData(CuMatrix layerWeights, ConnectionActivation input, CuMatrix output)
      : layerWeights(layerWeights), input(input), output(output) {}
};

struct TargetQValuesData {
  CuMatrix nextTargetActivation;
  CuMatrix nextActionMask;
  CuMatrix batchRewards;
  float discountFactor;
  bool useOnlyReward;
  CuMatrix outTargetValue;

  TargetQValuesData() = default;
  TargetQValuesData(CuMatrix nextTargetActivation, CuMatrix nextActionMask, CuMatrix batchRewards, float discountFactor,
                    bool useOnlyReward, CuMatrix outTargetValue)
      : nextTargetActivation(nextTargetActivation), nextActionMask(nextActionMask), batchRewards(batchRewards),
        discountFactor(discountFactor), useOnlyReward(useOnlyReward), outTargetValue(outTargetValue) {}
};

struct AdamUpdateData {
  CuMatrix gradient;
  CuMatrix momentum;
  CuMatrix rms;
  float beta1;
  float beta2;

  AdamUpdateData() = default;
  AdamUpdateData(CuMatrix gradient, CuMatrix momentum, CuMatrix rms, float beta1, float beta2)
      : gradient(gradient), momentum(momentum), rms(rms), beta1(beta1), beta2(beta2) {
    assert(gradient.rows == momentum.rows);
    assert(gradient.cols == momentum.cols);
    assert(rms.rows == momentum.rows);
    assert(rms.cols == momentum.cols);
  }
};

struct AdamIncrementData {
  CuMatrix weights;
  CuMatrix momentum;
  CuMatrix rms;
  float beta1;
  float beta2;
  float lr;
  float epsilon;

  AdamIncrementData() = default;
  AdamIncrementData(CuMatrix weights, CuMatrix momentum, CuMatrix rms, float beta1, float beta2,
                    float lr, float epsilon)
      : weights(weights), momentum(momentum), rms(rms), beta1(beta1), beta2(beta2), lr(lr),
        epsilon(epsilon) {
    assert(weights.rows == momentum.rows);
    assert(weights.cols == momentum.cols);
    assert(rms.rows == momentum.rows);
    assert(rms.cols == momentum.cols);
  }
};

struct CopyMatrixD2HData {
  CuMatrix src;
  math::MatrixView dst;

  CopyMatrixD2HData() = default;
  CopyMatrixD2HData(CuMatrix src, math::MatrixView dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct CopyMatrixH2DData {
  math::MatrixView src;
  CuMatrix dst;

  CopyMatrixH2DData() = default;
  CopyMatrixH2DData(math::MatrixView src, CuMatrix dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

struct CopyMatrixD2DData {
  CuMatrix src;
  CuMatrix dst;

  CopyMatrixD2DData() = default;
  CopyMatrixD2DData(CuMatrix src, CuMatrix dst) : src(src), dst(dst) {
    assert(dst.rows >= src.rows);
    assert(dst.cols >= src.cols);
    assert(src.data != nullptr && dst.data != nullptr);
  }
};

union TaskData {
  LayerActivationData layerActivationData;
  ErrorMeasureData errorMeasureData;
  PropagateDeltaData propagateDeltaData;
  GradientIncrementData gradientIncrementData;
  FillMatrixData fillMatrixData;
  ScaleMatrixData scaleMatrixData;
  TransposeMatrixData transposeMatrixData;
  ForwardIncrementData forwardIncrementData;
  TargetQValuesData targetQValuesData;
  AdamUpdateData adamUpdateData;
  AdamIncrementData adamIncrementData;
  CopyMatrixD2HData copyMatrixD2HData;
  CopyMatrixH2DData copyMatrixH2DData;
  CopyMatrixD2DData copyMatrixD2DData;
};

struct Task {
  TaskType type;
  TaskData data;

  static Task LayerActivation(ConnectionActivation layer, LayerActivation activation) {
    Task task;
    task.type = TaskType::LAYER_ACTIVATION;
    task.data.layerActivationData = LayerActivationData(layer, activation);
    return task;
  }

  static Task ErrorMeasure(ConnectionActivation networkOutput, TargetOutput targetOutput,
                           CuMatrix deltaMask, LayerBatchDeltas outputLayer) {
    Task task;
    task.type = TaskType::ERROR_MEASURE;
    task.data.errorMeasureData =
        ErrorMeasureData(networkOutput, targetOutput, deltaMask, outputLayer);
    return task;
  }

  static Task PropagateDelta(LayerBatchDeltas nextDelta, CuMatrix transposedWeights,
                             ConnectionActivation connection, LayerBatchDeltas outDelta) {
    Task task;
    task.type = TaskType::PROPAGATE_DELTA;
    task.data.propagateDeltaData =
        PropagateDeltaData(nextDelta, transposedWeights, connection, outDelta);
    return task;
  }

  static Task GradientIncrement(LayerBatchDeltas layerDeltas, ConnectionActivation connection,
                                CuMatrix outGradient) {
    Task task;
    task.type = TaskType::GRADIENT_INCREMENT;
    task.data.gradientIncrementData = GradientIncrementData(layerDeltas, connection, outGradient);
    return task;
  }

  static Task FillMatrix(CuMatrix target, float value) {
    Task task;
    task.type = TaskType::FILL_MATRIX;
    task.data.fillMatrixData = FillMatrixData(target, value);
    return task;
  }

  static Task ScaleMatrix(CuMatrix target, float scale) {
    Task task;
    task.type = TaskType::SCALE_MATRIX;
    task.data.scaleMatrixData = ScaleMatrixData(target, scale);
    return task;
  }

  static Task TransposeMatrix(CuMatrix src, CuMatrix dst) {
    Task task;
    task.type = TaskType::TRANSPOSE_MATRIX;
    task.data.transposeMatrixData = TransposeMatrixData(src, dst);
    return task;
  }

  static Task ForwardIncrement(CuMatrix layerWeights, ConnectionActivation input, CuMatrix output) {
    Task task;
    task.type = TaskType::FORWARD_INCREMENT;
    task.data.forwardIncrementData = ForwardIncrementData(layerWeights, input, output);
    return task;
  }

  static Task TargetQValues(CuMatrix nextTargetActivation, CuMatrix nextActionMask, CuMatrix batchRewards,
                            float discountFactor, bool useOnlyReward, CuMatrix outTargetValue) {
    Task task;
    task.type = TaskType::TARGET_QVALUES;
    task.data.targetQValuesData =
        TargetQValuesData(nextTargetActivation, nextActionMask, batchRewards, discountFactor, useOnlyReward, outTargetValue);
    return task;
  }

  static Task AdamUpdate(CuMatrix gradient, CuMatrix momentum, CuMatrix rms, float beta1,
                         float beta2) {
    Task task;
    task.type = TaskType::ADAM_UPDATE;
    task.data.adamUpdateData = AdamUpdateData(gradient, momentum, rms, beta1, beta2);
    return task;
  }

  static Task AdamIncrement(CuMatrix weights, CuMatrix momentum, CuMatrix rms, float beta1,
                            float beta2, float lr, float epsilon) {
    Task task;
    task.type = TaskType::ADAM_INCREMENT;
    task.data.adamIncrementData =
        AdamIncrementData(weights, momentum, rms, beta1, beta2, lr, epsilon);
    return task;
  }

  static Task CopyMatrixH2D(math::MatrixView src, CuMatrix dst) {
    Task task;
    task.type = TaskType::COPY_MATRIX_H2D;
    task.data.copyMatrixH2DData = CopyMatrixH2DData(src, dst);
    return task;
  }

  static Task CopyMatrixD2H(CuMatrix src, math::MatrixView dst) {
    Task task;
    task.type = TaskType::COPY_MATRIX_D2H;
    task.data.copyMatrixD2HData = CopyMatrixD2HData(src, dst);
    return task;
  }

  static Task CopyMatrixD2D(CuMatrix src, CuMatrix dst) {
    Task task;
    task.type = TaskType::COPY_MATRIX_D2D;
    task.data.copyMatrixD2DData = CopyMatrixD2DData(src, dst);
    return task;
  }
};
}
}
