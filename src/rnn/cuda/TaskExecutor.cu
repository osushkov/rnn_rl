
#include "TaskExecutor.hpp"
#include "kernels/ActivationKernel.cuh"
#include "kernels/AdamKernel.cuh"
#include "kernels/SoftmaxKernel.cuh"
#include "kernels/BackwardDeltaKernel.cuh"
#include "kernels/GradientIncrementKernel.cuh"
#include "kernels/MatrixFillKernel.cuh"
#include "kernels/MatrixScaleKernel.cuh"
#include "kernels/TransposeKernel.cuh"
#include "kernels/WeightedIncrementKernel.cuh"
#include "kernels/ErrorMeasureKernel.cuh"
#include "kernels/TargetValuesKernel.cuh"
#include "Util.cuh"
#include <cuda_runtime.h>

using namespace rnn;
using namespace rnn::cuda;

struct TaskExecutor::TaskExecutorImpl {
  cudaStream_t stream;

  TaskExecutorImpl() {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }

  ~TaskExecutorImpl() {
    cudaStreamDestroy(stream);
  }

  void Execute(const Task &t) {
    cudaError_t err;

    switch(t.type) {
    case TaskType::LAYER_ACTIVATION:
      if (t.data.layerActivationData.activation == LayerActivation::SOFTMAX) {
        SoftmaxKernel::Apply(t.data.layerActivationData.layer, stream);
      } else {
        ActivationKernel::Apply(
            t.data.layerActivationData.layer, t.data.layerActivationData.activation, stream);
      }
      return;
    case TaskType::ERROR_MEASURE:
      ErrorMeasureKernel::Apply(t.data.errorMeasureData.networkOutput,
        t.data.errorMeasureData.targetOutput, t.data.errorMeasureData.deltaMask,
        t.data.errorMeasureData.outputLayer, stream);
      return;
    case TaskType::PROPAGATE_DELTA:
      BackwardDeltaKernel::Apply(t.data.propagateDeltaData.nextDelta,
        t.data.propagateDeltaData.transposedWeights, t.data.propagateDeltaData.connection,
        t.data.propagateDeltaData.outDelta, stream);
      return;
    case TaskType::GRADIENT_INCREMENT:
      GradientIncrementKernel::Apply(t.data.gradientIncrementData.layerDeltas,
        t.data.gradientIncrementData.connection, t.data.gradientIncrementData.outGradient, stream);
      return;
    case TaskType::FILL_MATRIX:
      MatrixFillKernel::Apply(t.data.fillMatrixData.target, t.data.fillMatrixData.value, stream);
      return;
    case TaskType::SCALE_MATRIX:
      MatrixScaleKernel::Apply(t.data.scaleMatrixData.target, t.data.scaleMatrixData.scale, stream);
      return;
    case TaskType::TRANSPOSE_MATRIX:
      TransposeKernel::Apply(t.data.transposeMatrixData.src, t.data.transposeMatrixData.dst, stream);
      return;
    case TaskType::FORWARD_INCREMENT:
      WeightedIncrementKernel::Apply(t.data.forwardIncrementData.layerWeights,
        t.data.forwardIncrementData.input, t.data.forwardIncrementData.output, stream);
      return;
    case TaskType::TARGET_QVALUES:
      TargetValuesKernel::Apply(t.data.targetQValuesData.nextTargetActivation, t.data.targetQValuesData.nextActionMask,
        t.data.targetQValuesData.batchRewards, t.data.targetQValuesData.discountFactor,
        t.data.targetQValuesData.useOnlyReward, t.data.targetQValuesData.outTargetValue, stream);
      return;
    case TaskType::ADAM_UPDATE:
      AdamKernel::UpdateMomentumAndRMS(
        t.data.adamUpdateData.gradient, t.data.adamUpdateData.momentum, t.data.adamUpdateData.rms,
        t.data.adamUpdateData.beta1, t.data.adamUpdateData.beta2, stream);
      return;
    case TaskType::ADAM_INCREMENT:
      AdamKernel::UpdateWeightsWithAdam(
        t.data.adamIncrementData.weights, t.data.adamIncrementData.momentum,
        t.data.adamIncrementData.rms, t.data.adamIncrementData.beta1, t.data.adamIncrementData.beta2,
        t.data.adamIncrementData.lr, t.data.adamIncrementData.epsilon, stream);
      return;
    case TaskType::COPY_MATRIX_D2H:
      err = cudaMemcpy2DAsync(
        t.data.copyMatrixD2HData.dst.data, t.data.copyMatrixD2HData.dst.cols * sizeof(float),
        t.data.copyMatrixD2HData.src.data, t.data.copyMatrixD2HData.src.pitch,
        t.data.copyMatrixD2HData.src.cols * sizeof(float), t.data.copyMatrixD2HData.src.rows,
        cudaMemcpyDeviceToHost, stream);
      CheckError(err);
      return;
    case TaskType::COPY_MATRIX_H2D:
      err = cudaMemcpy2DAsync(
        t.data.copyMatrixH2DData.dst.data, t.data.copyMatrixH2DData.dst.pitch,
        t.data.copyMatrixH2DData.src.data, t.data.copyMatrixH2DData.src.cols * sizeof(float),
        t.data.copyMatrixH2DData.src.cols * sizeof(float), t.data.copyMatrixH2DData.src.rows,
        cudaMemcpyHostToDevice, stream);
      CheckError(err);
      return;
    case TaskType::COPY_MATRIX_D2D:
      err = cudaMemcpy2DAsync(
        t.data.copyMatrixD2DData.dst.data, t.data.copyMatrixD2DData.dst.pitch,
        t.data.copyMatrixD2DData.src.data, t.data.copyMatrixD2DData.src.pitch,
        t.data.copyMatrixD2DData.src.cols * sizeof(float), t.data.copyMatrixD2DData.src.rows,
        cudaMemcpyDeviceToDevice, stream);
      CheckError(err);
      return;
    default:
      assert(false);
    }
  }
};

TaskExecutor::TaskExecutor() : impl(new TaskExecutorImpl()) {}

TaskExecutor::~TaskExecutor() = default;

void TaskExecutor::Execute(const Task &task) { impl->Execute(task); }
