
#include "RNN.hpp"
#include "../common/Maybe.hpp"
#include "Activations.hpp"
#include "CudaTrainer.hpp"
#include "Layer.hpp"
#include "LayerDef.hpp"
#include "TimeSlice.hpp"
#include <cassert>
#include <utility>

using namespace rnn;

struct RNN::RNNImpl {
  RNNSpec spec;
  vector<Layer> layers;
  Maybe<TimeSlice> previous;

  CudaTrainer cudaTrainer;

  RNNImpl(const RNNSpec &spec) : spec(spec), previous(Maybe<TimeSlice>::none), cudaTrainer(spec) {
    for (const auto &ls : spec.layers) {
      layers.emplace_back(spec, ls);
    }

    vector<pair<LayerConnection, math::MatrixView>> weights = getHostWeights();
    cudaTrainer.SetWeights(weights);
  }

  void ClearMemory(void) { previous = Maybe<TimeSlice>::none; }

  EMatrix Process(const EMatrix &input) {
    assert(input.rows() == spec.numInputs);
    assert(input.cols() > 0);

    TimeSlice *prevSlice = previous.valid() ? &(previous.val()) : nullptr;
    TimeSlice curSlice(0, input, layers);

    EMatrix output = forwardPass(prevSlice, curSlice);
    previous = Maybe<TimeSlice>(curSlice);
    return output;
  }

  void Update(const vector<SliceBatch> &trace) { cudaTrainer.Train(trace); }

  void RefreshAndGetTarget(void) {
    vector<pair<LayerConnection, math::MatrixView>> weights = getHostWeights();
    cudaTrainer.GetWeights(weights);
    cudaTrainer.UpdateTarget();
  }

  vector<pair<LayerConnection, math::MatrixView>> getHostWeights(void) {
    vector<pair<LayerConnection, math::MatrixView>> weights;
    for (auto &l : layers) {
      for (auto &c : l.weights) {
        weights.emplace_back(c.first, math::GetMatrixView(c.second));
      }
    }
    return weights;
  }

  EMatrix forwardPass(const TimeSlice *prevSlice, TimeSlice &curSlice) {
    for (const auto &layer : layers) {
      pair<EMatrix, EMatrix> layerOut = getLayerOutput(layer, prevSlice, curSlice);

      for (const auto &oc : layer.outgoing) {
        ConnectionMemoryData *cmd = curSlice.GetConnectionData(oc);
        assert(cmd != nullptr);

        cmd->activation = layerOut.first;
        cmd->derivative = layerOut.second;
        cmd->haveActivation = true;

        if (oc.timeOffset == 0) {
          cmd->activation *= spec.nodeActivationRate;
        }
      }

      if (layer.isOutput) {
        curSlice.networkOutput = layerOut.first;
      }
    }

    assert(curSlice.networkOutput.rows() == spec.numOutputs);
    return curSlice.networkOutput;
  }

  // Returns the output vector of the layer, and the derivative vector for the layer.
  pair<EMatrix, EMatrix> getLayerOutput(const Layer &layer, const TimeSlice *prevSlice,
                                        const TimeSlice &curSlice) {
    EMatrix incoming(layer.numNodes, curSlice.networkInput.cols());
    incoming.fill(0.0f);

    for (const auto &connection : layer.weights) {
      incrementIncomingWithConnection(connection, prevSlice, curSlice, incoming);
    }

    return performLayerActivations(layer, incoming);
  }

  void incrementIncomingWithConnection(const pair<LayerConnection, EMatrix> &connection,
                                       const TimeSlice *prevSlice, const TimeSlice &curSlice,
                                       EMatrix &incoming) {

    if (connection.first.srcLayerId == 0) { // special case for input
      assert(connection.first.timeOffset == 0);
      incoming += connection.second * getInputWithBias(curSlice.networkInput);
    } else {
      const ConnectionMemoryData *connectionMemory = nullptr;

      if (connection.first.timeOffset == 0) {
        connectionMemory = curSlice.GetConnectionData(connection.first);
        assert(connectionMemory != nullptr);
      } else if (prevSlice != nullptr) {
        connectionMemory = prevSlice->GetConnectionData(connection.first);
        assert(connectionMemory != nullptr);
      }

      if (connectionMemory != nullptr) {
        assert(connectionMemory->haveActivation);
        incoming += connection.second * getInputWithBias(connectionMemory->activation);
      }
    }
  }

  pair<EMatrix, EMatrix> performLayerActivations(const Layer &layer, const EMatrix &incoming) {
    EMatrix activation(incoming.rows(), incoming.cols());
    EMatrix derivatives(incoming.rows(), incoming.cols());

    if (layer.isOutput && spec.outputActivation == LayerActivation::SOFTMAX) {
      for (int c = 0; c < activation.cols(); c++) {
        activation.col(c) = math::SoftmaxActivations(incoming.col(c));
      }
    } else {
      for (int c = 0; c < activation.cols(); c++) {
        for (int r = 0; r < activation.rows(); r++) {
          activation(r, c) = ActivationValue(spec.hiddenActivation, incoming(r, c));
          derivatives(r, c) =
              ActivationDerivative(spec.hiddenActivation, incoming(r, c), activation(r, c));
        }
      }
    }

    return make_pair(activation, derivatives);
  }

  EMatrix getInputWithBias(const EMatrix &noBiasInput) const {
    EMatrix result(noBiasInput.rows() + 1, noBiasInput.cols());
    result.topRightCorner(noBiasInput.rows(), result.cols()) = noBiasInput;
    result.bottomRightCorner(1, result.cols()).fill(1.0f);
    return result;
  }
};

RNN::RNN(const RNNSpec &spec) : impl(new RNNImpl(spec)) {}
RNN::~RNN() = default;

RNNSpec RNN::GetSpec(void) const {
  return impl->spec;
}

void RNN::ClearMemory(void) { impl->ClearMemory(); }

EMatrix RNN::Process(const EMatrix &input) { return impl->Process(input); }

void RNN::Update(const vector<SliceBatch> &trace) { impl->Update(trace); }

void RNN::RefreshAndGetTarget(void) { impl->RefreshAndGetTarget(); }
