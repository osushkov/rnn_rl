
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

  void Read(std::istream &in) {
    for (auto &layer : layers) {
      layer.Read(in);
    }

    vector<pair<LayerConnection, math::MatrixView>> weights = getHostWeights();
    cudaTrainer.SetWeights(weights);
  }

  void Write(std::ostream &out) const {
    spec.Write(out);

    for (const auto &layer : layers) {
      layer.Write(out);
    }
  }

  void ClearMemory(void) { previous = Maybe<TimeSlice>::none; }

  EVector Process(const EVector &input) {
    assert(input.rows() == spec.numInputs);

    TimeSlice *prevSlice = previous.valid() ? &(previous.val()) : nullptr;
    TimeSlice curSlice(0, input, layers);

    EVector output = forwardPass(prevSlice, curSlice);
    previous = Maybe<TimeSlice>(curSlice);
    return output;
  }

  void Update(const vector<SliceBatch> &trace, float learnRate) {
    cudaTrainer.Train(trace, learnRate);
  }

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

  EVector forwardPass(const TimeSlice *prevSlice, TimeSlice &curSlice) {
    for (const auto &layer : layers) {
      pair<EVector, EVector> layerOut = getLayerOutput(layer, prevSlice, curSlice);

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
  pair<EVector, EVector> getLayerOutput(const Layer &layer, const TimeSlice *prevSlice,
                                        const TimeSlice &curSlice) {
    EVector incoming(layer.numNodes);
    incoming.fill(0.0f);

    for (const auto &connection : layer.weights) {
      incrementIncomingWithConnection(connection, prevSlice, curSlice, incoming);
    }

    return performLayerActivations(layer, incoming);
  }

  void incrementIncomingWithConnection(const pair<LayerConnection, EMatrix> &connection,
                                       const TimeSlice *prevSlice, const TimeSlice &curSlice,
                                       EVector &incoming) {

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

  pair<EVector, EVector> performLayerActivations(const Layer &layer, const EVector &incoming) {
    EVector activation(incoming.rows());
    EVector derivatives(incoming.rows());

    if (layer.isOutput && spec.outputActivation == LayerActivation::SOFTMAX) {
      activation = math::SoftmaxActivations(incoming);
    } else {
      for (int r = 0; r < activation.rows(); r++) {
        activation(r) = ActivationValue(spec.hiddenActivation, incoming(r));
        derivatives(r) = ActivationDerivative(spec.hiddenActivation, incoming(r), activation(r));
      }
    }

    return make_pair(activation, derivatives);
  }

  EVector getInputWithBias(const EVector &noBiasInput) const {
    EVector result(noBiasInput.rows() + 1);
    result.topRightCorner(noBiasInput.rows(), result.cols()) = noBiasInput;
    result.bottomRightCorner(1, result.cols()).fill(1.0f);
    return result;
  }
};

uptr<RNN> RNN::Read(std::istream &in) {
  RNNSpec spec = RNNSpec::Read(in);
  uptr<RNN> result = make_unique<RNN>(spec);
  result->Read(in);
  return move(result);
}

RNN::RNN(const RNNSpec &spec) : impl(new RNNImpl(spec)) {}
RNN::~RNN() = default;

void RNN::Write(std::ostream &out) const { impl->Write(out); }

RNNSpec RNN::GetSpec(void) const { return impl->spec; }

void RNN::ClearMemory(void) { impl->ClearMemory(); }

EVector RNN::Process(const EVector &input) { return impl->Process(input); }

void RNN::Update(const vector<SliceBatch> &trace, float learnRate) {
  impl->Update(trace, learnRate);
}

void RNN::RefreshAndGetTarget(void) { impl->RefreshAndGetTarget(); }
