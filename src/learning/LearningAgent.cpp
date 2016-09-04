
#include "LearningAgent.hpp"
#include "../common/Common.hpp"
#include "../rnn/RNN.hpp"
#include "../rnn/RNNSpec.hpp"
#include "Constants.hpp"

#include <boost/thread/shared_mutex.hpp>
#include <cassert>
#include <random>

using namespace learning;

struct LearningAgent::LearningAgentImpl {
  mutable boost::shared_mutex rwMutex;

  float pRandom;
  float temperature;

  uptr<rnn::RNN> learningNet;
  uptr<rnn::RNN> targetNet;
  unsigned itersSinceTargetUpdated = 0;

  LearningAgentImpl() : pRandom(0.1f), temperature(0.1f) {
    rnn::RNNSpec spec;

    spec.numInputs = 3;
    spec.numOutputs = GameAction::ALL_ACTIONS().size();
    spec.hiddenActivation = neuralnetwork::LayerActivation::TANH;
    spec.outputActivation = neuralnetwork::LayerActivation::TANH;
    spec.nodeActivationRate = 1.0f;
    spec.maxBatchSize = EXPERIENCE_BATCH_SIZE;
    spec.maxTraceLength = EXPERIENCE_MAX_TRACE_LENGTH;

    // Connect layer 1 to the input.
    spec.connections.emplace_back(0, 1, 0);

    // Connection layer 1 to layer 2, layer 2 to the output layer.
    spec.connections.emplace_back(1, 2, 0);
    spec.connections.emplace_back(2, 3, 0);

    // Recurrent self-connections for layers 1 and 2.
    spec.connections.emplace_back(1, 1, 1);
    spec.connections.emplace_back(2, 2, 1);

    // 2 layers, 1 hidden.
    spec.layers.emplace_back(1, 64, false);
    spec.layers.emplace_back(2, 64, false);
    spec.layers.emplace_back(3, spec.numOutputs, true);

    learningNet = make_unique<rnn::RNN>(spec);
    targetNet = learningNet->RefreshAndGetTarget();
    itersSinceTargetUpdated = 0;
  }

  Action SelectAction(const State *state) {
    assert(state != nullptr);

    boost::shared_lock<boost::shared_mutex> lock(rwMutex);
    return chooseBestAction(state);
  }

  void SetPRandom(float pRandom) {
    assert(pRandom >= 0.0f && pRandom <= 1.0f);
    this->pRandom = pRandom;
  }

  void SetTemperature(float temperature) {
    assert(temperature > 0.0f);
    this->temperature = temperature;
  }

  Action SelectLearningAction(const State *state) {
    assert(state != nullptr);

    boost::shared_lock<boost::shared_mutex> lock(rwMutex);
    if (Util::RandInterval(0.0, 1.0) < pRandom) {
      return chooseExplorativeAction(state);
    } else {
      // return chooseWeightedAction(state);
      return chooseBestAction(state);
    }
  }

  void Learn(const vector<Experience> &experiences, float learnRate) {
    if (itersSinceTargetUpdated > TARGET_FUNCTION_UPDATE_RATE) {
      boost::unique_lock<boost::shared_mutex> lock(rwMutex);

      targetNet = learningNet->RefreshAndGetTarget();
      itersSinceTargetUpdated = 0;
    }
    itersSinceTargetUpdated++;

    vector<neuralnetwork::TrainingSample> learnSamples;
    learnSamples.reserve(moments.size());

    for (const auto &moment : moments) {
      learnSamples.emplace_back(moment.initialState, moment.successorState,
                                GameAction::ACTION_INDEX(moment.actionTaken),
                                moment.isSuccessorTerminal, moment.reward, REWARD_DELAY_DISCOUNT);
    }

    learningNet->Update(neuralnetwork::SamplesProvider(learnSamples), learnRate);
  }

  void Finalise(void) {
    // obtain a write lock
    boost::unique_lock<boost::shared_mutex> lock(rwMutex);

    if (learningNet == nullptr) {
      return;
    }

    targetNet = learningNet->RefreshAndGetTarget();
    learningNet.release();
    learningNet = nullptr;
  }

  Action chooseBestAction(const State *state) {
    EMatrix qvalues = targetNet->Process(state->Encode());
    assert(qvalues.rows() == static_cast<int>(GameAction::ALL_ACTIONS().size()));

    std::vector<unsigned> availableActions = state.AvailableActions();
    assert(availableActions.size() > 0);

    Action bestAction = GameAction::ACTION(availableActions[0]);
    float bestQValue = qvalues(availableActions[0], 0);

    for (unsigned i = 1; i < availableActions.size(); i++) {
      if (qvalues(availableActions[i]) > bestQValue) {
        bestQValue = qvalues(availableActions[i], 0);
        bestAction = Action::ACTION(availableActions[i]);
      }
    }
    return bestAction;
  }

  Action chooseExplorativeAction(const State *state) {
    auto aa = state->AvailableActions();
    return GameAction::ACTION(aa[rand() % aa.size()]);
  }

  Action chooseWeightedAction(const State *state) {
    EMatrix qvalues = targetNet->Process(state->Encode());
    assert(qvalues.rows() == static_cast<int>(GameAction::ALL_ACTIONS().size()));

    std::vector<unsigned> availableActions = state->AvailableActions();
    std::vector<float> weights;

    for (unsigned i = 0; i < availableActions.size(); i++) {
      weights.push_back(qvalues(availableActions[i]) / temperature, 0);
    }
    weights = Util::SoftmaxWeights(weights);

    float sample = Util::RandInterval(0.0, 1.0);
    for (unsigned i = 0; i < weights.size(); i++) {
      sample -= weights[i];
      if (sample <= 0.0f) {
        return Action::ACTION(availableActions[i]);
      }
    }

    return chooseExplorativeAction(state);
  }
};

LearningAgent::LearningAgent() : impl(new LearningAgentImpl()) {}
LearningAgent::~LearningAgent() = default;

Action LearningAgent::SelectAction(const State *state) { return impl->SelectAction(state); }

void LearningAgent::SetPRandom(float pRandom) { impl->SetPRandom(pRandom); }
void LearningAgent::SetTemperature(float temperature) { impl->SetTemperature(temperature); }

Action LearningAgent::SelectLearningAction(const State *state) {
  return impl->SelectLearningAction(state);
}

void LearningAgent::Learn(const vector<Experience> &experiences, float learnRate) {
  impl->Learn(experiences, learnRate);
}

void LearningAgent::Finalise(void) { impl->Finalise(); }
