
#include "Action.hpp"
#include <cassert>
#include <mutex>

using namespace simulation;

static std::once_flag stateFlag;
static std::vector<Action> actionSet;
static std::vector<unsigned> actionIndices;

static const std::vector<float> impulses{50.0f, 100.0f, 200.0f};

const std::vector<unsigned> &Action::ALL_ACTIONS(void) {
  std::call_once(stateFlag, []() {
    actionIndices.push_back(actionSet.size());
    actionSet.emplace_back(0.0f);

    for (const auto &impulse : impulses) {
      actionIndices.push_back(actionSet.size());
      actionSet.emplace_back(impulse);

      actionIndices.push_back(actionSet.size());
      actionSet.emplace_back(-impulse);
    }
  });

  return actionIndices;
}

Action Action::ACTION(unsigned index) { return ALL_ACTIONS()[index]; }

unsigned Action::ACTION_INDEX(const Action &ga) {
  for (const auto ai : ALL_ACTIONS()) {
    if (ga == ACTION(ai)) {
      return ai;
    }
  }

  assert(false);
  return 0;
}
