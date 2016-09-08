
#include "Action.hpp"
#include <cassert>
#include <mutex>

using namespace simulation;

static std::once_flag stateFlag;
static std::vector<Action> actionSet;

static const std::vector<float> impulses{10.0f, 20.0f};

static void initialiseActions(void) {
  actionSet.emplace_back(0.0f);

  for (const auto &impulse : impulses) {
    actionSet.emplace_back(impulse);
    actionSet.emplace_back(-impulse);
  }
}

unsigned Action::NUM_ACTIONS(void) {
  std::call_once(stateFlag, []() { initialiseActions(); });

  return actionSet.size();
}

Action Action::ACTION(unsigned index) {
  std::call_once(stateFlag, []() { initialiseActions(); });

  return actionSet[index];
}

unsigned Action::ACTION_INDEX(const Action &ga) {
  std::call_once(stateFlag, []() { initialiseActions(); });

  for (unsigned i = 0; i < actionSet.size(); i++) {
    if (ga == actionSet[i]) {
      return i;
    }
  }

  assert(false);
  return 0;
}
