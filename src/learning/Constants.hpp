
#pragma once

namespace learning {

static constexpr unsigned EXPERIENCE_BATCH_SIZE = 100;
static constexpr unsigned EXPERIENCE_MAX_TRACE_LENGTH = 8;
static constexpr unsigned TARGET_FUNCTION_UPDATE_RATE = 5000;
static constexpr float REWARD_DELAY_DISCOUNT = 0.99f;
}
