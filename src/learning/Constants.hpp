
#pragma once

namespace learning {

static constexpr unsigned EXPERIENCE_BATCH_SIZE = 64;
static constexpr unsigned EXPERIENCE_MAX_TRACE_LENGTH = 12;
static constexpr unsigned TARGET_FUNCTION_UPDATE_RATE = 10000;
static constexpr float REWARD_DELAY_DISCOUNT = 0.9f;

static constexpr float CART_WEIGHT_KG = 10.0f;
static constexpr float PENDULUM_LENGTH = 50.0f;
static constexpr float PENDULUM_WEIGHT_KG = 2.0f;
static constexpr float PENDULUM_WIND_STDDEV = 0.1f;

static constexpr float HINGE_ANGLE_THRESHOLD = 60.0f * M_PI / 180.0f;
static constexpr float PENALTY = -1.0f;
}
