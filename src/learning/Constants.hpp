
#pragma once

namespace learning {

static constexpr unsigned EXPERIENCE_BATCH_SIZE = 64;
static constexpr unsigned EXPERIENCE_MAX_TRACE_LENGTH = 8;
static constexpr unsigned TARGET_FUNCTION_UPDATE_RATE = 1000;
static constexpr float REWARD_DELAY_DISCOUNT = 0.9f;

static constexpr float CART_WEIGHT_KG = 10.0f;
static constexpr float PENDULUM_LENGTH = 50.0f;
static constexpr float PENDULUM_WEIGHT_KG = 2.0f;
static constexpr float PENDULUM_WIND_STDDEV = 0.5f;

static constexpr float HINGE_ANGLE_THRESHOLD = 60.0f * M_PI / 180.0f;
static constexpr float PENALTY = -1.0f;
}
