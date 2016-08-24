#pragma once

#include "Task.hpp"
#include <memory>

namespace rnn {
namespace cuda {

class TaskExecutor {
public:
  TaskExecutor();
  ~TaskExecutor();

  void Execute(const Task &task);

private:
  struct TaskExecutorImpl;
  std::unique_ptr<TaskExecutorImpl> impl;
};
}
}
