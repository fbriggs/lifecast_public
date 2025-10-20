// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <queue>
#include <chrono>

#include "logger.h"

namespace p11 { namespace concurrency {

using Canceler = std::shared_ptr<std::atomic<bool>>;

class CanceledException : public std::exception {
  using std::exception::exception;
};

template <typename RetT>
class CancelableTask {
public:
  using TaskT = std::function<RetT(Canceler)>;
  CancelableTask(
      std::function<RetT(Canceler)>&& func)
      : func(std::move(func)),
        promise(std::make_shared<std::promise<RetT>>())
  {}

  // if cancelable task goes out of scope before spawning, the promise gets a CancelException
  ~CancelableTask()
  {
    if (thread.has_value()) {
      thread->detach();
    } else {
      cancel();
    }
  }

  // only one thread should ever be able to call set_* on a promise. Main thread before the spawn, and spawned thread after

  void spawn(Canceler cancel_requested) {
    // the cancelable task lives at least as long as the thread, so there is no problem capturing this
    thread = std::thread([cancel_requested, promise=this->promise, func=std::move(this->func)]{
      RetT value;
      try {
        value = func(cancel_requested);
      } catch (...) {
        promise->set_exception(std::current_exception());
        return;
      }

      try {
        promise->set_value(value);
      } catch (...) {
        XPLERROR << "caught exception while setting promise value";
        try {
          promise->set_exception(std::current_exception());
        } catch (...) {
          // set exception can technically throw??
          XCHECK(false) << "set_exception threw";
        }
      }
    });
  }

  void cancel() {
    // If the thread is running, it handles cancel_requested itself
    if (!thread) {
      promise->set_exception(std::make_exception_ptr(CanceledException()));
    }
  }

  std::future<RetT> getFuture() {
    return promise->get_future();
  }

private:
  std::function<RetT(Canceler)> func;
  std::shared_ptr<std::promise<RetT>> promise;
  std::optional<std::thread> thread;
};


template <typename T>
class CancelableTaskQueue {
public:
  CancelableTaskQueue(int max_length, Canceler cancel_requested) : max_length(max_length), cancel_requested(cancel_requested) {
    XCHECK(max_length > 0);
    XCHECK(cancel_requested);
  }

  void push(CancelableTask<T>&& task) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      // This needs to be while in case of spurious wake-up
      while (queue.size() >= max_length) {
        using namespace std::chrono_literals;
        while(not_full.wait_for(lock, 100ms) == std::cv_status::timeout) {
          if (*cancel_requested) {
            return;
          } else if (queue.size() == 0) {
            goto escape_both_loops; // avoid race condition between push and waitPop
          }
        }
      }
      escape_both_loops:
      queue.emplace(task.getFuture());
    }
    not_empty.notify_one();

    // Runs the task on a new thread
    task.spawn(cancel_requested);
  }

  std::optional<T> waitPop() {
    std::optional<T> result;
    {
      std::unique_lock<std::mutex> lock(mutex);
      if (queue.empty()) {
        not_empty.wait(lock);
      }

      // still need to check empty for spurious wake-up
      if (!queue.empty()) {
        // Wait for processing of this future to finish
        std::future<T> future = std::move(queue.front());
        lock.unlock();
        try {
          result = future.get();
        } catch (CanceledException &e) {
          return std::nullopt;
        }
        lock.lock();
      }

      if (!queue.empty()) {
        queue.pop();
      }
    }
    not_full.notify_one();
    return result;
  }

  void endWaitForNonEmpty() {
    not_empty.notify_all();
    not_full.notify_all();
  }

  void cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      queue = std::queue<std::future<T>>();

      *cancel_requested = true;
    }
    endWaitForNonEmpty();
  }

  bool empty() const { return queue.empty(); }

private:
  std::queue<std::future<T>> queue;
  std::mutex mutex;
  std::condition_variable not_empty;
  std::condition_variable not_full;
  int max_length;
  Canceler cancel_requested;
};

}} // end namespace p11::concurrency
