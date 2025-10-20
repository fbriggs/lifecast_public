// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "logger.h"
#include <atomic>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <mutex>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdio.h>
#include <functional>

#ifdef _WIN32
#include <Windows.h>
#include <tchar.h>
#endif

namespace p11 {

// This is similar to std::system() but it returns the output of the command
std::string execBlockingWithOutput(const std::string& command);

enum CommandType {
    kCommandProcess = 0,
    kCommandThread = 1
};

struct CommandProgressDescription {
  std::string progress_str = "";
  float frac = 1;
  std::string phase; // keeps track of state for commands that have multiple "phases" of progress
};

struct QueableCommand {
    CommandType command_type;
    std::string command;
    std::function<void()> lambda;
    std::function<void(const std::string&, CommandProgressDescription&)> progress_parser;
    std::shared_ptr<std::atomic<bool>> thread_cancel_requested;
    bool capture_xpl_logs;
};

// This is for running commands asynchronously.
class CommandRunner {
 public:
  CommandRunner() 
#ifdef _WIN32
      : running_(false), kill_requested_(false)
#else
      : infp_(nullptr), outfp_(nullptr), child_pid_(0), running_(false), kill_requested_(false)
#endif
  {}
  ~CommandRunner() { cleanup(); }

  void setRedirectStdErr(const bool x) { redirect_stderr_to_stdout = x; }

  void setCompleteCallback(const std::function<void()>& complete_callback) {
    complete_callback_ = complete_callback;
  }

  void setKilledCallback(const std::function<void()>& killed_callback) {
    killed_callback_ = killed_callback;
  }

  void setCompleteOrKilledCallback(const std::function<void()>& callback) {
    complete_callback_ = callback;
    killed_callback_ = callback;
  }

  void queueShellCommand(
    const std::string& command,
    const std::function<void(const std::string&, CommandProgressDescription&)> progress_parser = nullptr
  );

  void queueThreadCommand(
      std::shared_ptr<std::atomic<bool>>& thread_cancel_requested,
      const std::function<void()>& lambda,
      const std::function<void(const std::string&, CommandProgressDescription&)> progress_parser = nullptr,
      const bool capture_xpl_logs = true);

  void runCommandQueue();

  // Kill the process or thread that is running
  void kill();

  std::string getOutput() { return output_.str(); }
  void clearOutput() { output_.clear(); }
  bool isRunning() const { return running_ || !command_queue_.empty(); }
  bool waitingForCancel() { return running_ && kill_requested_; }

  CommandProgressDescription updateAndGetProgress();

 private:

  void executeShellCommand(std::string command);

  // To properly handle kill(), you must have a std::atomic<bool> to
  // keep track of whether a cancel has been requested (which won't go out of scope),
  // and pass a pointer to it here. If null is provided, kill() just wont do anything.
  void runLambdaInThread(
      std::shared_ptr<std::atomic<bool>>& thread_cancel_requested,
      const std::function<void()>& lambda,
      const bool capture_xpl_logs = true);

#ifdef _WIN32
  HANDLE stdout_rd_;
  PROCESS_INFORMATION process_info_;
#else
  FILE* infp_;
  FILE* outfp_;
  pid_t child_pid_;
#endif
  std::queue<QueableCommand> command_queue_;
  QueableCommand curr_command_; // The command that is currently running, kept so we can do progress stuff
  std::atomic<bool> running_;
  bool kill_requested_;
  std::shared_ptr<std::atomic<bool>> thread_cancel_requested_;
  bool is_thread_;
  bool capture_xpl_logs_;
  xpl::ThreadSafeStringStream output_;
  std::mutex mutex_;
  std::function<void()> complete_callback_;
  std::function<void()> killed_callback_;
  bool redirect_stderr_to_stdout = false;
  CommandProgressDescription progress_description;

  void cleanup();
  void readOutput();
};

}  // namespace p11
