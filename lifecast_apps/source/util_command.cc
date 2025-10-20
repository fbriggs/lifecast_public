// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "util_command.h"

#include <sys/types.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <poll.h>
#endif
#include <signal.h>
#include <array>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <array>
#include <memory>
#include <stdexcept>
#include <windows.h>
#endif

namespace p11 {

std::string execBlockingWithOutput(const std::string& command) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        throw std::runtime_error("Failed to create pipe");
    }

    STARTUPINFOA si = {sizeof(STARTUPINFOA)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;

    PROCESS_INFORMATION pi;
    
    if (!CreateProcessA(
        NULL,
        const_cast<LPSTR>(command.c_str()),
        NULL,
        NULL,
        TRUE,
        CREATE_NO_WINDOW,
        NULL,
        NULL,
        &si,
        &pi
    )) {
        CloseHandle(hReadPipe);
        CloseHandle(hWritePipe);
        throw std::runtime_error("Failed to create process");
    }

    CloseHandle(hWritePipe);

    std::string result;
    std::array<char, 4096> buffer;
    DWORD bytesRead;

    while (ReadFile(hReadPipe, buffer.data(), buffer.size(), &bytesRead, NULL) && bytesRead > 0) {
        result.append(buffer.data(), bytesRead);
    }

    CloseHandle(hReadPipe);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return result;
#else
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen((command + " 2>&1").c_str(), "r"), pclose);

  //XCHECK(pipe) << "failed to open pipe while running command: " << command;
  if (!pipe) {
    return  "ERROR: failed to open pipe. command: " + command;
  }

  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
  }

  return result;
#endif
}

void CommandRunner::queueShellCommand(
  const std::string& command,
  const std::function<void(const std::string&, CommandProgressDescription&)> progress_parser
) {
  QueableCommand cmd;
  cmd.command_type = kCommandProcess;
  cmd.command = command;
  cmd.progress_parser = progress_parser;
  command_queue_.push(cmd);
}

void CommandRunner::queueThreadCommand(
    std::shared_ptr<std::atomic<bool>>& thread_cancel_requested,
    const std::function<void()>& lambda,
    const std::function<void(const std::string&, CommandProgressDescription&)> progress_parser,
    const bool capture_xpl_logs)
{
  QueableCommand cmd;
  cmd.command_type = kCommandThread;
  cmd.thread_cancel_requested = thread_cancel_requested;
  cmd.lambda = lambda;
  cmd.progress_parser = progress_parser;
  cmd.capture_xpl_logs = capture_xpl_logs;
  command_queue_.push(cmd);
}

void CommandRunner::executeShellCommand(std::string command)
{
  if (running_) return;

  std::lock_guard<std::mutex> lock(mutex_);

#ifndef _WIN32 // stderr redirect not implemented on windows
  if (redirect_stderr_to_stdout) {
    command += " 2>&1";
  }
#endif

  is_thread_ = false;
  output_ << command + "\n";

#ifdef _WIN32
  SECURITY_ATTRIBUTES  security_attributes;
  stdout_rd_ = INVALID_HANDLE_VALUE;
  HANDLE stdout_wr = INVALID_HANDLE_VALUE;
  STARTUPINFO startup_info;

  security_attributes.nLength              = sizeof(SECURITY_ATTRIBUTES);
  security_attributes.bInheritHandle       = TRUE;
  security_attributes.lpSecurityDescriptor = nullptr;

  if (!CreatePipe(&stdout_rd_, &stdout_wr, &security_attributes, 0) ||
          !SetHandleInformation(stdout_rd_, HANDLE_FLAG_INHERIT, 0)) {
      XPLINFO << "Error creating pipes";
      return;
  }

  ZeroMemory(&process_info_, sizeof(PROCESS_INFORMATION));
  ZeroMemory(&startup_info, sizeof(STARTUPINFO));

  startup_info.cb         = sizeof(STARTUPINFO);
  startup_info.hStdInput  = 0;
  startup_info.hStdOutput = stdout_wr;

  if(stdout_rd_) startup_info.dwFlags |= STARTF_USESTDHANDLES;

  static constexpr int kMaxCommandLength = 65536;
  char CmdLineStr[kMaxCommandLength];
  XCHECK_LT(command.size(), kMaxCommandLength);
  strncpy(CmdLineStr, command.c_str(), kMaxCommandLength);
  CmdLineStr[kMaxCommandLength-1] = 0;

  int success = CreateProcess(
      nullptr,
      CmdLineStr,
      nullptr,
      nullptr,
      true,
      0,
      nullptr,
      nullptr,
      &startup_info,
      &process_info_
  );
  CloseHandle(stdout_wr);

  if(!success) {
    XPLINFO << "CreateProcess failed";
    CloseHandle(process_info_.hProcess);
    CloseHandle(process_info_.hThread);
    CloseHandle(stdout_rd_);
    return;
  } else {
    CloseHandle(process_info_.hThread);
  }

#else
  int p_stdin[2], p_stdout[2];
  if (pipe(p_stdin) != 0 || pipe(p_stdout) != 0) {
    return;  // Failed to create pipes
  }

  pid_t pid = fork();
  if (pid < 0) {
    return;  // Fork failed
  } else if (pid == 0) {
    // Child process
    close(p_stdin[1]);
    dup2(p_stdin[0], STDIN_FILENO);
    close(p_stdout[0]);
    dup2(p_stdout[1], STDOUT_FILENO);

    // execute the command
    setpgid(0, 0);  // Set the child's process group ID to its own PID
    execl("/bin/bash", "bash", "-c", command.c_str(), NULL);
    exit(0);
  }

  // Parent process
  close(p_stdin[0]);
  infp_ = fdopen(p_stdin[1], "w");
  close(p_stdout[1]);
  outfp_ = fdopen(p_stdout[0], "r");
  setbuf(outfp_, nullptr);  // Disable output buffering
  child_pid_ = pid;
#endif

  running_ = true;
  std::thread read_thread(&CommandRunner::readOutput, this);
  read_thread.detach();
}

void CommandRunner::runLambdaInThread(
    std::shared_ptr<std::atomic<bool>>& thread_cancel_requested,
    const std::function<void()>& lambda,
    const bool capture_xpl_logs)
{
  if (running_) { return; }

  thread_cancel_requested_ = thread_cancel_requested;
  if (thread_cancel_requested_) (*thread_cancel_requested_) = false;

  std::lock_guard<std::mutex> lock(mutex_);

  capture_xpl_logs_ = capture_xpl_logs; 
  if(capture_xpl_logs) {
    p11_xpl_include_dev_info = true;
    xpl::stdoutLogger.attachStreamCapture(&output_);
  }

  is_thread_ = true;
  running_ = true;

  std::thread run_thread([&, lambda] {
    lambda();

    cleanup();
  });
  run_thread.detach();
}

void CommandRunner::kill()
{
  if (!running_) return;

  std::lock_guard<std::mutex> lock(mutex_);
  kill_requested_ = true;

  std::queue<QueableCommand>().swap(command_queue_); // Clear the command queue.

  if (is_thread_) {
    if (thread_cancel_requested_) (*thread_cancel_requested_) = true;
  } else {
#ifdef _WIN32
    TerminateProcess(process_info_.hProcess, 0);
    CloseHandle(process_info_.hProcess);
    CloseHandle(process_info_.hThread);
    CloseHandle(stdout_rd_);
#else
    std::string cmd = "kill -- -" + std::to_string(child_pid_); // kill the whole process group
    XPLINFO << "kill process cmd: " << cmd;
    system(cmd.c_str());
    std::string kill_cmd_result = execBlockingWithOutput(cmd);
    XPLINFO << "kill result: " << kill_cmd_result;

    ::kill(child_pid_, SIGKILL); // Kill the child process another way
#endif
    cleanup();
  }
}

void CommandRunner::cleanup()
{
  if (!running_) return;

  if (is_thread_) {
    if (capture_xpl_logs_) {
      p11_xpl_include_dev_info = true;
      xpl::stdoutLogger.removeStreamCapture();
    }
  } else {
#ifdef _WIN32
  // TODO: should this be done in all cleanup scenarios, or only when readOutput finishes normally?
  WaitForSingleObject(process_info_.hProcess, INFINITE);
  CloseHandle(process_info_.hProcess); 
#else
    fclose(infp_);
    fclose(outfp_);
    int status;
    waitpid(child_pid_, &status, WNOHANG);
#endif
  }

  running_ = false;
}

void CommandRunner::readOutput()
{
  static constexpr int kBufferSize = 4096;

#ifdef _WIN32

  DWORD  bytes_read;
  char buffer[kBufferSize];
  while(true) {
    bytes_read = 0;
    int success = ReadFile(
        stdout_rd_,
        buffer,
        (DWORD)kBufferSize,
        &bytes_read,
        nullptr
    );

    if (!success || bytes_read == 0) break;
    output_.write(buffer, bytes_read);
  }
  
  // Note: in case we need the process exit code, try code below
  //int exit_code;
  //GetExitCodeProcess(process_info_.hProcess, (DWORD*) &exit_code);

#else

  char buffer[kBufferSize];
  ssize_t bytes_read;

  int fd = fileno(outfp_);

  while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
    output_.write(buffer, bytes_read);
  }

#endif

  cleanup();
}

void CommandRunner::runCommandQueue() {
  kill_requested_ = false;
  progress_description = CommandProgressDescription();
  std::thread queue_thread([this]{
    while(!command_queue_.empty()) {
      // Wait for any other commands that are currently running before starting the next one.
      while(running_ && !kill_requested_) std::this_thread::sleep_for(std::chrono::milliseconds(10));
      
      if (kill_requested_) break;

      QueableCommand next_cmd = command_queue_.front();
      command_queue_.pop();

      curr_command_ = next_cmd;

      if (next_cmd.command_type == kCommandProcess) {
        executeShellCommand(next_cmd.command);
      } else if (next_cmd.command_type == kCommandThread) {
        runLambdaInThread(
          next_cmd.thread_cancel_requested,
          next_cmd.lambda,
          next_cmd.capture_xpl_logs);
      }
    }

    while(running_) std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::queue<QueableCommand>().swap(command_queue_); // Clear the command queue.
    
    if (!kill_requested_ && complete_callback_) complete_callback_();
    if (kill_requested_ && killed_callback_) killed_callback_();
  });
  queue_thread.detach();
}

CommandProgressDescription CommandRunner::updateAndGetProgress() {
  if (!running_) return CommandProgressDescription();

  const std::vector<std::string> new_lines = output_.getNewIncrementalLines();
  for (const std::string& line : new_lines) {
    if (curr_command_.progress_parser != nullptr) {
      curr_command_.progress_parser(line, progress_description);
    } else {
      progress_description = CommandProgressDescription();
    }
  }
  return progress_description;
}

}  // namespace p11
