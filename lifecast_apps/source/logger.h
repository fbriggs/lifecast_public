// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
xpl is an x-platform logger to replace glog. To make it a drop-in replacement, we also provide the
replacements for the gCHECK macros in glog.

Example basic useage:

#include "source/logger.h"
...
p11::xpl::info << "Hello world";

or to include filename and line numbers:

XPLINFO << "Hello world";

Toggle source code filenames and line numbers:
p11_xpl_include_dev_info = false;

More advanced examples in hello_logger.cc
*/

#pragma once

#include <string>
#include <mutex>
#include <ostream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <variant>
#include <type_traits>
#include <fstream>
#include <vector>
#include "check.h"

// To avoid showing users source code details, filename and line numbers can be toggled on or off with this.
extern bool p11_xpl_include_dev_info;

#define XPLDEBUG (p11_xpl_include_dev_info ? (p11::xpl::debug << __FILE__ << ":" << __LINE__ << "] ") : p11::xpl::debug << "")
#define XPLINFO  (p11_xpl_include_dev_info ? (p11::xpl::info  << __FILE__ << ":" << __LINE__ << "] ") : p11::xpl::info << "")
#define XPLWARN  (p11_xpl_include_dev_info ? (p11::xpl::warn  << __FILE__ << ":" << __LINE__ << "] ") : p11::xpl::warn << "")
#define XPLERROR (p11_xpl_include_dev_info ? (p11::xpl::error << __FILE__ << ":" << __LINE__ << "] ") : p11::xpl::error << "")

namespace p11 { namespace xpl {

class ThreadSafeStringStream {
 public:
  void clear() {
    stream_ = std::stringstream();
    incremental_stream_ = std::stringstream();
  }

  template <typename T>
  ThreadSafeStringStream& operator<<(const T& input)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_ << input;
    incremental_stream_ << input;
    return *this;
  }

  ThreadSafeStringStream& operator<<(std::ostream& (*func)(std::ostream&))
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_ << func;
    incremental_stream_ << func;
    return *this;
  }

  ThreadSafeStringStream& write(const char* input, std::streamsize size)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_.write(input, size);
    incremental_stream_.write(input, size);
    return *this;
  }

  std::string str() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return stream_.str();
  }

  // Use this to efficiently get a few lines at a time, which have not already 
  // been processed (unlike calling .str() which gets everything). This is intended
  // to enable extracting progress updates, for example.
  std::vector<std::string> getNewIncrementalLines() 
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> lines;
    std::string line;
    std::stringstream new_data;

    // Swap contents with a temporary stringstream to efficiently clear it
    new_data.swap(incremental_stream_);

    while (std::getline(new_data, line)) {
      if (!new_data.eof()) {
        // Complete line found, add it to the lines vector
        lines.push_back(line);
      } else {
        // Incomplete line found, put it back into incremental_stream_
        incremental_stream_ << line;
      }
    }
    return lines;
  }

 private:
  std::stringstream stream_;
  std::stringstream incremental_stream_;
  mutable std::mutex mutex_;
};

// type tags (extend as needed)
struct LogLevel {};
using NONE = LogLevel;

namespace level {
struct XPL_ERROR : public LogLevel {
  static constexpr const int value = 3;
  operator std::string() const { return "XPL_ERROR"; }
};

struct XPL_WARN : public XPL_ERROR {
  static constexpr const int value = 2;
  operator std::string() const { return "WARNING"; }
};

struct XPL_INFO : public XPL_WARN {
  static constexpr const int value = 1;
  operator std::string() const { return "XPL_INFO"; }
};

struct XPL_DEBUG : public XPL_INFO {
  static constexpr const int value = 0;
  operator std::string() const { return "XPL_DEBUG"; }
};
}  // namespace level

namespace detail {
using CoutType = std::basic_ostream<char, std::char_traits<char>>;
using StandardEndLine = CoutType& (*)(CoutType&);
}  // namespace detail

template <typename T, bool DYNAMIC, typename LEVEL>
class Logger {};

// strong optimization for avoiding logging overhead at runtime in builds without
// dynamic log level switching at runtime
template <typename T, typename LEVEL>
class Logger<T, false, LEVEL> {
 protected:
  std::mutex mutex;
  T& stream;
  ThreadSafeStringStream* capture_stream;
  std::ofstream log_file_stream; // Used if we activate file logging
  std::string log_file_path;

 public:
  Logger(T& stream) : stream(stream), capture_stream(nullptr) {}

  template <typename L, typename U>
  Logger& log(const U& expr)
  {
    auto lk = std::lock_guard(mutex);

    if constexpr (std::is_same<L, LEVEL>::value || L::value >= LEVEL::value) {
      stream << expr;
      if (capture_stream) (*capture_stream) << expr;
      if (log_file_stream.is_open()) log_file_stream << expr;
    }
    return *this;
  }

  template <typename L>
  Logger& log(detail::StandardEndLine manip)
  {
    auto lk = std::lock_guard(mutex);
    if constexpr (std::is_same<L, LEVEL>::value || L::value >= LEVEL::value) {
      stream << std::endl;
      if (capture_stream) (*capture_stream) << std::endl;
      if (log_file_stream.is_open()) log_file_stream << std::endl;
    }
    return *this;
  }

  void attachStreamCapture(ThreadSafeStringStream* ss) { capture_stream = ss; }
  void removeStreamCapture() { capture_stream = nullptr; }
  void attachTextFileLog(const std::string& log_path) {
    if (log_path == log_file_path && log_file_stream.is_open()) return; // already logging on this file
    if (log_file_stream.is_open()) log_file_stream.close(); 
    log_file_path = log_path;
    log_file_stream.open(log_file_path);
  }
  void stopTextFileLog() {
    log_file_path = "";
    if (log_file_stream.is_open()) log_file_stream.close();
  }

  ~Logger() {
    if (log_file_stream.is_open()) log_file_stream.close();
  }
};

template <typename T>
class Logger<T, true, LogLevel> {
 protected:
  std::mutex mutex;
  T& stream;
  ThreadSafeStringStream* capture_stream;
  std::ofstream log_file_stream; // Used if we activate file logging
  std::string log_file_path;

  std::variant<level::XPL_DEBUG, level::XPL_INFO, level::XPL_WARN, level::XPL_ERROR> level;

 public:
  template <typename L>
  Logger(T& stream, const L&) : stream(stream), capture_stream(nullptr), level(L{})
  {}

  template <typename NEW_LEVEL>
  void setLevel()
  {
    level = NEW_LEVEL{};
  }

  template <typename L, typename U>
  Logger& log(const U& expr)
  {
    auto lk = std::lock_guard(mutex);
    if (level.index() <= L::value) {
      stream << expr;
      if (capture_stream) (*capture_stream) << expr;
      if (log_file_stream.is_open()) log_file_stream << expr;
    }
    return *this;
  }

  template <typename L>
  Logger& log(detail::StandardEndLine manip)
  {
    auto lk = std::lock_guard(mutex);
    if (level.index() <= L::value) {
      stream << std::endl;
      if (capture_stream) (*capture_stream) << std::endl;
      if (log_file_stream.is_open()) log_file_stream << std::endl;
    }
    return *this;
  }

  void attachStreamCapture(ThreadSafeStringStream* ss) { capture_stream = ss; }
  void removeStreamCapture() { capture_stream = nullptr; }
  void attachTextFileLog(const std::string& log_path) {
    if (log_path == log_file_path && log_file_stream.is_open()) return; // already logging on this file
    if (log_file_stream.is_open()) log_file_stream.close(); 
    log_file_path = log_path;
    log_file_stream.open(log_file_path);
  }
  void stopTextFileLog() {
    log_file_path = "";
    if (log_file_stream.is_open()) log_file_stream.close();
  }
  ~Logger() {
    if (log_file_stream.is_open()) log_file_stream.close();
  }
};

template <typename L, typename LOG_LEVEL>
class LevelLogger {
 protected:
  L& logger;

  // This avoids the glog(level) functional style API and simplifies the user interface
  class EOLProxy {
  public:
    friend class LevelLogger;
    L* logger = nullptr;

    EOLProxy(L& l) : logger(&l) {}
    EOLProxy(EOLProxy&& rhs) : logger(rhs.logger) { rhs.logger = nullptr; }


    template <typename T>
    EOLProxy operator<<(const T& t) &&
    {
      logger->template log<LOG_LEVEL>(t);
      return std::move(*this);
    }

    EOLProxy operator<<(detail::StandardEndLine manip)
    {
      logger->template log<LOG_LEVEL>(manip);
      return std::move(*this);
    }

    ~EOLProxy()
    {
      if (logger) {
        logger->template log<LOG_LEVEL>(detail::StandardEndLine{});
      }
    }
  };

 public:
  LevelLogger(L& logger) : logger(logger) {}

  template <typename U>
  EOLProxy operator<<(const U& t)
  {
    return EOLProxy(logger) << t;
  }
};

extern Logger<std::ostream, true, LogLevel> stdoutLogger;

namespace detail {
using STDOUT_LOGGER = decltype(stdoutLogger);
}

extern LevelLogger<detail::STDOUT_LOGGER, level::XPL_DEBUG> debug;
extern LevelLogger<detail::STDOUT_LOGGER, level::XPL_INFO> info;
extern LevelLogger<detail::STDOUT_LOGGER, level::XPL_ERROR> error;
extern LevelLogger<detail::STDOUT_LOGGER, level::XPL_WARN> warn;

}}  // end namespace p11::xpl
