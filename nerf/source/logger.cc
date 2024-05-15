// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "logger.h"

#include <ostream>

// For printStackTrace
//#include <csignal>
//#include <cstring>
//#include <execinfo.h>
//#include <unistd.h>

bool p11_xpl_include_dev_info = true;

namespace p11 { namespace xpl {

Logger<std::ostream, true, LogLevel> stdoutLogger(std::cout, level::XPL_DEBUG{});

using detail::STDOUT_LOGGER;

LevelLogger<STDOUT_LOGGER, level::XPL_DEBUG> debug(stdoutLogger);
LevelLogger<STDOUT_LOGGER, level::XPL_INFO> info(stdoutLogger);
LevelLogger<STDOUT_LOGGER, level::XPL_WARN> warn(stdoutLogger);
LevelLogger<STDOUT_LOGGER, level::XPL_ERROR> error(stdoutLogger);

namespace {

// To get a full stack trace, use: bazel run -c dbg
/*
void printStackTrace(int sig) {
  void* array[10];
  size_t size;
  char** strings;
  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  write(STDERR_FILENO, "Stack trace:\n", 13);
  for (size_t i = 0; i < size; i++) {
    write(STDERR_FILENO, strings[i], strlen(strings[i]));
    write(STDERR_FILENO, "\n", 1);
  }
  free(strings);
}
*/

class TerminateHandler {
 public:
  TerminateHandler()
  {
    //std::signal(SIGABRT, printStackTrace);

    std::set_terminate([]() {
      try {
        std::rethrow_exception(std::current_exception());
      } catch (const std::exception& e) {
        XPLERROR << "Unhandled exception: " << e.what();
      } catch (std::string s) {
        XPLERROR << "Unhandled exception: " << s;
      } catch (...) {
        XPLERROR << "Unhandled exception of unknown type";
      }

      std::abort();
    });
  }
};

}  // namespace

TerminateHandler terminateHandler;

}}  // end namespace p11::xpl
