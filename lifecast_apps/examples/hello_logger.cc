// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "source/logger.h"
#include <chrono>

namespace p11 {

void testMacros()
{
  XPLINFO << "info " << 123 << " " << false << " " << 3.1415;
  XPLWARN << "warn";
  XPLERROR << "error";
  XPLDEBUG << "debug";
}

void testDynamicLogger()
{
  xpl::info << "Hello world";
  xpl::debug << "test";
  xpl::stdoutLogger.setLevel<xpl::level::XPL_ERROR>();
  xpl::debug << "test";
  xpl::error << "test";
}

void testStaticLogger()
{
  // static logger example (log level can be reconfigured at compilation
  // and during runtime, unused code paths and deactivated log code will have no runtime overhead
  // due to template elimination)
  xpl::Logger<std::ostream, false, xpl::level::XPL_ERROR> staticErrorStdoutLogger(std::cout);
  xpl::LevelLogger<decltype(staticErrorStdoutLogger), xpl::level::XPL_ERROR> error(
      staticErrorStdoutLogger);
  xpl::LevelLogger<decltype(staticErrorStdoutLogger), xpl::level::XPL_WARN> warn(
      staticErrorStdoutLogger);
  xpl::LevelLogger<decltype(staticErrorStdoutLogger), xpl::level::XPL_INFO> info(
      staticErrorStdoutLogger);
  xpl::LevelLogger<decltype(staticErrorStdoutLogger), xpl::level::XPL_DEBUG> debug(
      staticErrorStdoutLogger);

  // example: print a lot of data points to console. in release build, the following loop will be
  // optimized and removed by the optimizer without any code changes thanks to constexpr and
  // constexpr if()s code can be changed at compile time (it makes it possible to leave debug loops
  // in the code without ifdefs, etc. and no runtime overhead) this approach allows to instrument
  // performance sensitive code without runtime penalty in release code

  // use static logger in a loop
  for (auto i = 0; i < 5; ++i) {
    xpl::error << "this will appear " << i;
  }

  // below is an example of how to leave debug code in that looks like a normal code,
  // but gets removed in release build without any special code treatment
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < 100000ull; ++i) {  // <----+
    xpl::debug
        << "zomg";  //      |-- this code gets removed in release build without any ifdef magic
  }                 // <----+
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double dt = (t1 - t0).count();

  // use dynamic logger again
  xpl::stdoutLogger.setLevel<xpl::level::XPL_INFO>();
  xpl::info << "duration=" << dt;
}

}  // end namespace p11

int main()
{
  p11::testMacros();
  p11::testDynamicLogger();
  p11::testStaticLogger();
}
