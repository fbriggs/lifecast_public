// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// A drop-in replacement for the gCHECK macros in glog
#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <exception>

// TODO: we wont need these macros if we switch to C++20 and use std::source_location
#define XCHECK(p) p11::Check(static_cast<bool>(p), #p, __FILE__, __LINE__)

#define XCHECK_EQ(a, b) p11::CheckEQ(a, b, #a, #b, __FILE__, __LINE__)
#define XCHECK_NE(a, b) p11::CheckNE(a, b, #a, #b, __FILE__, __LINE__)
#define XCHECK_LT(a, b) p11::CheckLT(a, b, #a, #b, __FILE__, __LINE__)
#define XCHECK_LE(a, b) p11::CheckLE(a, b, #a, #b, __FILE__, __LINE__)
#define XCHECK_GT(a, b) p11::CheckGT(a, b, #a, #b, __FILE__, __LINE__)
#define XCHECK_GE(a, b) p11::CheckGE(a, b, #a, #b, __FILE__, __LINE__)

namespace p11 {

struct CheckBase {
  const bool condition;
  std::ostringstream message;

  CheckBase(const bool& condition, const char* filename, const int line) : condition(condition)
  {
    if (!condition) {
      message << "\n" << filename << ":" << line << ": ";
    }
  }

  virtual ~CheckBase() noexcept(false)
  {
    if (!condition) {
      std::cerr << message.str() << std::endl;
      std::abort();
    }
  }

  template <typename S>
  CheckBase& operator<<(const S& s)
  {
    if (!condition) {
      message << s;
    }
    return *this;
  }
};

struct Check : public CheckBase {
  Check(const bool& condition, const char* expr, const char* filename, const int line)
      : CheckBase(condition, filename, line)
  {
    if (!condition) {
      message << "XCHECK(" << expr << ") FAILED: ";
    }
  }
};

#define BINARY_CHECK(OP_NAME, OP)                                                                \
  template <typename TA, typename TB>                                                            \
  struct Check##OP_NAME : public CheckBase {                                                     \
    Check##OP_NAME(                                                                              \
        const TA& a,                                                                             \
        const TB& b,                                                                             \
        const char* a_expr,                                                                      \
        const char* b_expr,                                                                      \
        const char* filename,                                                                    \
        const int line)                                                                          \
        : CheckBase(a OP b, filename, line)                                                      \
    {                                                                                            \
      if (!this->condition) {                                                                    \
        this->message << "XCHECK_" #OP_NAME "(" << a_expr << ", " << b_expr << ") FAILED: " << a \
                      << " " #OP " " << b << ", ";                                               \
      }                                                                                          \
    }                                                                                            \
  }

BINARY_CHECK(EQ, ==);
BINARY_CHECK(NE, !=);
BINARY_CHECK(LT, <);
BINARY_CHECK(LE, <=);
BINARY_CHECK(GT, >);
BINARY_CHECK(GE, >=);

#undef BINARY_CHECK

}  // end namespace p11
