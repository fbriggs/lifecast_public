// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "source/logger.h"

int main(int argc, char** argv)
{
  XPLINFO << "Hello! " << 123;

  XCHECK(true) << "multi-stream error message " << 123 << " wut";

  XCHECK_EQ(123, 100 + 23) << "nooo " << 444 << " huh?";
  XCHECK_EQ(std::string("cat"), std::string("ca") + std::string("t")) << "bad";
  XCHECK_LT(1.23, 4.56) << "fail";
  XCHECK_GT(6.66, 4.56) << "argh";
  XCHECK_NE(3, 6) << "lol";
  XCHECK_LE(77, 77) << "gaah";
  XCHECK_GE(55, 55) << "wtf";
  XCHECK(false) << "Use XCHECK(false) to terminate with an exception " << 123 + 456 << " " << false;
}
