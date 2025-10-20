// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <cstring>
#include <string>
#include <vector>

namespace p11 { namespace string {
inline std::string intToZeroPad(const int x, const int len)
{
  std::string s = std::to_string(x);
  while (s.size() < len) {
    s = "0" + s;
  }
  return s;
}

inline std::vector<std::string> split(const std::string& s, const char delim)
{
  std::vector<std::string> tokens;
  std::string part = "";
  for (size_t i = 0; i < s.length(); ++i) {
    if (s[i] == delim) {
      tokens.push_back(part);
      part = "";
    } else {
      part += s[i];
    }
  }
  tokens.push_back(part);
  return tokens;
}

template <typename TDelim>  // could be string or character, as long as it has + operator
inline std::string join(const std::vector<std::string>& strs, const TDelim& delim)
{
  std::string s;
  for (int i = 0; i < strs.size(); ++i) {
    s += strs[i];
    if (i != strs.size() - 1) {
      s += delim;
    }
  }
  return s;
}

// Returns true if the string src begins with prefix
inline bool beginsWith(const std::string& src, const std::string& prefix) {
  return src.rfind(prefix, 0) == 0;
}

// Returns true if the string src ends with suffix
inline bool endsWith(const std::string& src, const std::string& suffix) {
  return src.rfind(suffix) == src.size() - suffix.size();
}

// strncpy with null termination
inline char* copyBuffer(char* dest, const char* source, int dest_size) {
  std::strncpy(dest, source, dest_size - 1);
  dest[dest_size - 1] = 0;
  return dest;
}


inline std::string replaceAll(std::string s, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}


}}  // namespace p11::string
