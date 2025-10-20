// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#include <string>
#include <cstring>

namespace std {
  template<>
  struct char_traits<unsigned char> {
    using char_type   = unsigned char;
    using int_type    = make_unsigned_t<int>;
    using off_type    = streamoff;
    using pos_type    = streampos;

    static void     assign(char_type& r, const char_type& a) { r = a; }
    static bool     eq(char_type a, char_type b)         { return a == b; }
    static bool     lt(char_type a, char_type b)         { return a < b; }
    static int      compare(const char_type* s1, const char_type* s2, size_t n) {
      return memcmp(s1, s2, n * sizeof(char_type));
    }
    static size_t   length(const char_type* s) {
      const char_type* p = s;
      while (*p) ++p;
      return p - s;
    }
    static const char_type* find(const char_type* s, size_t n, const char_type& a) {
      for (size_t i = 0; i < n; ++i) if (s[i] == a) return s + i;
      return nullptr;
    }
    static char_type* move(char_type* s1, const char_type* s2, size_t n) {
      return static_cast<char_type*>(memmove(s1, s2, n * sizeof(char_type)));
    }
    static char_type* copy(char_type* s1, const char_type* s2, size_t n) {
      return static_cast<char_type*>(memcpy(s1, s2, n * sizeof(char_type)));
    }
    static char_type* assign(char_type* s, size_t n, char_type a) {
      memset(s, a, n * sizeof(char_type));
      return s;
    }
    static constexpr int_type  eof()           { return char_traits<char>::eof(); }
    static constexpr int_type  not_eof(int_type c) { return c; }
    static char_type to_char_type(int_type c)    { return static_cast<char_type>(c); }
    static int_type  to_int_type(char_type c)    { return static_cast<int_type>(c); }
    static bool      eq_int_type(int_type x, int_type y) { return x == y; }
  };
}
