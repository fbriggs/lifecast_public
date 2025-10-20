// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <cstdlib>
#include <cstddef>
#include <cstdint>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "source/logger.h"

#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif

namespace p11 { namespace math {

constexpr double kRadToDeg = 180.0 / M_PI;
constexpr double kDegToRad = M_PI / 180.0;

inline float randUnif() { return float(rand()) / float(RAND_MAX); }

template <typename T>
inline T clamp(const T& x, const T& a, const T& b)
{
  return x < a ? a : x > b ? b : x;
}

template <typename T>
T percentile(const std::vector<T>& v, float frac)
{
  if (v.empty()) { XPLINFO << "WARNING: attempted to get percentile from empty vector"; return 0; }
  frac = std::clamp(frac, 0.0f, 1.0f);
  std::vector<T> v_copy(v);
  std::sort(v_copy.begin(), v_copy.end());
  int idx = std::min<int>(v_copy.size() - 1, std::floor(frac * v_copy.size()));
  return v_copy[idx];
}


inline Eigen::Vector3d randomUnitVec()
{
  const double r1 = math::randUnif();
  const double r2 = math::randUnif();
  const double x = 2.0 * cos(2.0 * M_PI * r1) * std::sqrt(r2 * (1.0 - r2));
  const double y = 2.0 * sin(2.0 * M_PI * r1) * std::sqrt(r2 * (1.0 - r2));
  const double z = 1.0 - 2.0 * r2;
  return Eigen::Vector3d(x, y, z);
}

template <typename T>
T median(std::vector<T> v) {
  XCHECK(!v.empty());

  std::sort(v.begin(), v.end());
  return (v.size() % 2 == 0) ? (v[v.size()/2 - 1] + v[v.size()/2]) / 2 : v[v.size()/2];
}

}}  // namespace p11::math
