/*
This code is a translation of turbo_colormap.glsl from GLSL to C++. The original license of
turbo_colormap.glsl is reproduced below:

== License for turbo_colormap.glsl ==

Copyright 2019 Google LLC.
SPDX-License-Identifier: Apache-2.0

Polynomial approximation in GLSL for the Turbo colormap
Original LUT: https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f

Authors:
  Colormap Design: Anton Mikhailov (mikhailov@google.com)
  GLSL Approximation: Ruofei Du (ruofei@google.com)
*/
#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "source/util_math.h"

namespace p11 {
namespace turbo_colormap {

inline Eigen::Vector3f float01ToColor(float x)
{
  static const Eigen::Vector4f kRedVec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  static const Eigen::Vector4f kGreenVec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  static const Eigen::Vector4f kBlueVec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  static const Eigen::Vector2f kRedVec2(-152.94239396, 59.28637943);
  static const Eigen::Vector2f kGreenVec2(4.27729857, 2.82956604);
  static const Eigen::Vector2f kBlueVec2(-89.90310912, 27.34824973);

  x = p11::math::clamp<float>(x, 0.0f, 1.0f);
  const Eigen::Vector4f v4(1.0, x, x * x, x * x * x);
  const Eigen::Vector2f v2(v4.z() * v4.z(), v4.w() * v4.z());
  return Eigen::Vector3f(
      v4.dot(kRedVec4) + v2.dot(kRedVec2),
      v4.dot(kGreenVec4) + v2.dot(kGreenVec2),
      v4.dot(kBlueVec4) + v2.dot(kBlueVec2));
}

}  // namespace turbo_colormap
}  // namespace p11
